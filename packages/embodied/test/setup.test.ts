import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { SPECS } from "../src/infra/check.ts";
import {
	AS,
	type Axes,
	type Cell,
	cellFlags,
	gridLines,
	gridMove,
	OS,
	PAPERS,
	papersFor,
	pickEntries,
	pickerLines,
	pickMatches,
	supported,
	supportedCells,
	VISER,
	WORKS,
	worksIn,
} from "../src/infra/setup/axes.ts";
import setup, {
	BUDGET,
	COMMAND,
	FINETUNED,
	GRID,
	KINDS,
	NO_MODEL,
	NO_PAPER,
	NOTICE,
	NOTICE_ENTRY,
	PANELS,
	PAPER,
	PKG,
	ROBOTS,
	RUN,
	RUN_HOW,
	runDeps,
} from "../src/infra/setup/index.ts";
import { parseRequest } from "../src/infra/setup/request.ts";
import {
	DASHBOARD_PORT,
	episodeFlags,
	experimentLabel,
	freePort,
	gpus,
	headlessCommand,
	panelFlags,
	panelUrls,
	RECORD,
	readExperiment,
	runCommand,
	scanExperiments,
	tunnelHint,
} from "../src/infra/setup/run.ts";

type Handler = (event: unknown, ctx: unknown) => unknown;
/** `keys`: the raw terminal input each custom component (the grid, then the paper picker) receives, in order; enter picks the cursor's entry. */
type Answers = { select?: string[]; input?: (string | undefined)[]; confirm?: boolean[]; keys?: string[][] };
type Component = { render(width: number): string[]; handleInput?(data: string): void };
type Factory = (
	tui: unknown,
	theme: unknown,
	kb: unknown,
	done: (r: unknown) => void,
) => Component | Promise<Component>;
/** pi's default select keybindings, as the real KeybindingsManager would match them. */
const SELECT_KEYS: Record<string, string[]> = {
	"tui.select.up": ["\x1b[A"],
	"tui.select.down": ["\x1b[B"],
	"tui.select.confirm": ["\r"],
	"tui.select.cancel": ["\x1b"],
};

/** One pi runtime with the setup extension and a faux UI answering from `answers`. */
function fakePi(
	o: {
		robot?: boolean;
		sessionDir?: string;
		answers?: Answers;
		cwd?: string;
		/** ctx.mode: the grid is a custom component in "tui", a select elsewhere. */
		mode?: string;
		model?: { provider: string; id: string };
		available?: { provider: string; id: string }[];
	} = {},
) {
	const handlers = new Map<string, Handler[]>();
	const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
	const entries: { type: string; customType: string; data: unknown }[] = [];
	const sent: { text: string; options: unknown }[] = [];
	const notes: string[] = [];
	const asked: string[] = [];
	const confirms: string[] = [];
	const offered: Record<string, string[]> = {};
	/** Every render of a grid: the first (its title line is what was asked), then one per key. */
	const grids: string[][] = [];
	const a = {
		select: [...(o.answers?.select ?? [])],
		input: [...(o.answers?.input ?? [])],
		confirm: [...(o.answers?.confirm ?? [])],
		keys: [...(o.answers?.keys ?? [])],
	};
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerCommand: (name: string, c: { handler: (args: string, ctx: unknown) => Promise<void> }) =>
			commands.set(name, c),
		getCommands: () => [
			...[...commands.keys()].map((name) => ({
				name,
				source: "extension",
				sourceInfo: { path: "", source: "../../pi/packages/embodied", scope: "user", origin: "package" },
			})),
			...(o.robot ? [{ name: "robot-task", source: "extension" }] : []),
		],
		appendEntry: (customType: string, data: unknown) => entries.push({ type: "custom", customType, data }),
		sendUserMessage: (text: string, options: unknown) => sent.push({ text, options }),
	} as unknown as ExtensionAPI;
	const ctx = {
		mode: o.mode ?? "tui",
		hasUI: true,
		cwd: o.cwd ?? tmpdir(),
		model: o.model ?? { provider: "anthropic", id: "claude-opus-5" },
		modelRegistry: { getAvailable: () => o.available ?? [{ provider: "openai", id: "gpt-6" }] },
		sessionManager: { getEntries: () => entries, getSessionDir: () => o.sessionDir ?? "/nonexistent" },
		isIdle: () => true,
		ui: {
			notify: (m: string) => notes.push(m),
			select: async (title: string, options: string[]) => {
				asked.push(title);
				offered[title] = options;
				const want = a.select.shift();
				return options.find((x) => want !== undefined && x.startsWith(want));
			},
			input: async (title: string) => {
				asked.push(title);
				return a.input.shift();
			},
			confirm: async (title: string, message: string) => {
				asked.push(title);
				confirms.push(message);
				return a.confirm.shift() ?? false;
			},
			custom: async (factory: Factory) => {
				let result: unknown;
				const theme = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
				const kb = { matches: (data: string, id: string) => SELECT_KEYS[id]?.includes(data) ?? false };
				const c = await factory({}, theme, kb, (r: unknown) => {
					result = r;
				});
				const first = c.render(80);
				asked.push(first[0]);
				grids.push(first);
				for (const k of a.keys.shift() ?? []) {
					c.handleInput?.(k);
					grids.push(c.render(80));
				}
				return result;
			},
		},
	};
	setup(pi);
	return {
		ctx,
		entries,
		sent,
		notes,
		asked,
		confirms,
		offered,
		grids,
		commands,
		start: async () => {
			for (const fn of handlers.get("session_start") ?? [])
				await fn({ type: "session_start", reason: "startup" }, ctx);
		},
		run: (args = "") => commands.get(COMMAND)?.handler(args, ctx) ?? Promise.resolve(),
	};
}

test("the notice shows once per session and records a session entry", async () => {
	const p = fakePi();
	await p.start();
	assert.deepEqual(p.notes, [NOTICE]);
	assert.equal(p.entries[0]?.customType, NOTICE_ENTRY);
	await p.start();
	assert.equal(p.notes.length, 1);
});

test("another session of the project that recorded the notice suppresses it", async () => {
	const dir = mkdtempSync(join(tmpdir(), "pi-embodied-sessions-"));
	writeFileSync(join(dir, "old.jsonl"), `${JSON.stringify({ type: "custom", customType: NOTICE_ENTRY, data: {} })}\n`);
	const p = fakePi({ sessionDir: dir });
	await p.start();
	assert.deepEqual(p.notes, []);
	assert.deepEqual(p.entries, []);
});

test("with a robot extension active nothing is shown, recorded or configured", async () => {
	const p = fakePi({ robot: true, answers: { select: ["sim", "LIBERO"], confirm: [true] } });
	await p.start();
	assert.deepEqual(p.entries, []);
	assert.deepEqual(p.notes, []);
	await p.run();
	assert.deepEqual(p.asked, []);
	assert.deepEqual(p.sent, []);
	assert.equal(p.notes.length, 1);
});

test("/embodied-setup writes the experiment settings and hands the install to the agent", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	const dir = join(cwd, "exp");
	mkdirSync(join(dir, ".pi"), { recursive: true });
	writeFileSync(
		join(dir, ".pi/settings.json"),
		JSON.stringify({ theme: "dark", extensions: [`${PKG}/src/maniskill/index.ts`, "/other/ext.ts"] }),
	);
	const p = fakePi({
		cwd,
		answers: {
			select: ["sim", "LIBERO-PRO", "here", "openai/gpt-6"],
			input: [dir],
			confirm: [true],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run();
	assert.deepEqual(p.asked, [
		"Simulation or real robot?",
		"Robot or benchmark",
		`${GRID} for LIBERO-PRO`,
		PAPER,
		"Where do the services run?",
		"Planner model",
		`Experiment directory (empty: ${join(cwd, "embodied-libero-pro")})`,
		"Set up pi-embodied?",
	]);
	const settings = JSON.parse(readFileSync(join(dir, ".pi/settings.json"), "utf8"));
	assert.deepEqual(settings, {
		theme: "dark",
		packages: [{ source: PKG, autoload: false, extensions: ["-src/infra/setup/index.ts"] }],
		extensions: ["/other/ext.ts", `${PKG}/src/robots/libero/index.ts`, `${PKG}/src/capabilities/dashboard/index.ts`],
		defaultProvider: "openai",
		defaultModel: "gpt-6",
	});
	assert.equal(p.sent.length, 1);
	const { text, options } = p.sent[0];
	assert.deepEqual(options, { expandPromptTemplates: true });
	assert.match(text, /^\/skill:embodied-quickstart /);
	assert.match(text, /setup\.sh libero-pro`/);
	assert.match(
		text,
		/`cd \S+\/exp && source \S+\/services\/\.venv-libero-pro\/pi-embodied\.env && pi --libero-type pro --suite libero_10 --task 0 --seed 0 --units=true --max-turns 40 --time-limit 900 --dashboard=true`/,
	);
	assert.match(text, /trust the project/);
	assert.doesNotMatch(text, /adapter|serve\.sh/, "a registry planner serves no adapter");
	// The cursor starts on the first supported cell, O1 × A1 Code here; enter takes it.
	assert.match(
		text,
		/^- setting: O1 × A1 Code \(§3\.1 Sensory Inputs × §3\.4\.1 Behavior Selection · Code\); flags `--units=true`$/m,
	);
	// The install and the experiment are separate: the agent installs and checks, the user launches.
	assert.match(text, /Do not start an episode/);
	assert.match(text, /I start episodes with it myself, afterwards/);
	// The confirmation says what happens: one confirmation, then the agent runs the whole install.
	const summary = p.confirms.at(-1) ?? "";
	assert.match(
		summary,
		/the agent runs it after this confirmation \(pi does not ask per step\), then the preflight; no episode is started/,
	);
	assert.match(summary, /^launch \(you run it afterwards\): /m);
	assert.match(
		summary,
		/^LIBERO-PRO \(sim\): O1 × A1 Code \(§3\.1 Sensory Inputs × §3\.4\.1 Behavior Selection · Code\)$/m,
	);
	// The confirmation's launch line is the agent's: the environment setup.sh writes, and the episode budget.
	assert.match(
		summary,
		/^launch \(you run it afterwards\): cd \S+ && source \S+\/\.venv-libero-pro\/pi-embodied\.env && pi .* --max-turns 40 --time-limit 900 /m,
	);
	// Only the settings it describes, not the paths in it (a checkout's path may contain any word).
	const described = summary
		.split("\n")
		.filter((line) => !/^(services at|writes|install:)/.test(line))
		.map((line) => line.replace(/(^|\s)\/\S+/g, "$1<path>"));
	assert.doesNotMatch(described.join("\n"), /approval/);
});

test("without a configured model /embodied-setup says how to add one and offers nothing", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	// pi's placeholder when no model is configured.
	const p = fakePi({
		cwd,
		model: { provider: "unknown", id: "unknown" },
		available: [],
		answers: {
			select: ["sim", "LIBERO", "here", "unknown/unknown"],
			input: [""],
			confirm: [true],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run();
	assert.deepEqual(p.asked, []);
	assert.deepEqual(p.sent, []);
	assert.deepEqual(p.notes, [NO_MODEL]);
	assert.match(NO_MODEL, /no model is configured.*\/login.*models\.json/);
	assert.throws(() => readFileSync(join(cwd, "embodied-libero/.pi/settings.json")));
});

test("the placeholder model is never offered as the planner", async () => {
	const p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		model: { provider: "unknown", id: "unknown" },
		answers: {
			select: ["sim", "LIBERO", "here", "openai/gpt-6"],
			input: [""],
			confirm: [false],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run();
	// A1 Code without a preset also offers the robot's fine-tuned adapter as the planner.
	assert.deepEqual(p.offered["Planner model"], ["openai/gpt-6", FINETUNED]);
	assert.match(p.confirms.at(-1) ?? "", /planner openai\/gpt-6/);
	assert.ok(BUDGET.includes("--max-turns 40") && BUDGET.includes("--time-limit 900"));
});

test("A2 asks about weights; a declined confirmation writes and sends nothing", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	const p = fakePi({
		cwd,
		answers: {
			select: ["sim", "RoboCasa", "here", "anthropic/claude-opus-5"],
			input: [""],
			confirm: [true, false],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run();
	// RoboCasa has no A1 Code: the cursor starts on O1 × A1 Model (its VLA, which the weights serve).
	assert.match(p.confirms.at(-1) ?? "", /^RoboCasa365 \(sim\): O1 × A1 Model /m);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --require-skills rldx; planner/m);
	assert.ok(p.asked.includes("Model weights"));
	assert.deepEqual(p.sent, []);
	assert.throws(() => readFileSync(join(cwd, "embodied-robocasa/.pi/settings.json")));
	assert.match(p.notes.at(-1) ?? "", /Nothing written/);
});

test("a remote box writes nothing locally and gives the agent the settings for the box", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	const p = fakePi({
		cwd,
		answers: {
			select: ["sim", "ManiSkill", "a remote", "fine-tuned"],
			input: ["me@gpu", "/data/pi-embodied", ""],
			confirm: [true],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run();
	assert.ok(p.offered["Planner model"]?.includes(FINETUNED), "A1 Code offers the fine-tuned adapter as the planner");
	assert.throws(() => readFileSync(join(cwd, "embodied-maniskill/.pi/settings.json")));
	const text = p.sent[0]?.text ?? "";
	assert.match(text, /ssh me@gpu/);
	assert.match(text, /\/data\/pi-embodied\/services\/setup\.sh maniskill`/);
	assert.match(text, /"\/data\/pi-embodied\/packages\/embodied\/src\/modes\/finetuned\/index\.ts"/);
	assert.match(text, /--units=true --model finetuned\/qwen3_5_2b_showharness_sim/);
	// Fine-tuned mode: the adapter download and the vLLM server are part of the handoff.
	assert.match(text, /FT_ADAPTER=qwen3_5_2b_sim \/data\/pi-embodied\/services\/setup\.sh finetuned/);
	assert.match(text, /\/data\/pi-embodied\/services\/pi_embodied_services\/finetuned\/serve\.sh/);
	assert.match(text, /services\.finetuned defaults to http:\/\/127\.0\.0\.1:8010\/v1/);
	assert.match(
		p.confirms.at(-1) ?? "",
		/the adapter download and the vLLM server; the agent runs it after this confirmation/,
	);
});

test("setup.sh parses and its dry run prints the plan without running it", async () => {
	const script = join(PKG, "../../services/setup.sh");
	execFileSync("bash", ["-n", script]);
	const home = mkdtempSync(join(tmpdir(), "pi-embodied-home-"));
	const out = execFileSync("bash", [script, "libero-pro", "--weights", "--dry-run", "--venv", join(home, "venv")], {
		env: { ...process.env, HOME: home, PI_EMBODIED_WEIGHTS: join(home, "w") },
		encoding: "utf8",
	});
	assert.match(out, /pip install .*services\\\[libero-pro\\\]/);
	assert.match(out, /liberopro-download-assets --skip-existing/);
	assert.match(out, /hf download facebook\/sam3 sam3\.pt/);
	assert.match(out, /export PI05_CHECKPOINT_PATH=/);
	assert.throws(() => readFileSync(join(home, "venv/pi-embodied.env")));
	assert.throws(() => execFileSync("bash", [script, "nope", "--dry-run"], { stdio: "pipe" }));
});

test("setup.sh libero links LIBERO's scene assets; libero-pro links its own tree for standard LIBERO too", () => {
	const script = join(PKG, "../../services/setup.sh");
	const home = mkdtempSync(join(tmpdir(), "pi-embodied-home-"));
	const dry = (...args: string[]) =>
		execFileSync("bash", [script, ...args, "--dry-run", "--venv", join(home, "v")], {
			env: { ...process.env, HOME: home, PI_EMBODIED_WEIGHTS: join(home, "w") },
			encoding: "utf8",
		});
	assert.match(dry("libero"), /\/v\/bin\/python -m pi_embodied_services\.robots\.libero\.assets$/m);
	assert.doesNotMatch(dry("libero", "--no-assets"), /robots\.libero\.assets/);
	assert.match(
		dry("libero-pro"),
		/liberopro-download-assets --skip-existing\n.*-m pi_embodied_services\.robots\.libero\.assets --from-package liberopro\.liberopro$/m,
	);
});

test("setup.sh writes absolute paths for a relative --venv and --weights-dir", () => {
	const script = join(PKG, "../../services/setup.sh");
	const home = mkdtempSync(join(tmpdir(), "pi-embodied-home-"));
	const out = execFileSync(
		"bash",
		[script, "libero", "--weights", "--dry-run", "--venv", "rel/venv", "--weights-dir=w"],
		{
			cwd: home,
			env: { ...process.env, HOME: home },
			encoding: "utf8",
		},
	);
	const here = realpathSync(home);
	assert.match(out, new RegExp(`^export PI_EMBODIED_PYTHON=${here}/rel/venv/bin/python$`, "m"));
	assert.match(out, new RegExp(`^export PI05_CHECKPOINT_PATH=${here}/w/RLinf-Pi05-LIBERO-130-fullshot-SFT$`, "m"));
	assert.match(out, new RegExp(`preflight: source ${here}/rel/venv/pi-embodied\\.env `));
	const tilde = execFileSync("bash", [script, "libero", "--dry-run", "--venv=~/v2"], {
		cwd: home,
		env: { ...process.env, HOME: home },
		encoding: "utf8",
	});
	assert.match(tilde, new RegExp(`^export PI_EMBODIED_PYTHON=${home}/v2/bin/python$`, "m"));
});

test("setup.sh writes pi-embodied.env once the venv is installed: a failed asset step leaves it and says how to resume", () => {
	const script = join(PKG, "../../services/setup.sh");
	const home = mkdtempSync(join(tmpdir(), "pi-embodied-home-"));
	const venv = join(home, "v");
	const bin = join(home, "bin");
	mkdirSync(join(venv, "bin"), { recursive: true });
	mkdirSync(bin);
	// An installed venv whose asset step fails (a mirror answering 429), and a uv whose installs succeed.
	const python = (assets: string) =>
		writeFileSync(
			join(venv, "bin/python"),
			`#!/bin/sh\ncase "$*" in -V) echo Python 3.11.0 ;; *robots.libero.assets*) ${assets} ;; esac\n`,
			{ mode: 0o755 },
		);
	python('echo "LIBERO assets: HTTP 429" >&2; exit 1');
	writeFileSync(join(bin, "uv"), "#!/bin/sh\nexit 0\n", { mode: 0o755 });
	writeFileSync(join(venv, "bin/hf"), "#!/bin/sh\nexit 0\n", { mode: 0o755 });
	const setupSh = (...args: string[]) =>
		spawnSync("bash", [script, "libero", "--venv", venv, ...args], {
			env: { ...process.env, HOME: home, PATH: `${bin}:${process.env.PATH}`, PI_EMBODIED_WEIGHTS: join(home, "w") },
			encoding: "utf8",
		});
	const failed = setupSh("--weights");
	assert.equal(failed.status, 1, failed.stderr);
	const early = readFileSync(join(venv, "pi-embodied.env"), "utf8");
	assert.match(early, new RegExp(`^export PI_EMBODIED_PYTHON=${venv}/bin/python$`, "m"));
	assert.match(early, /^export PI_EMBODIED_SERVICES=\//m);
	assert.doesNotMatch(early, /PI05_CHECKPOINT_PATH/);
	assert.match(
		failed.stderr,
		/LIBERO scene assets are not complete .*unset HF_ENDPOINT to fetch from huggingface\.co/,
	);
	assert.match(
		failed.stderr,
		/stopped \(exit 1\) after its venv was installed: \S+\/pi-embodied\.env holds the venv's environment/,
	);
	assert.match(failed.stderr, /rerun the same command to resume/);
	// The rerun gets past the assets: the file gains the later steps' lines and nothing asks to resume.
	python("exit 0");
	const ok = setupSh("--weights");
	assert.equal(ok.status, 0, ok.stderr);
	const full = readFileSync(join(venv, "pi-embodied.env"), "utf8");
	assert.match(full, /^export PI05_CHECKPOINT_PATH=/m);
	assert.equal(full.match(/^export PI_EMBODIED_PYTHON=/gm)?.length, 1);
	assert.doesNotMatch(ok.stderr, /resume/);
});

test("every robot has a setup.sh target, a check.ts spec and a /embodied-setup entry", () => {
	const script = join(PKG, "../../services/setup.sh");
	const home = mkdtempSync(join(tmpdir(), "pi-embodied-home-"));
	const robots = readdirSync(join(PKG, "src/robots"), { withFileTypes: true })
		.filter((d) => d.isDirectory())
		.map((d) => d.name);
	// piper's venv is the system python3 with its site packages: its dry run needs one on PATH.
	const python3 = spawnSync("python3", ["-V"]).status === 0;
	for (const robot of robots) {
		const target = robot.replace(/_/g, "-");
		if (target === "piper" && !python3) continue;
		const out = execFileSync("bash", [script, target, "--dry-run", "--venv", join(home, target)], {
			env: { ...process.env, HOME: home, PI_EMBODIED_WEIGHTS: join(home, "w") },
			encoding: "utf8",
		});
		assert.equal(out.match(/^export PI_EMBODIED_PYTHON=/gm)?.length, 1, `${target}: one PI_EMBODIED_PYTHON`);
		const checked = /preflight: .*check\.ts (\S+)$/m.exec(out)?.[1];
		assert.equal(checked, robot, `${target}: preflight robot`);
		assert.ok(SPECS[robot], `${robot}: check.ts spec`);
		assert.ok(
			ROBOTS.some((r) => r.extension === `src/robots/${robot}/index.ts`),
			`${robot}: offered by /embodied-setup`,
		);
	}
	// /embodied-setup's ids are setup.sh targets.
	for (const r of ROBOTS.filter((r) => python3 || r.id !== "piper"))
		execFileSync("bash", [script, r.id, "--dry-run", "--venv", join(home, r.id)], {
			env: { ...process.env, HOME: home, PI_EMBODIED_WEIGHTS: join(home, "w") },
			stdio: "pipe",
		});
});

test("setup.sh flywheel installs the LeRobot v3.0 export venv and names it for --flywheel-python", () => {
	const script = join(PKG, "../../services/setup.sh");
	const home = mkdtempSync(join(tmpdir(), "pi-embodied-home-"));
	const out = execFileSync("bash", [script, "flywheel", "--dry-run", "--venv", join(home, "fly")], {
		env: { ...process.env, HOME: home, PI_EMBODIED_WEIGHTS: join(home, "w") },
		encoding: "utf8",
	});
	assert.match(out, /pip install .*services\\\[flywheel\\\]/);
	assert.match(out, /export PI_EMBODIED_FLYWHEEL_PYTHON=/);
	// Not a robot venv: no PI_EMBODIED_PYTHON and no robot preflight.
	assert.doesNotMatch(out, /export PI_EMBODIED_PYTHON=/);
	assert.doesNotMatch(out, /preflight:/);
});

test("setup.sh behavior builds the Isaac Sim 6.1 stack unless B1K_STACK=isaac51 asks for OmniGibson's own", () => {
	const script = join(PKG, "../../services/setup.sh");
	const home = mkdtempSync(join(tmpdir(), "pi-embodied-home-"));
	const run = (env: Record<string, string>) =>
		execFileSync("bash", [script, "behavior", "--dry-run", "--venv", join(home, "b1k")], {
			env: { ...process.env, HOME: home, PI_EMBODIED_WEIGHTS: join(home, "w"), ...env },
			encoding: "utf8",
		});
	const out = run({});
	assert.match(out, /robots\/behavior\/install_isaac61\.sh \S*\/b1k \S*\/BEHAVIOR-1K/);
	assert.doesNotMatch(out, /robots\/behavior\/install\.sh /);
	assert.match(out, /export OMNIGIBSON_DATA_PATH=/);
	assert.match(run({ B1K_STACK: "isaac51" }), /robots\/behavior\/install\.sh \S*\/b1k \S*\/BEHAVIOR-1K/);
});

test("the robot menu is two-level (sim or real, then names only); what a robot needs is in the confirmation", async () => {
	const p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["real", "UR5e", "here", "openai/gpt-6"],
			input: [""],
			confirm: [false],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run();
	assert.deepEqual(p.offered["Simulation or real robot?"], Object.values(KINDS));
	const real = ROBOTS.filter((r) => r.kind === "real");
	assert.deepEqual(
		p.offered["Robot or benchmark"],
		real.map((r) => r.label),
		"the second menu lists the chosen kind only",
	);
	assert.ok(real.length >= 5 && ROBOTS.filter((r) => r.kind === "sim").length >= 10);
	for (const r of ROBOTS) {
		assert.doesNotMatch(r.label, /\b(sim|real)\b|GPU|e-stop/, `${r.id}: the menu shows a name, not requirements`);
		assert.ok(r.needs.length > 0, `${r.id}: needs is stated`);
	}
	// A sim entry is the benchmark's name alone; a real one is qualified only to tell two Frankas apart.
	for (const r of ROBOTS.filter((r) => r.kind === "sim")) assert.doesNotMatch(r.label, /\(/, `${r.id}: name only`);
	assert.deepEqual(
		ROBOTS.filter((r) => r.kind === "sim")
			.slice(0, 3)
			.map((r) => [r.label, r.task.split(" ")[1]]),
		[
			["LIBERO", "standard"],
			["LIBERO-PRO", "pro"],
			["LIBERO-plus", "plus"],
		],
		"every --libero-type is offered, the original first",
	);
	const summary = p.confirms.at(-1) ?? "";
	assert.match(summary, /^UR5e \(real\): O1 × A1 Code /);
	assert.match(summary, /^needs: UR5e with a Robotiq gripper/m);
});

const cell = (o: Cell["o"], a: Cell["a"]): Cell => ({ o, a });
const robot = (id: string) => ROBOTS.find((r) => r.id === id)!;
const ids = (ps: { id: string }[]) => ps.map((p) => p.id);

test("the grid: every robot has O1 × A2, only simulators have O3, only robots with a VLA have A1 Model, the cursor skips cells a robot lacks", () => {
	for (const r of ROBOTS) {
		// HumanCLAW is A1 Model only: its units are run by its motion model.
		const col = r.id === "humanclaw" ? "A1M" : "A2";
		assert.ok(supported(r.axes, cell("O1", col)), `${r.id}: the robot's defaults`);
		assert.equal(supported(r.axes, cell("O3", col)), r.kind === "sim", `${r.id}: ground truth is a simulator's`);
		assert.equal(
			supported(r.axes, cell("O1", "A1M")),
			/^(libero|robocasa|robotwin|humanclaw|franka|dual-franka)/.test(r.id),
			`${r.id}: A1 Model needs a learned executor`,
		);
		for (const o of OS) for (const a of AS) assert.equal(cellFlags(r.axes, cell(o, a)).includes("undefined"), false);
	}
	assert.deepEqual(supportedCells(robot("libero").axes).length, 12, "LIBERO has every cell");
	assert.equal(cellFlags(robot("libero").axes, cell("O2", "A1M")), "--require-skills pi0 --detections");
	assert.equal(cellFlags(robot("libero").axes, cell("O3", "A1C")), "--units=true --privileged");
	assert.equal(cellFlags(robot("libero").axes, cell("O1", "A2")), "");
	assert.equal(cellFlags(robot("franka").axes, cell("O2", "A1M")), "--vla --segment");
	// RoboCasa: no A1 Code. Left of A1 Model is nothing, right is A2, down is O2 × A1 Model.
	const rc = robot("robocasa").axes;
	assert.deepEqual(supportedCells(rc)[0], cell("O1", "A1M"));
	assert.deepEqual(gridMove(rc, cell("O1", "A1M"), "left"), cell("O1", "A1M"));
	assert.deepEqual(gridMove(rc, cell("O1", "A1M"), "right"), cell("O1", "A2"));
	assert.deepEqual(gridMove(rc, cell("O1", "A1M"), "down"), cell("O2", "A1M"));
	// HumanCLAW: A1 Model only, O1 and O3. Down from O1 skips O2; right goes nowhere.
	const hc = robot("humanclaw").axes;
	assert.deepEqual(supportedCells(hc), [cell("O1", "A1M"), cell("O3", "A1M")]);
	assert.deepEqual(gridMove(hc, cell("O1", "A1M"), "down"), cell("O3", "A1M"));
	assert.deepEqual(gridMove(hc, cell("O1", "A1M"), "right"), cell("O1", "A1M"));
	assert.equal(cellFlags(hc, cell("O1", "A1M")), "", "its task flags carry the paper's mode");
	// A real arm without a VLA: O2 is its own --segment, no O3, no A1 Model.
	assert.equal(cellFlags(robot("ur5e").axes, cell("O2", "A3")), "--code=true --segment");
	assert.deepEqual(gridMove(robot("ur5e").axes, cell("O1", "A1C"), "right"), cell("O1", "A2"));
});

test("the grid renders 3 × 4 with A1's two sub-columns under one head, the cursor marked, missing cells dashed, the survey's count under it", () => {
	const lines = gridLines("T", robot("robocasa").axes, cell("O2", "A2"));
	assert.equal(lines[0], "T");
	assert.ok(
		lines.some((l) => /^\s+A1 Behavior Selection\s+A2 Tool\s+A3 Code\s*$/.test(l)),
		"the column heads",
	);
	assert.ok(
		lines.some((l) => /^\s+Code\s+Model\s+Orchestration\s+Synthesis\s*$/.test(l)),
		"A1's sub-columns",
	);
	const rows = lines.filter((l) => /^O[123] (Sensory|Perception|Privileged)/.test(l));
	assert.equal(rows.length, 3);
	assert.match(rows[0], /^O1 Sensory Inputs\s+–\s+\[ \]\s+\[ \]\s+\[ \]\s*$/);
	assert.match(rows[1], /^O2 Perception Tools\s+–\s+\[ \]\s+\[●\]\s+\[ \]\s*$/);
	assert.match(
		lines.find((l) => l.startsWith("O2 × A2")) ?? "",
		/^O2 × A2: §3\.2 Perception Tools × §3\.5 Tool Orchestration$/,
	);
	assert.ok(lines.includes("  Decisions use supplied semantic or geometric perception tools or outputs."));
	assert.ok(lines.includes("  flags: --detections"));
	assert.match(
		lines.find((l) => l.startsWith("  works the survey files here")) ?? "",
		/^ {2}works the survey files here: \d\d+, 1 ported$/,
	);
	for (const l of gridLines("T", robot("libero").axes, cell("O1", "A1C"))) assert.ok([...l].length <= 120, l);
});

test("the grid takes the keys: arrows and vi keys move, enter picks, escape cancels", async () => {
	// Every cell has works, so the picker always follows the grid; its enter takes "none".
	const run = (keys: string[]) =>
		fakePi({
			cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
			answers: {
				select: ["sim", "LIBERO", "here", "openai/gpt-6"],
				input: [""],
				confirm: [false],
				keys: [keys, ["\r"]],
			},
		});
	// down, right, right: O1 × A1 Code -> O2 × A1 Code -> O2 × A1 Model -> O2 × A2.
	let p = run(["\x1b[B", "l", "l", "\r"]);
	await p.run();
	assert.match(p.confirms.at(-1) ?? "", /^LIBERO \(sim\): O2 × A2 /m);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --detections; planner openai\/gpt-6$/m);
	assert.ok(p.grids[3].some((l) => /^O2 Perception Tools\s+\[ \]\s+\[ \]\s+\[●\]/.test(l)));
	// Escape: nothing more is asked.
	p = run(["j", "\x1b"]);
	await p.run();
	assert.deepEqual(p.asked, ["Simulation or real robot?", "Robot or benchmark", `${GRID} for LIBERO`]);
	assert.deepEqual(p.sent, []);
	// O3 × A3 with a registry planner: the privileged code setting; no fine-tuned planner offered outside A1 Code.
	p = run(["j", "j", "l", "l", "l", "\r"]);
	await p.run();
	assert.match(p.confirms.at(-1) ?? "", /^flags: --code=true --privileged; planner/m);
	assert.deepEqual(p.offered["Planner model"], ["anthropic/claude-opus-5", "openai/gpt-6"]);
	// O1 × A1 Model on LIBERO: the VLA skill is required; the fine-tuned planner is A1 Code's only.
	p = run(["l", "\r"]);
	await p.run();
	assert.match(p.confirms.at(-1) ?? "", /^flags: --require-skills pi0; planner/m);
	assert.ok(p.asked.includes("Model weights"), "A1 Model's VLA comes with the weights");
	assert.deepEqual(p.offered["Planner model"], ["anthropic/claude-opus-5", "openai/gpt-6"]);
});

test("outside the TUI the grid and the picker are selects", async () => {
	const p = fakePi({
		mode: "rpc",
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "HumanCLAW", "O3 × A1 Model", "here", "openai/gpt-6"],
			input: [""],
			confirm: [false],
		},
	});
	await p.run();
	assert.deepEqual(p.offered[`${GRID} for HumanCLAW`], [
		"O1 × A1 Model: Sensory Inputs · Behavior Selection · Model",
		"O3 × A1 Model: Privileged State · Behavior Selection · Model",
	]);
	assert.equal(p.grids.length, 0);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --privileged; planner/m);
	// The picker: none, then the cell's ported settings only (the survey's other works are not offered); answered by its text.
	const q = fakePi({
		mode: "rpc",
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "LIBERO", "O1 × A1 Code", "showharness", "here", "openai/gpt-6"],
			input: [""],
			confirm: [false],
		},
	});
	await q.run();
	assert.deepEqual(q.offered[PAPER], [
		NO_PAPER,
		"showharness: Show-Harness, action units with image feedback, no VLA (--preset showharness)",
	]);
	assert.match(q.confirms.at(-1) ?? "", /^flags: --preset showharness; planner/m);
});

test("the work catalog: the survey's policy works per cell, every ported setting's work filed in its cell", () => {
	assert.ok(WORKS.length > 400);
	const cells = new Set(WORKS.flatMap((w) => w.cells));
	assert.equal(cells.size, 12);
	for (const o of OS) for (const a of AS) assert.ok(cells.has(`${o}/${a}`), `${o}/${a}`);
	for (const w of WORKS) assert.ok(w.id && w.name && w.title, w.id);
	for (const paper of PAPERS) {
		const w = WORKS.find((x) => x.id === paper.work);
		assert.ok(w, `${paper.id}: ${paper.work} is in works.json`);
		assert.ok(
			w.cells.includes(`${paper.cell.o}/${paper.cell.a}`),
			`${paper.id}: filed in ${paper.cell.o}/${paper.cell.a}, not ${w.cells}`,
		);
		assert.ok(
			ROBOTS.some((r) => papersFor(r, paper.cell).includes(paper)),
			`${paper.id}: some robot has its cell`,
		);
	}
	assert.deepEqual(ids(papersFor(robot("libero-pro"), cell("O1", "A1C"))), ["showharness"]);
	assert.deepEqual(ids(papersFor(robot("libero"), cell("O2", "A1C"))), ["motormind"]);
	assert.deepEqual(ids(papersFor(robot("maniskill"), cell("O2", "A1C"))), [], "MotorMind is LIBERO's");
	assert.deepEqual(ids(papersFor(robot("humanclaw"), cell("O1", "A1M"))), ["humanclaw"]);
	assert.deepEqual(ids(papersFor(robot("libero"), cell("O1", "A2"))), ["rpent"]);
	assert.deepEqual(ids(papersFor(robot("libero"), cell("O2", "A2"))), ["openeta"]);
	assert.deepEqual(ids(papersFor(robot("maniskill"), cell("O3", "A3"))), ["capx-S1"]);
	assert.deepEqual(ids(papersFor(robot("maniskill"), cell("O2", "A3"))), [
		"capx-S2",
		"capx-S3",
		"capx-S4",
		"capx-M1",
		"capx-M2",
		"capx-M3",
		"capx-M4",
	]);
	assert.deepEqual(papersFor(robot("maniskill"), cell("O1", "A3")), [], "CaP-X's API supplies perception: O2");
	assert.deepEqual(papersFor(robot("ur5e"), cell("O3", "A3")), [], "a real arm has no O3");
	assert.deepEqual(papersFor(robot("robocasa"), cell("O1", "A1C")), [], "RoboCasa has no A1 Code, so no Show-Harness");
	assert.ok(worksIn(cell("O1", "A2")).some((w) => w.name === "Reflective Planning"));
	assert.ok(
		worksIn(cell("O1", "A2")).some((w) => w.id === "harnessvla"),
		"RPent is a work of the cell too",
	);
});

test("the picker: none, then the cell's ported settings only; typing filters; a window follows the cursor", () => {
	const c = cell("O2", "A3");
	const entries = pickEntries(papersFor(robot("libero"), c), NO_PAPER);
	assert.deepEqual(ids(entries), [
		"none",
		"capx-S2",
		"capx-S3",
		"capx-S4",
		"capx-M1",
		"capx-M2",
		"capx-M3",
		"capx-M4",
	]);
	assert.ok(worksIn(c).length > 70, "the survey files many more works there");
	assert.ok(!entries.some((e) => /not ported/.test(e.text)), "which the picker does not offer");
	assert.deepEqual(ids(pickMatches(entries, "capx-m")), ["capx-M1", "capx-M2", "capx-M3", "capx-M4"]);
	assert.deepEqual(ids(pickMatches(entries, "low-level")), ["capx-S3", "capx-S4", "capx-M4"], "the note matches too");
	assert.deepEqual(pickMatches(entries, "zzzz"), []);
	const one = pickerLines("T", entries, "capx-s3", 0);
	assert.ok(one.includes("→ capx-S3: CaP-X S3, one program, low-level API with examples (--tier S3)"));
	assert.equal(one.filter((l) => /^(→ | {2})capx/.test(l)).length, 1, "one entry; the filter line is not an entry");
	assert.match(one[1], /^ {2}type to filter: capx-s3_\s+1 of 8$/);
	// The window: 12 entries around the cursor, the rest counted above and below (a long list).
	const many = Array.from({ length: 30 }, (_, i) => ({ id: `w${i}`, text: `w${i}: work ${i}` }));
	const top = pickerLines("T", many, "", 0);
	assert.equal(top[0], "T");
	assert.equal(top.filter((l) => /^(→ | {2})w\d/.test(l)).length, 12);
	assert.ok(top.some((l) => /^ {2}… 18 below$/.test(l)));
	assert.ok(!top.some((l) => /above/.test(l)));
	const mid = pickerLines("T", many, "", 20);
	assert.ok(mid.some((l) => /^ {2}… 14 above$/.test(l)));
	assert.ok(mid.some((l) => /^ {2}… 4 below$/.test(l)));
	assert.ok(mid.includes("→ w20: work 20"));
});

test("a ported work replaces the cell's flags with its preset or tier; a cell without one skips the picker", async () => {
	// O3 × A3 on LIBERO: typing "cap" filters the picker to CaP-X S1 (and the survey's works that mention it); enter takes the first.
	const p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "LIBERO", "here", "openai/gpt-6"],
			input: [""],
			confirm: [true],
			keys: [
				["j", "j", "l", "l", "l", "\r"],
				["c", "a", "p", "x", "-", "\r"],
			],
		},
	});
	await p.run();
	assert.ok(p.asked.includes(PAPER));
	const picker = p.grids[p.grids.length - 1];
	assert.ok(picker[1].startsWith("  type to filter: capx-_"), "the filter line");
	assert.ok(picker.includes("→ capx-S1: CaP-X S1, one program, high-level API, simulator ground truth (--tier S1)"));
	const text = p.sent[0]?.text ?? "";
	assert.match(
		text,
		/^- setting: O3 × A3 \(§3\.3 Privileged State × §3\.6 Code Synthesis\), CaP-X S1's setting; flags `--tier S1`$/m,
	);
	assert.match(text, /pi --libero-type standard --suite libero_10 --task 0 --seed 0 --tier S1 --max-turns 40/);
	assert.doesNotMatch(text, /--code=true|--privileged/);
	// A preset in A1 Code is the paper's planner setting: no fine-tuned adapter is offered with it. The filter
	// also matches the name; a filter nothing matches is erased with backspace; vi keys are typed text here.
	const q = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "LIBERO", "here", "openai/gpt-6"],
			input: [""],
			confirm: [false],
			keys: [["\r"], ["z", "z", "z", "z", "\x7f", "\x7f", "\x7f", "\x7f", "s", "h", "o", "w", "-", "h", "\r"]],
		},
	});
	await q.run();
	const after = q.grids.slice(1);
	assert.ok(
		after.some((g) => g.some((l) => /nothing matches/.test(l))),
		"zzzz matches no work",
	);
	assert.ok(
		after.at(-1)?.some((l) => /^→ showharness: Show-Harness/.test(l)),
		"show-h matches the name",
	);
	assert.deepEqual(q.offered["Planner model"], ["anthropic/claude-opus-5", "openai/gpt-6"]);
	assert.match(q.confirms.at(-1) ?? "", /^flags: --preset showharness; planner/m);
	// A cell with no ported setting (O1 × A3 on ManiSkill) skips the picker.
	const r = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "ManiSkill", "here", "openai/gpt-6"],
			input: [""],
			confirm: [false],
			keys: [["l", "l", "l", "\r"]],
		},
	});
	await r.run();
	assert.equal(r.grids.length, 5, "the grid's renders only: the first and one per key");
	assert.ok(!r.asked.includes(PAPER));
	assert.match(r.confirms.at(-1) ?? "", /^ManiSkill 3 \(sim\): O1 × A3 /m);
	assert.match(r.confirms.at(-1) ?? "", /^flags: --code=true; planner/m);
});

test("O2 tells the agent about the SAM3 server; the fine-tuned planner adds its extension and flags", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	const p = fakePi({
		cwd,
		answers: {
			select: ["sim", "ManiSkill", "here", "fine-tuned"],
			input: [""],
			confirm: [true],
			keys: [["j", "\r"], ["\r"]],
		},
	});
	await p.run();
	const text = p.sent[0]?.text ?? "";
	assert.match(text, /^- setting: O2 × A1 Code /m);
	assert.match(text, /^- perception \(O2\): a SAM3 server at the deployment's services\.sam3 must be running/m);
	assert.match(text, /--units=true --detections --model finetuned\/qwen3_5_2b_showharness_sim --max-turns 40/);
	assert.match(text, /FT_ADAPTER=qwen3_5_2b_sim/);
	const settings = JSON.parse(readFileSync(join(cwd, "embodied-maniskill/.pi/settings.json"), "utf8"));
	assert.ok(settings.extensions.includes(`${PKG}/src/modes/finetuned/index.ts`), "the fine-tuned extension is loaded");
	assert.equal(settings.defaultModel, undefined, "the adapter is the planner: no registry model is set");
	const axes: Axes = robot("maniskill").axes;
	assert.equal(cellFlags(axes, cell("O2", "A1C")), "--units=true --detections");
});

test("a request settles what it names: robot, setting or work, levels, where, planner", () => {
	const models = ["anthropic/claude-opus-5", "openai/gpt-6-astra"];
	let r = parseRequest("在 LIBERO-pro 里复现 RPent 的实验", ROBOTS, models);
	assert.deepEqual(
		r.robots?.map((x) => x.id),
		["libero-pro"],
	);
	assert.equal(r.kind, "sim");
	assert.equal(r.paper?.id, "rpent");
	assert.equal(r.work, undefined);
	assert.deepEqual(r.settled, ["robot LIBERO-PRO", "RPent's setting (O1 × A2, --preset rpent)"]);
	// A name several robots share: all of them. Levels, a remote host.
	r = parseRequest("franka, units, privileged, on me@gpu-box", ROBOTS, models);
	assert.deepEqual(
		r.robots?.map((x) => x.id),
		["franka", "franka-polymetis", "dual-franka"],
	);
	assert.equal(r.kind, "real");
	assert.deepEqual([r.o, r.a, r.where, r.host], ["O3", "A1C", "remote", "me@gpu-box"]);
	// A work of the survey, a planner by its id.
	r = parseRequest("Code as Policies on maniskill with gpt-6-astra, here", ROBOTS, models);
	assert.equal(r.work?.id, "grid66");
	assert.deepEqual(
		r.robots?.map((x) => x.id),
		["maniskill"],
	);
	assert.equal(r.model, "openai/gpt-6-astra");
	assert.equal(r.where, "here");
	assert.equal(r.a, undefined, "a work places the cell; 'code' is not read as A3");
	// A tier by its short name, in either spelling; a planner as provider/id.
	assert.equal(parseRequest("cap-x s2 on libero", ROBOTS, models).paper?.id, "capx-S2");
	assert.equal(parseRequest("capx-M3", ROBOTS, models).paper?.id, "capx-M3");
	assert.equal(parseRequest("anthropic/claude-opus-5", ROBOTS, models).model, "anthropic/claude-opus-5");
	// Exact beats partial: LIBERO is the standard one; "real" is a kind, not the work REAL; nothing is nothing.
	assert.deepEqual(
		parseRequest("LIBERO", ROBOTS, models).robots?.map((x) => x.id),
		["libero"],
	);
	assert.equal(parseRequest("a real robot", ROBOTS, models).work, undefined);
	assert.equal(parseRequest("a real robot", ROBOTS, models).kind, "real");
	assert.deepEqual(parseRequest("", ROBOTS, models), { settled: [], unknown: [] });
	assert.deepEqual(parseRequest("a1 model sam3", ROBOTS, models).settled, ["O2 × A1M"]);
});

test("/embodied-setup <request>: what the request settles is not asked; an ambiguous robot narrows its menu", async () => {
	// Robot and setting named: only the services, the planner and the directory are asked.
	let p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6"], input: [""], confirm: [false] },
	});
	await p.run("在 LIBERO-pro 里复现 RPent 的实验");
	assert.deepEqual(p.asked.slice(0, 2), ["Where do the services run?", "Planner model"]);
	assert.equal(p.grids.length, 0, "no grid, no picker");
	let summary = p.confirms.at(-1) ?? "";
	assert.match(summary, /^from your request: robot LIBERO-PRO; RPent's setting \(O1 × A2, --preset rpent\)$/m);
	assert.match(summary, /^LIBERO-PRO \(sim\): O1 × A2 .*, RPent's setting$/m);
	assert.match(summary, /^flags: --preset rpent; planner openai\/gpt-6$/m);
	// Everything named: only the directory and the confirmation remain.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { input: [""], confirm: [false] },
	});
	await p.run("reproduce Show-Harness on metaworld here with openai/gpt-6");
	assert.deepEqual(
		p.asked.map((a) => a.replace(/\(empty: .*\)/, "(empty: ...)")),
		["Experiment directory (empty: ...)", "Set up pi-embodied?"],
	);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --preset showharness; planner openai\/gpt-6$/m);
	// "franka" is three robots: the menu offers those three only; the rest is the usual flow.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["Dual Franka", "here", "openai/gpt-6"],
			input: [""],
			confirm: [false],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run("franka");
	assert.deepEqual(p.offered["Robot or benchmark"], ["Franka (RLinf)", "Franka (Polymetis)", "Dual Franka"]);
	assert.ok(!p.asked.includes("Simulation or real robot?"), "the kind follows the robots");
	assert.match(p.confirms.at(-1) ?? "", /^Dual Franka \(real\): O1 × A1 Code /m);
	// Levels named: the grid is skipped.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6"], input: [""], confirm: [false], keys: [["\r"]] },
	});
	await p.run("libero units privileged");
	assert.equal(p.grids.length, 0, "no grid; and O3 × A1 Code has no ported setting, so no picker");
	assert.ok(!p.asked.includes(PAPER));
	summary = p.confirms.at(-1) ?? "";
	assert.match(summary, /^LIBERO \(sim\): O3 × A1 Code /m);
	assert.match(summary, /^flags: --units=true --privileged; planner/m);
	// A remote host named: only the checkout on the box is asked for.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["openai/gpt-6"], input: ["/data/pi-embodied", ""], confirm: [false] },
	});
	await p.run("rpent on libero, ssh me@gpu");
	assert.ok(!p.asked.includes("ssh host"));
	assert.ok(p.asked.includes("pi-embodied checkout on the box"));
	assert.match(p.confirms.at(-1) ?? "", /^services on me@gpu:\/data\/pi-embodied\/services$/m);
});

test("/embodied-setup <request> with a work of the survey: its ported settings narrow the cell and the picker; one that is not ported is a notice", async () => {
	// CaP-X (ported as its tiers) sits in O2 × A3 and O3 × A3; ManiSkill has both: the cell is asked, the picker lists the tiers there.
	let p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["O2 × A3", "here", "openai/gpt-6"], input: [""], confirm: [false], keys: [["\x1b[B", "\r"]] },
	});
	await p.run("CaP-X on maniskill");
	assert.deepEqual(
		p.notes.filter((n) => !/^Nothing written/.test(n)),
		[],
		"no warning",
	);
	assert.deepEqual(p.offered["Cell for CaP-X"], [
		"O2 × A3: Perception Tools · Code Synthesis",
		"O3 × A3: Privileged State · Code Synthesis",
	]);
	const picker = p.grids[0];
	assert.ok(
		picker.some((l) => /capx-S2: CaP-X S2/.test(l)),
		picker.join("\n"),
	);
	assert.ok(!picker.some((l) => /capx-S1/.test(l)), "S1 is O3's");
	assert.match(p.confirms.at(-1) ?? "", /^flags: --tier S2; planner/m);
	// A real arm has no O3: CaP-X on UR5e is O2 × A3 without asking.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6"], input: [""], confirm: [false], keys: [["\r"]] },
	});
	await p.run("CaP-X on ur5e");
	assert.ok(!p.asked.some((a) => a.startsWith("Cell for")));
	assert.match(p.confirms.at(-1) ?? "", /^UR5e \(real\): O2 × A3 /m);
	// Code as Policies is in the survey but not ported: a notice names its cells, then the grid.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6"], input: [""], confirm: [false], keys: [["\x1b"]] },
	});
	await p.run("Code as Policies on LIBERO");
	assert.ok(
		p.notes.some((n) =>
			/^Code as Policies is not ported yet \(the survey files it in O2 × A3, O3 × A3\); pick a cell/.test(n),
		),
		p.notes.join("|"),
	);
	assert.ok(!p.asked.some((a) => a.startsWith("Cell for")));
	assert.equal(p.grids.length, 2, "the grid: its first render and the one after escape; no picker");
	assert.deepEqual(p.sent, []);
	// A setting the robot cannot run (MotorMind's, LIBERO's only, on ManiSkill): a warning, then the grid.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6"], input: [""], confirm: [false], keys: [["\r"], ["\r"]] },
	});
	await p.run("motormind on maniskill");
	assert.ok(
		p.notes.some((n) => /^MotorMind's setting is not on ManiSkill 3; pick a cell$/.test(n)),
		p.notes.join("|"),
	);
	assert.equal(p.grids.length, 4, "the grid and the picker of O1 × A1 Code (Show-Harness), each rendered twice");
	assert.match(p.confirms.at(-1) ?? "", /^ManiSkill 3 \(sim\): O1 × A1 Code /m);
	// Names nothing matches are called out as possibly not supported, in a notice and in the confirmation.
	const r = parseRequest("reproduce VoxPoser-2 on Aloha with units, here", ROBOTS, ["openai/gpt-6"]);
	assert.deepEqual(r.unknown, ["VoxPoser-2", "Aloha"]);
	assert.deepEqual([r.a, r.where], ["A1C", "here"]);
	assert.deepEqual(parseRequest("在 LIBERO-pro 里复现 RPent 的实验", ROBOTS, []).unknown, []);
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["sim", "LIBERO", "openai/gpt-6"], input: [""], confirm: [false], keys: [["\r"], ["\r"]] },
	});
	await p.run("reproduce VoxPoser-2 on Aloha, here");
	assert.ok(
		p.notes.some((n) =>
			/^not recognized: VoxPoser-2, Aloha; no robot, ported setting, survey work, level or planner by that name \(maybe not supported yet\)/.test(
				n,
			),
		),
		p.notes.join("|"),
	);
	assert.ok(p.asked.includes("Simulation or real robot?"), "the rest is asked");
	assert.match(p.confirms.at(-1) ?? "", /^not recognized \(maybe not supported yet\): VoxPoser-2, Aloha$/m);
});

test("/embodied-setup records the setting in the experiment directory for /embodied-run", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	// O1 × A2 on LIBERO asks about the weights (its VLA) before the confirmation.
	const p = fakePi({ cwd, answers: { select: ["here", "openai/gpt-6"], input: [""], confirm: [true, true] } });
	await p.run("在 LIBERO-pro 里复现 RPent 的实验");
	const record = JSON.parse(readFileSync(join(cwd, "embodied-libero-pro", RECORD), "utf8"));
	assert.deepEqual(record, {
		robot: "libero-pro",
		cell: { o: "O1", a: "A2" },
		paper: "rpent",
		flags: "--preset rpent",
		task: "--libero-type pro --suite libero_10 --task 0 --seed 0",
		services: record.services,
		planner: "openai/gpt-6",
		created: record.created,
	});
	assert.match(record.services, /\/services$/);
	// The directory is an experiment: its settings load LIBERO; not installed here (no venv).
	const e = readExperiment(join(cwd, "embodied-libero-pro"), ROBOTS);
	assert.equal(e?.robot.id, "libero-pro");
	assert.deepEqual(
		e?.robots.map((r) => r.id),
		["libero", "libero-pro", "libero-plus"],
	);
	assert.equal(e?.planner, "openai/gpt-6");
	assert.equal(e?.env, undefined);
	assert.equal(experimentLabel(e!), "embodied-libero-pro: LIBERO-PRO, O1 × A2, rpent (not installed)");
	assert.deepEqual(
		scanExperiments(cwd, ROBOTS).map((x) => basename(x.dir)),
		["embodied-libero-pro"],
	);
	assert.deepEqual(
		scanExperiments(join(cwd, "embodied-libero-pro"), ROBOTS).map((x) => basename(x.dir)),
		["embodied-libero-pro"],
	);
	assert.deepEqual(scanExperiments(mkdtempSync(join(tmpdir(), "pi-embodied-none-")), ROBOTS), []);
});

/** An installed experiment: settings loading a robot, a record, and the env file setup.sh would write. */
function installed(cwd: string, name: string, robot: string, record?: Partial<SetupRecordLike>) {
	const dir = join(cwd, name);
	const services = join(cwd, "services");
	mkdirSync(join(dir, ".pi"), { recursive: true });
	mkdirSync(join(services, `.venv-${robot}`), { recursive: true });
	writeFileSync(join(services, `.venv-${robot}`, "pi-embodied.env"), "export PI_EMBODIED_PYTHON=/x\n");
	const ext = ROBOTS.find((r) => r.id === robot)!.extension;
	writeFileSync(
		join(dir, ".pi/settings.json"),
		JSON.stringify({
			extensions: [`${cwd}/packages/embodied/${ext}`],
			defaultProvider: "openai",
			defaultModel: "gpt-6",
		}),
	);
	if (record)
		writeFileSync(
			join(dir, RECORD),
			JSON.stringify({
				robot,
				cell: { o: "O1", a: "A1C" },
				flags: "--units=true",
				task: "--task x --seed 0",
				services,
				...record,
			}),
		);
	return dir;
}
type SetupRecordLike = {
	robot: string;
	cell: Cell;
	paper?: string;
	flags: string;
	task: string;
	services: string;
	planner?: string;
};

test("the run module: experiments are read from their settings and record, the commands carry the env, the GPU and the budget", () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	const dir = installed(cwd, "embodied-metaworld", "metaworld", { paper: undefined });
	const e = readExperiment(dir, ROBOTS)!;
	assert.equal(e.robot.id, "metaworld");
	assert.equal(e.env, join(cwd, "services/.venv-metaworld/pi-embodied.env"));
	assert.equal(e.record?.flags, "--units=true");
	assert.equal(experimentLabel(e), "embodied-metaworld: Metaworld, O1 × A1 Code");
	const flags = episodeFlags("--task reach-v3 --seed 0", "--units=true", BUDGET);
	assert.equal(flags, `--task reach-v3 --seed 0 --units=true ${BUDGET} --dashboard=true`);
	assert.equal(panelFlags(8765), "--dashboard=true --dashboard-port 8765");
	assert.equal(panelFlags(8766, [VISER]), "--dashboard=true --dashboard-port 8766 --viser --viser-port 8080");
	assert.deepEqual(panelUrls(8765, [VISER]), [
		"dashboard http://127.0.0.1:8765/",
		"Viser 3D view http://127.0.0.1:8080/",
	]);
	assert.equal(tunnelHint([8765, 8080], {}), undefined, "not over ssh: no tunnel needed");
	assert.equal(
		tunnelHint([8765, 8080], { SSH_CONNECTION: "1.2.3.4 5 6.7.8.9 22" }, "l40s-0003"),
		"ssh -N -L 8765:127.0.0.1:8765 -L 8080:127.0.0.1:8080 l40s-0003",
	);
	assert.equal(
		runCommand(e, flags, 5),
		`cd ${dir} && source ${e.env} && CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 pi ${flags} "Solve the task."`,
	);
	assert.equal(runCommand(e, flags), `cd ${dir} && source ${e.env} && pi ${flags} "Solve the task."`);
	assert.equal(
		headlessCommand(e, flags, 5),
		`cd ${dir} && source ${e.env} && CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 pi -a -p ${flags} "Solve the task." < /dev/null`,
	);
	// No record, three LIBEROs share the extension: the one whose venv exists is the robot.
	const lib = installed(cwd, "libero-old", "libero-plus");
	const l = readExperiment(lib, ROBOTS)!;
	assert.equal(l.robot.id, "libero-plus");
	assert.equal(l.record, undefined);
	assert.equal(experimentLabel(l), "libero-old: LIBERO-plus, setting not recorded");
	// GPUs from nvidia-smi, the freest first; none when it is missing or fails.
	const smi = (stdout: string, status = 0) => (() => ({ status, stdout })) as unknown as Parameters<typeof gpus>[0];
	assert.deepEqual(gpus(smi("0, 41355, 46068\n5, 29793, 46068\n6, 29793, 46068\n")), [
		{ index: 5, freeMiB: 16275, totalMiB: 46068 },
		{ index: 6, freeMiB: 16275, totalMiB: 46068 },
		{ index: 0, freeMiB: 4713, totalMiB: 46068 },
	]);
	assert.deepEqual(gpus(smi("", 1)), []);
	assert.deepEqual(
		gpus((() => {
			throw new Error("ENOENT");
		}) as unknown as Parameters<typeof gpus>[0]),
		[],
	);
});

test("/embodied-run: the installed experiment, its recorded setting, the task, the budget; headless hands the command to the agent, print shows it", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	installed(cwd, "embodied-metaworld", "metaworld", {});
	// One experiment: nothing to pick; the setting is recorded; no GPU on this machine (none asked, none in the command); the dashboard's port is free.
	runDeps.gpus = () => [];
	runDeps.freePort = async () => 8765;
	runDeps.env = {};
	let p = fakePi({ cwd, answers: { select: [RUN_HOW.print], input: ["", ""], confirm: [true] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.ok(!p.asked.includes("Experiment"));
	assert.ok(p.asked.some((a) => a.startsWith("Task flags (empty: --task x --seed 0)")));
	assert.ok(p.asked.some((a) => a.startsWith("Episode budget (empty: --max-turns 40")));
	assert.deepEqual(p.offered["How to run?"], [RUN_HOW.here, RUN_HOW.headless, RUN_HOW.print]);
	assert.match(p.confirms.at(-1) ?? "", /^Metaworld: O1 × A1 Code; planner openai\/gpt-6$/m);
	assert.match(p.confirms.at(-1) ?? "", /^task: --task x --seed 0$/m);
	assert.ok(!p.asked.includes("GPU"));
	assert.ok(!p.asked.includes(PANELS), "Metaworld has no panel of its own");
	assert.equal(
		p.notes.at(-1),
		`cd ${cwd}/embodied-metaworld && source ${cwd}/services/.venv-metaworld/pi-embodied.env && pi --task x --seed 0 --units=true --max-turns 40 --time-limit 900 --dashboard=true --dashboard-port 8765 "Solve the task."`,
	);
	assert.match(p.confirms.at(-1) ?? "", /^panels on this machine: dashboard http:\/\/127\.0\.0\.1:8765\/$/m);
	assert.deepEqual(p.sent, []);
	// Headless: the agent gets the pi -a -p command and what to report; the task flags typed replace the recorded ones.
	p = fakePi({
		cwd,
		answers: { select: [RUN_HOW.headless], input: ["--task push-v3 --seed 3", "--max-turns 10"], confirm: [true] },
	});
	await p.commands.get(RUN)?.handler("", p.ctx);
	const text = p.sent[0]?.text ?? "";
	assert.match(text, /^\/skill:embodied-quickstart Run one episode now, in the background/);
	assert.match(
		text,
		/pi -a -p --task push-v3 --seed 3 --units=true --max-turns 10 --dashboard=true --dashboard-port 8765 "Solve the task\." < \/dev\/null`$/m,
	);
	assert.match(text, /robot_result/);
	// Outside the TUI "here" is not offered.
	p = fakePi({ cwd, mode: "rpc", answers: { select: [RUN_HOW.print], input: ["", ""], confirm: [false] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.deepEqual(p.offered["How to run?"], [RUN_HOW.headless, RUN_HOW.print]);
	assert.match(p.notes.at(-1) ?? "", /^Nothing started/);
	// No experiment here: told to set up first.
	p = fakePi({ cwd: mkdtempSync(join(tmpdir(), "pi-embodied-none-")) });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.match(p.notes.at(-1) ?? "", /^no experiment here: .*run \/embodied-setup first$/);
});

test("/embodied-run: several experiments are a menu, a request's robot picks one, an unrecorded setting is chosen as in setup, not installed is refused", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	installed(cwd, "embodied-metaworld", "metaworld", {});
	installed(cwd, "libero-old", "libero");
	mkdirSync(join(cwd, "embodied-ur5e/.pi"), { recursive: true });
	writeFileSync(
		join(cwd, "embodied-ur5e/.pi/settings.json"),
		JSON.stringify({ extensions: [`${cwd}/packages/embodied/src/robots/ur5e/index.ts`] }),
	);
	// Two GPUs: a menu, the freest first.
	runDeps.gpus = () => [
		{ index: 5, freeMiB: 16275, totalMiB: 46068 },
		{ index: 0, freeMiB: 4713, totalMiB: 46068 },
	];
	// The menu, by directory; the request names LIBERO: no menu, and its setting is not recorded, so the grid and the picker (O1 × A1 Code: Show-Harness) are asked.
	let p = fakePi({
		cwd,
		answers: {
			select: ["GPU 5", "the dashboard only", RUN_HOW.print],
			input: ["", ""],
			confirm: [true],
			keys: [["\r"], ["\x1b[B", "\r"]],
		},
	});
	await p.commands.get(RUN)?.handler("libero", p.ctx);
	assert.ok(!p.asked.includes("Experiment"));
	assert.equal(p.grids.length, 5, "the grid (two renders) and the picker (three)");
	assert.deepEqual(p.offered.GPU, [
		"GPU 5: 16275 MiB free of 46068",
		"GPU 0: 4713 MiB free of 46068",
		"any: no CUDA_VISIBLE_DEVICES",
	]);
	assert.match(p.confirms.at(-1) ?? "", /^LIBERO: O1 × A1 Code, showharness; planner openai\/gpt-6$/m);
	assert.match(p.confirms.at(-1) ?? "", /^GPU 5$/m);
	assert.match(
		p.notes.at(-1) ?? "",
		/CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 pi --libero-type standard --suite libero_10 --task 0 --seed 0 --preset showharness --max-turns 40/,
	);
	// No request: the menu lists all three, the UR5e one as not installed; choosing it is refused.
	p = fakePi({ cwd, answers: { select: ["embodied-ur5e"] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.deepEqual(p.offered.Experiment, [
		"embodied-metaworld: Metaworld, O1 × A1 Code",
		"embodied-ur5e: UR5e, setting not recorded (not installed)",
		"libero-old: LIBERO, setting not recorded",
	]);
	assert.match(
		p.notes.at(-1) ?? "",
		/^UR5e is not installed: .*\.venv-ur5e\/pi-embodied\.env is missing; run \/embodied-setup$/,
	);
	// A request's levels override a recorded setting (the grid is skipped), the picker of that cell still asks.
	p = fakePi({ cwd, answers: { select: ["any", RUN_HOW.print], input: ["", ""], confirm: [true], keys: [["\r"]] } });
	await p.commands.get(RUN)?.handler("metaworld privileged code", p.ctx);
	assert.equal(p.grids.length, 2, "no grid; the picker of O3 × A3 (CaP-X S1), answered none");
	assert.match(p.confirms.at(-1) ?? "", /^Metaworld: O3 × A3; planner/m);
	assert.match(p.confirms.at(-1) ?? "", /^GPU: any$/m);
	assert.match(p.notes.at(-1) ?? "", /&& pi --task x --seed 0 --code=true --privileged --max-turns 40/);
	runDeps.gpus = () => [];
});

test("/embodied-run panels: a work's own panel comes first when the robot serves it, the dashboard port is fixed, over ssh the tunnel is shown", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	installed(cwd, "embodied-libero", "libero", {
		cell: { o: "O3", a: "A3" },
		paper: "capx-S1",
		flags: "--tier S1",
		task: "--suite libero_10 --task 0 --seed 0",
	});
	runDeps.gpus = () => [];
	runDeps.freePort = async (start) => (start ?? DASHBOARD_PORT) + 1;
	runDeps.env = { SSH_CONNECTION: "10.0.0.2 50000 172.28.181.197 22" };
	// CaP-X's setting on LIBERO: its Viser view is offered first, with the dashboard.
	let p = fakePi({ cwd, answers: { select: ["CaP-X S1's own", RUN_HOW.print], input: ["", ""], confirm: [true] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.deepEqual(p.offered[PANELS], [
		"CaP-X S1's own: Viser 3D view (CaP-X's viser_debug: point clouds, camera frustums, EEF, grasp candidates), with the dashboard",
		"the dashboard only",
	]);
	const summary = p.confirms.at(-1) ?? "";
	assert.match(
		summary,
		/^panels on this machine: dashboard http:\/\/127\.0\.0\.1:8766\/; Viser 3D view http:\/\/127\.0\.0\.1:8080\/$/m,
	);
	assert.match(
		summary,
		/^from your own machine, first: ssh -N -L 8766:127\.0\.0\.1:8766 -L 8080:127\.0\.0\.1:8080 \S+ \(or a LocalForward/m,
	);
	assert.match(
		p.notes.at(-1) ?? "",
		/--tier S1 --max-turns 40 --time-limit 900 --dashboard=true --dashboard-port 8766 --viser --viser-port 8080 "Solve the task\."\nssh -N -L 8766/,
	);
	// The dashboard only: no --viser.
	p = fakePi({ cwd, answers: { select: ["the dashboard only", RUN_HOW.print], input: ["", ""], confirm: [true] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.match(p.notes.at(-1) ?? "", /--dashboard-port 8766 "Solve the task\."\nssh/);
	assert.doesNotMatch(p.notes.at(-1) ?? "", /--viser/);
	// A plain cell on LIBERO: the robot's own Viser view is the offer; Show-Harness has no panel of its own.
	const lib = installed(cwd, "libero-units", "libero", { paper: "showharness", flags: "--preset showharness" });
	p = fakePi({
		cwd,
		answers: { select: ["libero-units", "the robot's", RUN_HOW.print], input: ["", ""], confirm: [true] },
	});
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.equal(
		p.offered[PANELS]?.[0],
		"the robot's: Viser 3D view (CaP-X's viser_debug: point clouds, camera frustums, EEF, grasp candidates), with the dashboard",
	);
	assert.match(
		p.notes.at(-1) ?? "",
		new RegExp(`^cd ${lib} .*--preset showharness .*--viser --viser-port 8080 "Solve the task\\."\\n`),
	);
	runDeps.env = {};
	runDeps.freePort = freePort;
	// freePort itself answers with a port at or after the start.
	const port = await freePort(DASHBOARD_PORT);
	assert.ok(port >= DASHBOARD_PORT && port < DASHBOARD_PORT + 20);
});
