import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { SPECS } from "../src/infra/check.ts";
import { TYPE_OWN } from "../src/infra/setup/ask.ts";
import {
	AS,
	type Axes,
	type Cell,
	cellFlags,
	gridLines,
	gridMove,
	OS,
	PAPERS,
	paperFlag,
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
import { robotCaps, settingsFor, tierProblems } from "../src/infra/setup/caps.ts";
import { evalUsage, knownFlags, runCatalog, setupCatalog } from "../src/infra/setup/catalog.ts";
import setup, {
	agentDeps,
	BATCH_HOW,
	BUDGET,
	BUDGETS,
	COMMAND,
	FINETUNED,
	FINETUNED_AS,
	FLASH,
	GRID,
	KINDS,
	NO_MODEL,
	NO_PAPER,
	NOTICE,
	NOTICE_ENTRY,
	OTHER,
	PANELS,
	PAPER,
	PKG,
	PLANNER,
	ROBOTS,
	RUN,
	RUN_HOW,
	rec,
	runDeps,
	SETUP_NOW,
	setupDeps,
	sshHosts,
} from "../src/infra/setup/index.ts";
import {
	type Ask,
	checkRun,
	checkSetup,
	interpret,
	openQuestions,
	SETUP_QUESTIONS,
	withAnswers,
} from "../src/infra/setup/interpret.ts";
import { parseRequest } from "../src/infra/setup/request.ts";
import { modelsJsonEntry, SELFHOST, selfhostHandoff, selfhostLabel, selfhostOf } from "../src/infra/setup/selfhost.ts";

// The machine's ssh config is not the test's: no hosts, the host is typed.
setupDeps.sshHosts = () => [];

/** The self-hosted planners every planner menu ends with (before the fine-tuned adapter). */
const HOSTED = SELFHOST.map(selfhostLabel);

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
	runOptions,
	scanExperiments,
	tunnelHint,
	withRunOptions,
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
		getFlag: () => undefined,
		getCommands: () => [
			...[...commands.keys()].map((name) => ({
				name,
				source: "extension",
				sourceInfo: { path: "", source: "../../pi/packages/embodied", scope: "user", origin: "package" },
			})),
			...(o.robot ? [{ name: "robot-task", source: "extension" }] : []),
		],
		appendEntry: (customType: string, data: unknown) => entries.push({ type: "custom", customType, data }),
		registerTool: () => {},
		getActiveTools: () => [],
		setActiveTools: () => {},
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
				// The planner list is searchable: type the select answer into its filter, then enter (escape when none).
				if (first[0] === "Planner model") {
					offered[first[0]] = (c as Component & { options: string[] }).options;
					const want = a.select.shift();
					for (const ch of want ?? "") c.handleInput?.(ch);
					c.handleInput?.(want === undefined ? "\x1b" : "\r");
					return result;
				}
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
			select: ["sim", "LIBERO-PRO", "here", "openai/gpt-6", "other"],
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
		"Experiment directory",
		"Experiment directory",
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
			select: ["sim", "LIBERO", "here", "unknown/unknown", ""],
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
			select: ["sim", "LIBERO", "here", "openai/gpt-6", ""],
			confirm: [false],
			keys: [["\r"], ["\r"]],
		},
	});
	await p.run();
	// A1 Code without a preset also offers the robot's fine-tuned adapter, and LIBERO's Flash hook a replay.
	assert.deepEqual(p.offered["Planner model"], ["openai/gpt-6", ...HOSTED, FINETUNED, FLASH]);
	assert.match(p.confirms.at(-1) ?? "", /planner openai\/gpt-6/);
	assert.ok(BUDGET.includes("--max-turns 40") && BUDGET.includes("--time-limit 900"));
});

test("A2 asks about weights; a declined confirmation writes and sends nothing", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	const p = fakePi({
		cwd,
		answers: {
			select: ["sim", "RoboCasa", "here", "anthropic/claude-opus-5", ""],
			confirm: [true, false],
			keys: [["l", "\r"]],
		},
	});
	await p.run();
	// Right of A1 Code is RoboCasa's O1 × A1 Model (its VLA, which the weights serve).
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
			select: ["sim", "ManiSkill", "a remote", "other", "fine-tuned", ""],
			input: ["me@gpu", "/data/pi-embodied"],
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
			select: ["real", "UR5e", "here", "openai/gpt-6", ""],
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
			/^(libero|robocasa|robotwin|humanclaw|franka|dual-franka)$|^libero-/.test(r.id),
			`${r.id}: A1 Model needs a learned executor`,
		);
		for (const o of OS) for (const a of AS) assert.equal(cellFlags(r.axes, cell(o, a)).includes("undefined"), false);
	}
	assert.deepEqual(supportedCells(robot("libero").axes).length, 12, "LIBERO has every cell");
	assert.equal(cellFlags(robot("libero").axes, cell("O2", "A1M")), "--require-skills pi0 --detections");
	assert.equal(cellFlags(robot("libero").axes, cell("O3", "A1C")), "--units=true --privileged");
	assert.equal(cellFlags(robot("libero").axes, cell("O1", "A2")), "");
	assert.equal(cellFlags(robot("franka").axes, cell("O2", "A1M")), "--vla --segment");
	// RoboCasa: units and its VLA skill. Left of A1 Model is A1 Code, right is A2, down is O2 × A1 Model.
	const rc = robot("robocasa").axes;
	assert.deepEqual(supportedCells(rc)[0], cell("O1", "A1C"));
	assert.deepEqual(gridMove(rc, cell("O1", "A1M"), "left"), cell("O1", "A1C"));
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
	assert.match(rows[0], /^O1 Sensory Inputs\s+\[ \]\s+\[ \]\s+\[ \]\s+\[ \]\s*$/);
	assert.match(rows[1], /^O2 Perception Tools\s+\[ \]\s+\[ \]\s+\[●\]\s+\[ \]\s*$/);
	// A real arm without a VLA: A1 Model is dashed.
	assert.match(
		gridLines("T", robot("ur5e").axes, cell("O1", "A2")).find((l) => l.startsWith("O1 ")) ?? "",
		/^O1 Sensory Inputs\s+\[ \]\s+–\s+\[●\]\s+\[ \]\s*$/,
	);
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
				select: ["sim", "LIBERO", "here", "openai/gpt-6", ""],
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
	// O3 × A3 with a registry planner: the privileged code setting; the student is offered on the code surface.
	p = run(["j", "j", "l", "l", "l", "\r"]);
	await p.run();
	assert.match(p.confirms.at(-1) ?? "", /^flags: --code=true --privileged; planner/m);
	assert.deepEqual(p.offered["Planner model"], [
		rec("anthropic/claude-opus-5"),
		"openai/gpt-6",
		...HOSTED,
		FINETUNED_AS.code,
		FLASH,
	]);
	// O1 × A1 Model on LIBERO: the VLA skill is required; the student speaks the tools surface there.
	p = run(["l", "\r"]);
	await p.run();
	assert.match(p.confirms.at(-1) ?? "", /^flags: --require-skills pi0; planner/m);
	assert.ok(p.asked.includes("Model weights"), "A1 Model's VLA comes with the weights");
	assert.deepEqual(p.offered["Planner model"], [
		rec("anthropic/claude-opus-5"),
		"openai/gpt-6",
		...HOSTED,
		FINETUNED_AS.tools,
		FLASH,
	]);
});

test("outside the TUI the grid and the picker are selects", async () => {
	const p = fakePi({
		mode: "rpc",
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "HumanCLAW", "O3 × A1 Model", "here", "openai/gpt-6", ""],
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
			select: ["sim", "LIBERO", "O1 × A1 Code", "showharness", "here", "openai/gpt-6", ""],
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
	assert.deepEqual(
		ids(papersFor(robot("maniskill"), cell("O2", "A1C"))),
		["motormind"],
		"MotorMind runs on ManiSkill",
	);
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
	assert.deepEqual(ids(papersFor(robot("robocasa"), cell("O1", "A1C"))), ["showharness"], "RoboCasa has units");
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
			select: ["sim", "LIBERO", "here", "openai/gpt-6", ""],
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
	// Show-Harness's preset is the units the released adapter was trained on: the student is offered with it. The
	// filter also matches the name; a filter nothing matches is erased with backspace; vi keys are typed text here.
	const q = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "LIBERO", "here", "openai/gpt-6", ""],
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
	assert.deepEqual(q.offered["Planner model"], [
		rec("anthropic/claude-opus-5"),
		"openai/gpt-6",
		...HOSTED,
		FINETUNED,
		FLASH,
	]);
	assert.match(q.confirms.at(-1) ?? "", /^flags: --preset showharness; planner/m);
	// A cell with no ported setting (O1 × A3 on ManiSkill) skips the picker.
	const r = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["sim", "ManiSkill", "here", "openai/gpt-6", ""],
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
			select: ["sim", "ManiSkill", "here", "fine-tuned", ""],
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
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false] },
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
		answers: { select: [""], confirm: [false] },
	});
	await p.run("reproduce Show-Harness on metaworld here with openai/gpt-6");
	assert.deepEqual(p.asked, ["Experiment directory", "Set up pi-embodied?"]);
	// The directory's menu: the default, recommended, or another one typed.
	assert.deepEqual(p.offered["Experiment directory"], [rec(join(p.ctx.cwd, "embodied-metaworld")), OTHER]);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --preset showharness; planner openai\/gpt-6$/m);
	// The episode is /embodied-run with the same request.
	assert.match(
		p.confirms.at(-1) ?? "",
		/^or, from .*: \/embodied-run reproduce Show-Harness on metaworld here with openai\/gpt-6$/m,
	);
	// "franka" is three robots: the menu offers those three only; the rest is the usual flow.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: {
			select: ["Dual Franka", "here", "openai/gpt-6", ""],
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
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\r"]] },
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
		answers: { select: ["other", "openai/gpt-6", ""], input: ["/data/pi-embodied"], confirm: [false] },
	});
	await p.run("rpent on libero, ssh me@gpu");
	assert.ok(!p.asked.includes("ssh host"));
	assert.ok(p.asked.includes("pi-embodied checkout on the box"));
	assert.match(p.confirms.at(-1) ?? "", /^services on me@gpu:\/data\/pi-embodied\/services$/m);
});

test("/embodied-setup <request> with a work of the survey: its ported settings narrow the cell and the picker; one that is not ported is a notice", async () => {
	// CaP-X (ported as its tiers) sits in O2 × A3 and O3 × A3; Robosuite runs both: the cell is asked, the picker lists the tiers there.
	let p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["O2 × A3", "here", "openai/gpt-6", ""], confirm: [false], keys: [["\x1b[B", "\r"]] },
	});
	await p.run("CaP-X on robosuite");
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
	// ManiSkill's primitives are low and raw only: S1 (O3) is refused, so O2 × A3 without asking, with S3, S4 and M4.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\x1b[B", "\r"]] },
	});
	await p.run("CaP-X on maniskill");
	assert.ok(!p.asked.some((a) => a.startsWith("Cell for")));
	assert.deepEqual(
		p.grids[0].filter((l) => /capx-/.test(l)).map((l) => /capx-(\w+)/.exec(l)?.[1]),
		["S3", "S4", "M4"],
	);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --tier S3; planner/m);
	// A real arm has no O3: CaP-X on UR5e is O2 × A3 without asking.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\r"]] },
	});
	await p.run("CaP-X on ur5e");
	assert.ok(!p.asked.some((a) => a.startsWith("Cell for")));
	assert.match(p.confirms.at(-1) ?? "", /^UR5e \(real\): O2 × A3 /m);
	// Code as Policies is in the survey but not ported: a notice names its cells, then the grid.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\x1b"]] },
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
	// A setting the robot cannot run (MotorMind's on Genesis, which mounts no MotorMind mode): a warning, then the grid.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\r"], ["\r"]] },
	});
	await p.run("motormind on genesis");
	assert.ok(
		p.notes.some((n) => /^MotorMind's setting is not on Genesis; pick a cell$/.test(n)),
		p.notes.join("|"),
	);
	assert.equal(p.grids.length, 4, "the grid and the picker of O1 × A1 Code (Show-Harness), each rendered twice");
	assert.match(p.confirms.at(-1) ?? "", /^Genesis \(sim\): O1 × A1 Code /m);
	// Names nothing matches are called out as possibly not supported, in a notice and in the confirmation.
	const r = parseRequest("reproduce VoxPoser-2 on Aloha with units, here", ROBOTS, ["openai/gpt-6"]);
	assert.deepEqual(r.unknown, ["VoxPoser-2", "Aloha"]);
	assert.deepEqual([r.a, r.where], ["A1C", "here"]);
	assert.deepEqual(parseRequest("在 LIBERO-pro 里复现 RPent 的实验", ROBOTS, []).unknown, []);
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["sim", "LIBERO", "openai/gpt-6", ""], confirm: [false], keys: [["\r"], ["\r"]] },
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
	const p = fakePi({ cwd, answers: { select: ["here", "openai/gpt-6", ""], confirm: [true, true] } });
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
	let p = fakePi({ cwd, answers: { select: ["", "", RUN_HOW.print], confirm: [true] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.ok(!p.asked.includes("Experiment"));
	assert.deepEqual(p.offered["Task flags"], [rec("the recorded task: --task x --seed 0"), OTHER]);
	assert.deepEqual(p.offered["Episode budget"], [...BUDGETS.map((b, i) => (i ? b.label : rec(b.label))), OTHER]);
	assert.deepEqual(p.offered["How to run?"], [rec(RUN_HOW.here), RUN_HOW.headless, RUN_HOW.print]);
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
		answers: {
			select: ["other", "other", RUN_HOW.headless],
			input: ["--task push-v3 --seed 3", "--max-turns 10"],
			confirm: [true],
		},
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
	p = fakePi({ cwd, mode: "rpc", answers: { select: ["", "", RUN_HOW.print], confirm: [false] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.deepEqual(p.offered["How to run?"], [rec(RUN_HOW.headless), RUN_HOW.print]);
	assert.match(p.notes.at(-1) ?? "", /^Nothing started/);
	// No experiment here: told to set up first.
	p = fakePi({ cwd: mkdtempSync(join(tmpdir(), "pi-embodied-none-")) });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.match(p.notes.at(-1) ?? "", /^no experiment here: .*run \/embodied-setup first$/);
});

test("/embodied-run: several experiments are a menu, a request's robot picks one, an unrecorded setting is chosen as in setup, not installed offers setup", async () => {
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
			select: ["", "GPU 5", "", "the dashboard only", RUN_HOW.print],
			confirm: [true],
			keys: [["\r"], ["\x1b[B", "\r"]],
		},
	});
	await p.commands.get(RUN)?.handler("libero", p.ctx);
	assert.ok(!p.asked.includes("Experiment"));
	assert.equal(p.grids.length, 5, "the grid (two renders) and the picker (three)");
	assert.deepEqual(p.offered.GPU, [
		rec("GPU 5: 16275 MiB free of 46068"),
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
		/^UR5e is not installed: .*\.venv-ur5e\/pi-embodied\.env is missing\nto install it: \/embodied-setup ur5e$/,
	);
	// Another setting than the installed one is not started here: setup is offered with the same request.
	p = fakePi({ cwd, answers: { select: ["no"] } });
	await p.commands.get(RUN)?.handler("metaworld privileged code", p.ctx);
	assert.match(
		p.notes.at(-1) ?? "",
		/^nothing installed here is robot Metaworld; O3 × A3; installed: .*\nto install it: \/embodied-setup metaworld privileged code$/,
	);
	assert.deepEqual(p.offered["Not installed"], [rec(`${SETUP_NOW}: /embodied-setup metaworld privileged code`), "no"]);
	assert.equal(p.grids.length, 0);
	assert.deepEqual(p.sent, []);
	runDeps.gpus = () => [];
});

test("the run options: what a request says about one episode is split off, the rest is what /embodied-setup reads", () => {
	assert.deepEqual(runOptions("reproduce RPent on LIBERO libero_90 task 3 seed 1 gpu 2 20 turns headless"), {
		rest: "reproduce RPent on LIBERO libero",
		opts: { suite: "libero_90", task: "3", seed: "1", gpu: 2, turns: 20, how: "headless" },
	});
	assert.deepEqual(runOptions("在 LIBERO-pro 里复现 RPent 任务 5 显卡 3 后台"), {
		rest: "在 LIBERO-pro 里复现 RPent",
		opts: { task: "5", gpu: 3, how: "headless" },
	});
	assert.deepEqual(runOptions("metaworld --task push-v3 --max-turns 10 print"), {
		rest: "metaworld",
		opts: { task: "push-v3", turns: 10, how: "print" },
	});
	assert.deepEqual(runOptions("Show-Harness on metaworld here with openai/gpt-6"), {
		rest: "Show-Harness on metaworld here with openai/gpt-6",
		opts: {},
	});
	assert.equal(
		withRunOptions("--libero-type standard --suite libero_10 --task 0 --seed 0", { suite: "libero_90", task: "3" }),
		"--libero-type standard --suite libero_90 --task 3 --seed 0",
	);
	assert.equal(withRunOptions("--task reach-v3", { seed: "2" }), "--task reach-v3 --seed 2");
});

test("/embodied-run <request>: setup's request finds the installed experiment, the episode's words answer its questions, nothing installed offers setup", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	installed(cwd, "embodied-libero", "libero", {
		cell: { o: "O1", a: "A2" },
		paper: "rpent",
		flags: "--preset rpent",
		task: "--libero-type standard --suite libero_10 --task 0 --seed 0",
	});
	installed(cwd, "embodied-metaworld", "metaworld", {});
	runDeps.gpus = () => [
		{ index: 5, freeMiB: 16275, totalMiB: 46068 },
		{ index: 0, freeMiB: 4713, totalMiB: 46068 },
	];
	runDeps.freePort = async () => 8765;
	runDeps.env = {};
	// The text /embodied-setup took, with the episode's words: only the panels and the confirmation are asked.
	let p = fakePi({ cwd, answers: { select: ["the dashboard only"], confirm: [true] } });
	await p.commands.get(RUN)?.handler("reproduce RPent on LIBERO libero_90 task 3 seed 1 gpu 2 20 turns print", p.ctx);
	assert.deepEqual(p.asked, [PANELS, "Start the episode?"]);
	assert.match(p.confirms.at(-1) ?? "", /^LIBERO: O1 × A2, rpent; planner openai\/gpt-6$/m);
	assert.equal(
		p.notes.at(-1),
		`cd ${cwd}/embodied-libero && source ${cwd}/services/.venv-libero/pi-embodied.env && CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 pi --libero-type standard --suite libero_90 --task 3 --seed 1 --preset rpent --max-turns 20 --time-limit 900 --dashboard=true --dashboard-port 8765 "Solve the task."`,
	);
	// A planner named is this episode's --model; headless goes to the agent.
	p = fakePi({
		cwd,
		available: [
			{ provider: "openai", id: "gpt-6" },
			{ provider: "anthropic", id: "claude-opus-5" },
		],
		answers: { select: ["GPU 5", "", "the dashboard only"], confirm: [true] },
	});
	await p.commands.get(RUN)?.handler("rpent libero anthropic/claude-opus-5 task 2 headless", p.ctx);
	assert.match(p.confirms.at(-1) ?? "", /^LIBERO: O1 × A2, rpent; planner anthropic\/claude-opus-5$/m);
	assert.match(
		p.sent[0]?.text ?? "",
		/pi -a -p --libero-type standard --suite libero_10 --task 2 --seed 0 --preset rpent --model anthropic\/claude-opus-5 --max-turns 40/,
	);
	// Not installed: why, and /embodied-setup with the same request; declined, nothing happens.
	p = fakePi({ cwd, answers: { select: ["no"] } });
	await p.commands.get(RUN)?.handler("capx-S1 on libero task 3", p.ctx);
	assert.match(
		p.notes.at(-1) ?? "",
		/^nothing installed here is robot LIBERO; CaP-X S1's setting .*; installed: .*embodied-libero: LIBERO, O1 × A2, rpent.*\nto install it: \/embodied-setup capx-S1 on libero task 3$/,
	);
	assert.deepEqual(p.sent, []);
	// Taken: /embodied-setup runs on that request (the episode's words are left for /embodied-run).
	p = fakePi({ cwd, answers: { select: [SETUP_NOW, "here", "openai/gpt-6", ""], confirm: [false] } });
	await p.commands.get(RUN)?.handler("capx-S1 on libero task 3", p.ctx);
	assert.ok(p.asked.includes("Set up pi-embodied?"));
	assert.match(p.confirms.at(-1) ?? "", /^LIBERO \(sim\): O3 × A3/m);
	assert.doesNotMatch(p.confirms.at(-1) ?? "", /not recognized/);
	assert.match(p.confirms.at(-1) ?? "", /^or, from .*: \/embodied-run capx-S1 on libero task 3$/m);
	runDeps.gpus = () => [];
	runDeps.freePort = freePort;
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
	let p = fakePi({ cwd, answers: { select: ["", "", "CaP-X S1's own", RUN_HOW.print], confirm: [true] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.deepEqual(p.offered[PANELS], [
		rec(
			"CaP-X S1's own: Viser 3D view (CaP-X's viser_debug: point clouds, camera frustums, EEF, grasp candidates), with the dashboard",
		),
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
	p = fakePi({ cwd, answers: { select: ["", "", "the dashboard only", RUN_HOW.print], confirm: [true] } });
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.match(p.notes.at(-1) ?? "", /--dashboard-port 8766 "Solve the task\."\nssh/);
	assert.doesNotMatch(p.notes.at(-1) ?? "", /--viser/);
	// A plain cell on LIBERO: the robot's own Viser view is the offer; Show-Harness has no panel of its own.
	const lib = installed(cwd, "libero-units", "libero", { paper: "showharness", flags: "--preset showharness" });
	p = fakePi({
		cwd,
		answers: { select: ["libero-units", "", "", "the robot's", RUN_HOW.print], confirm: [true] },
	});
	await p.commands.get(RUN)?.handler("", p.ctx);
	assert.equal(
		p.offered[PANELS]?.[0],
		rec(
			"the robot's: Viser 3D view (CaP-X's viser_debug: point clouds, camera frustums, EEF, grasp candidates), with the dashboard",
		),
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

/** A deployment config for one test: PI_EMBODIED_CONFIG points at it; restore() puts the old value back. */
function withConfig(cfg: unknown) {
	const old = process.env.PI_EMBODIED_CONFIG;
	const file = join(mkdtempSync(join(tmpdir(), "pi-embodied-cfg-")), "embodied.json");
	writeFileSync(file, JSON.stringify(cfg));
	process.env.PI_EMBODIED_CONFIG = file;
	return () => {
		if (old === undefined) delete process.env.PI_EMBODIED_CONFIG;
		else process.env.PI_EMBODIED_CONFIG = old;
	};
}
const NINFER = SELFHOST.find((x) => x.engine === "NInfer")!;
const STRATA = SELFHOST.find((x) => x.engine === "Strata")!;

test("the self-hosted planners: vLLM's Qwen3.8-27B BF16, NInfer's Qwen3.8-27B and Strata's Qwen3.8-Flash-Next, as selfhost/<id>", () => {
	assert.deepEqual(
		SELFHOST.map((x) => `${x.engine}:${x.id}`),
		["vLLM:qwen3.8-27b-bf16", "NInfer:qwen3.8-27b", "Strata:qwen3.8-flash-next-unsloth-q8_0"],
	);
	assert.equal(selfhostLabel(NINFER), "selfhost/qwen3.8-27b: NInfer · Qwen3.8-27B NVFP4 (started by the agent)");
	assert.equal(selfhostOf("selfhost/qwen3.8-27b"), NINFER);
	assert.equal(selfhostOf("openai/gpt-6"), undefined);
	assert.equal(selfhostOf("selfhost/other"), undefined);
	const e = modelsJsonEntry(STRATA, "http://127.0.0.1:18080/v1");
	assert.equal(e.id, "qwen3.8-flash-next-unsloth-q8_0");
	assert.deepEqual(e.input, ["text", "image"]);
	assert.equal(e.contextWindow, 65536);
	// With a deployment entry: start (on the Mac), register, check. Without: set it up, asking which box.
	const lines = selfhostHandoff(
		NINFER,
		{
			base_url: "http://127.0.0.1:18000/v1",
			start: "selfhost-planner ninfer start",
			stop: "selfhost-planner ninfer stop",
			run_on: "the Mac",
		},
		"default",
	);
	assert.match(
		lines[0],
		/^- planner server: NInfer · Qwen3\.8-27B NVFP4 at http:\/\/127\.0\.0\.1:18000\/v1; if .*\/models does not answer, start it on the Mac \(this box cannot run it: show me the command, I run it there, then you continue\): `selfhost-planner ninfer start`; it stops with `selfhost-planner ninfer stop`/,
	);
	assert.match(
		lines[1],
		/^- planner registration: register it in ~\/\.pi\/agent\/models\.json under providers\.selfhost .*"id":"qwen3\.8-27b".*"baseUrl":"http:\/\/127\.0\.0\.1:18000\/v1"/,
	);
	assert.match(lines[2], /^- planner check: .*one chat request with a camera-sized image and one tool/);
	const bare = selfhostHandoff(STRATA, undefined, "lab");
	assert.equal(bare.length, 1);
	assert.match(bare[0], /deployment lab has no selfhost\.qwen3\.8-flash-next-unsloth-q8_0 .*ask me which box first/);
});

test("/embodied-setup offers the self-hosted planners; choosing one hands the start, registration and check to the agent", async () => {
	const restore = withConfig({
		deployments: {
			default: {
				selfhost: {
					"qwen3.8-27b": {
						base_url: "http://127.0.0.1:18000/v1",
						start: "selfhost-planner ninfer start",
						stop: "selfhost-planner ninfer stop",
						run_on: "the Mac",
					},
				},
			},
		},
	});
	try {
		const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
		const p = fakePi({
			cwd,
			answers: { select: ["here", "selfhost/qwen3.8-27b:", ""], confirm: [true] },
		});
		await p.run("reproduce Show-Harness on metaworld");
		assert.deepEqual(p.offered["Planner model"], [rec("anthropic/claude-opus-5"), "openai/gpt-6", ...HOSTED]);
		const summary = p.confirms.at(-1) ?? "";
		assert.match(summary, /^flags: --preset showharness; planner selfhost\/qwen3\.8-27b$/m);
		assert.match(
			summary,
			/^planner server: NInfer · Qwen3\.8-27B NVFP4 at http:\/\/127\.0\.0\.1:18000\/v1; the agent starts it if it is down \(its start runs on the Mac: you run that command\) and registers it in models\.json$/m,
		);
		const text = p.sent[0]?.text ?? "";
		assert.match(text, /^- planner: selfhost\/qwen3\.8-27b$/m);
		assert.match(text, /^- planner server: NInfer .*`selfhost-planner ninfer start`/m);
		assert.match(text, /^- planner registration: /m);
		const settings = JSON.parse(readFileSync(join(cwd, "embodied-metaworld/.pi/settings.json"), "utf8"));
		assert.deepEqual([settings.defaultProvider, settings.defaultModel], ["selfhost", "qwen3.8-27b"]);
		// Strata has no deployment entry: the agent sets it up, asking which box; the confirmation says so.
		const q = fakePi({
			cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
			answers: { select: ["here", "selfhost/qwen3.8-flash-next", ""], confirm: [true] },
		});
		await q.run("reproduce Show-Harness on metaworld");
		assert.match(
			q.confirms.at(-1) ?? "",
			/^planner server: Strata · Qwen3\.8-Flash-Next Q8_0 is not in the deployment yet; the agent sets it up on an NVIDIA card and about 140 GB of RAM/m,
		);
		assert.match(
			q.sent[0]?.text ?? "",
			/^- planner server: Strata · Qwen3\.8-Flash-Next Q8_0; deployment default has no selfhost\.qwen3\.8-flash-next-unsloth-q8_0/m,
		);
		// A request naming the planner by its id settles it without the menu.
		const r = fakePi({
			cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
			available: [
				{ provider: "openai", id: "gpt-6" },
				{ provider: "selfhost", id: "qwen3.8-27b" },
			],
			answers: { select: ["here", ""], confirm: [false] },
		});
		await r.run("reproduce Show-Harness on metaworld with selfhost/qwen3.8-27b");
		assert.ok(!r.asked.includes("Planner model"));
		assert.match(r.confirms.at(-1) ?? "", /^planner server: NInfer /m);
		assert.deepEqual(
			fakePiOffered(r),
			undefined,
			"a registered selfhost model is not listed twice (it is the self-hosted entry)",
		);
	} finally {
		restore();
	}
});

/** The planner menu a run offered, if it asked. */
const fakePiOffered = (p: { offered: Record<string, string[]> }) => p.offered["Planner model"];

test("/embodied-run checks a self-hosted planner before the episode and names its start when it does not answer", async () => {
	const restore = withConfig({
		deployments: {
			default: {
				selfhost: {
					"qwen3.8-27b": {
						base_url: "http://127.0.0.1:18000/v1",
						start: "selfhost-planner ninfer start",
						run_on: "the Mac",
					},
				},
			},
		},
	});
	try {
		const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
		const dir = installed(cwd, "embodied-metaworld", "metaworld", {});
		const f = join(dir, ".pi/settings.json");
		writeFileSync(
			f,
			JSON.stringify({
				...JSON.parse(readFileSync(f, "utf8")),
				defaultProvider: "selfhost",
				defaultModel: "qwen3.8-27b",
			}),
		);
		runDeps.gpus = () => [];
		runDeps.freePort = async () => 8765;
		runDeps.env = {};
		runDeps.selfhostUp = async () => false;
		let p = fakePi({ cwd, answers: { select: ["", "", RUN_HOW.print], confirm: [false] } });
		await p.commands.get(RUN)?.handler("", p.ctx);
		assert.match(p.confirms.at(-1) ?? "", /^Metaworld: O1 × A1 Code; planner selfhost\/qwen3\.8-27b$/m);
		assert.match(
			p.confirms.at(-1) ?? "",
			/^planner server: NInfer does NOT answer at http:\/\/127\.0\.0\.1:18000\/v1; start it first on the Mac: selfhost-planner ninfer start$/m,
		);
		runDeps.selfhostUp = async () => true;
		p = fakePi({ cwd, answers: { select: ["", "", RUN_HOW.print], confirm: [false] } });
		await p.commands.get(RUN)?.handler("", p.ctx);
		assert.match(p.confirms.at(-1) ?? "", /^planner server: NInfer answers at http:\/\/127\.0\.0\.1:18000\/v1$/m);
	} finally {
		restore();
	}
});

test("the planner list is searchable: typing narrows it, enter takes the first match, escape cancels", async () => {
	const p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		available: [
			{ provider: "openai", id: "gpt-6" },
			{ provider: "openai", id: "gpt-6-astra" },
			{ provider: "anthropic", id: "claude-opus-5" },
		],
		answers: { select: ["here", "astra", ""], confirm: [false] },
	});
	await p.run("reproduce Show-Harness on metaworld");
	assert.deepEqual(p.offered[PLANNER], [
		rec("anthropic/claude-opus-5"),
		"openai/gpt-6",
		"openai/gpt-6-astra",
		...HOSTED,
	]);
	assert.match(
		p.confirms.at(-1) ?? "",
		/^flags: --preset showharness; planner openai\/gpt-6-astra$/m,
		"'astra' narrowed it to one",
	);
	// Typing an engine's name finds the self-hosted planner.
	const q = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "strata", ""], confirm: [false] },
	});
	await q.run("reproduce Show-Harness on metaworld");
	assert.match(q.confirms.at(-1) ?? "", /planner selfhost\/qwen3\.8-flash-next-unsloth-q8_0$/m);
	// And "bf16" the unquantized one vLLM serves (with a deployment that does not list it, whatever this machine's has).
	const restore = withConfig({ deployments: { default: {} } });
	try {
		const b = fakePi({
			cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
			answers: { select: ["here", "bf16", ""], confirm: [false] },
		});
		await b.run("reproduce Show-Harness on metaworld");
		assert.match(b.confirms.at(-1) ?? "", /planner selfhost\/qwen3\.8-27b-bf16$/m);
		assert.match(
			b.confirms.at(-1) ?? "",
			/^planner server: vLLM · Qwen3\.8-27B BF16 \(unquantized\) is not in the deployment yet/m,
		);
	} finally {
		restore();
	}
	// Nothing typed and escape: nothing more is asked.
	const r = fakePi({ cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")), answers: { select: ["here"] } });
	await r.run("reproduce Show-Harness on metaworld");
	assert.equal(r.asked.at(-1), PLANNER);
	assert.deepEqual(r.sent, []);
	// The picker's lines: the filter and the count.
	const entries = ["openai/gpt-6", "openai/gpt-6-astra", ...HOSTED].map((m) => ({ id: m, text: m }));
	const lines = pickerLines(PLANNER, entries, "qwen", 0);
	assert.match(lines[1], /^ {2}type to filter: qwen_\s+3 of 5$/);
	assert.ok(lines.includes(`→ ${HOSTED[0]}`));
});

test("the ssh config's hosts are the remote boxes offered; patterns are not hosts", () => {
	const file = join(mkdtempSync(join(tmpdir(), "pi-embodied-ssh-")), "config");
	writeFileSync(file, "Host *\n  User x\nHost l40s-0003 autodl-bjb2\n  HostName a\nhost gpu-?\nHost !bad\n");
	assert.deepEqual(sshHosts(file), ["l40s-0003", "autodl-bjb2"]);
	assert.deepEqual(sshHosts("/nonexistent"), []);
});

/** Whether a robot's extension mounts a module, as ../src/infra/setup/caps.ts reads its spec. */
const mounts = (r: { id: string; extension: string }, key: "units" | "flash" | "motormind" | "memory") =>
	robotCaps(r)[key];

test("a ported setting is offered only where the robot runs it: CaP-X's high-API tiers need high-tier primitives", () => {
	for (const r of ROBOTS)
		for (const o of OS)
			for (const a of AS)
				for (const p of settingsFor(r, cell(o, a)))
					assert.deepEqual(tierProblems(r, paperFlag(p, r.id)), [], `${r.id} ${p.id}`);
	// ManiSkill's and RoboTwin's manifests have low and raw code primitives only: S3, S4 and M4 run, S1, S2, M1-M3 are refused.
	for (const id of ["maniskill", "robotwin"]) {
		assert.deepEqual(robotCaps(robot(id)).codeTiers, ["low", "raw"]);
		assert.deepEqual(ids(settingsFor(robot(id), cell("O2", "A3"))), ["capx-S3", "capx-S4", "capx-M4"], id);
		assert.deepEqual(settingsFor(robot(id), cell("O3", "A3")), [], `${id}: S1 needs the high level`);
		assert.match(tierProblems(robot(id), "--tier S2")[0] ?? "", /needs high-tier code primitives/);
	}
	assert.equal(settingsFor(robot("robosuite"), cell("O2", "A3")).length, 7, "robosuite has every level");
});

test("the wizard offers what the robots mount: units are A1 Code, MotorMind's preset goes where MotorMind mode is, Flash where its hook is", () => {
	for (const r of ROBOTS) {
		// HumanCLAW's units are run by its motion model: the survey files it under A1 Model only.
		if (r.id !== "humanclaw")
			assert.equal(supported(r.axes, cell("O1", "A1C")), mounts(r, "units"), `${r.id}: units`);
		assert.equal(Boolean(r.flash), mounts(r, "flash"), `${r.id}: Flash`);
		const mm = PAPERS.find((p) => p.id === "motormind")!;
		assert.equal(papersFor(r, mm.cell).includes(mm), mounts(r, "motormind"), `${r.id}: MotorMind`);
	}
	for (const p of PAPERS) for (const id of p.robots ?? []) assert.ok(robot(id), `${p.id}: ${id} is a robot`);
	// RoboCasa and RoboTwin run Show-Harness's units and MotorMind's preset now.
	for (const id of ["robocasa", "robotwin"]) {
		assert.equal(cellFlags(robot(id).axes, cell("O1", "A1C")), "--units=true");
		assert.deepEqual(ids(papersFor(robot(id), cell("O1", "A1C"))), ["showharness"], id);
		assert.deepEqual(ids(papersFor(robot(id), cell("O2", "A1C"))), ["motormind"], id);
	}
	// The Polymetis Franka has no VLA action space: no A1 Model, no RPent.
	assert.equal(supported(robot("franka-polymetis").axes, cell("O1", "A1M")), false);
	assert.deepEqual(ids(papersFor(robot("franka-polymetis"), cell("O1", "A2"))), []);
});

test("/embodied-setup MotorMind on RoboCasa, RoboTwin, ManiSkill: the preset, no warning, no grid", async () => {
	for (const [text, label] of [
		["MotorMind on RoboCasa", "RoboCasa365"],
		["reproduce MotorMind on RoboTwin", "RoboTwin 2"],
		["motormind on maniskill", "ManiSkill 3"],
	]) {
		const p = fakePi({
			cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
			answers: { select: ["here", "openai/gpt-6", ""], confirm: [false] },
		});
		await p.run(text);
		assert.deepEqual(p.notes, ["Nothing written or installed."], text);
		assert.equal(p.grids.length, 0, text);
		const summary = p.confirms.at(-1) ?? "";
		assert.match(
			summary,
			new RegExp(`^${label.replace(/[()]/g, "\\$&")} \\(sim\\): O2 × A1 Code .*, MotorMind's setting$`, "m"),
		);
		assert.match(summary, /^flags: --preset motormind; planner openai\/gpt-6$/m);
	}
});

test("RPent's preset goes where memory and a VLA skill of the robot's own are: the Frankas add --vla, ManiSkill's adapter is no such skill", async () => {
	const rp = PAPERS.find((p) => p.id === "rpent")!;
	const on = ROBOTS.filter((r) => papersFor(r, rp.cell).includes(rp)).map((r) => r.id);
	assert.deepEqual(on, ["libero", "libero-pro", "libero-plus", "robocasa", "robotwin", "franka", "dual-franka"]);
	// Each mounts memory, and its manifest declares a skill tiers.ts counts as RPent's VLA.
	const { VLA_SKILLS } = await import("../src/infra/tiers.ts");
	const { loadManifest } = await import("../src/primitives/manifest.ts");
	const manifestOf = (r: { extension: string }) =>
		/manifest: "(\w+)"/.exec(readFileSync(join(PKG, r.extension), "utf8"))?.[1] ?? "";
	for (const r of ROBOTS.filter((x) => on.includes(x.id))) {
		assert.ok(mounts(r, "memory"), `${r.id}: memory`);
		assert.ok(
			loadManifest(manifestOf(r)).primitives.some((e) => VLA_SKILLS.has(e.name)),
			`${r.id}: a VLA skill in ${manifestOf(r)}.json`,
		);
	}
	// ManiSkill: memory, but its only VLA is the opt-in OpenVLA adapter (openvla_act), which is no skill of its own.
	assert.ok(mounts(robot("maniskill"), "memory"));
	assert.equal(
		loadManifest("maniskill").primitives.some((e) => VLA_SKILLS.has(e.name)),
		false,
	);
	assert.equal(paperFlag(rp, "franka"), "--preset rpent --vla");
	assert.equal(paperFlag(rp, "robotwin"), "--preset rpent");
	assert.ok(pickEntries([rp], NO_PAPER, "dual-franka").some((e) => e.text.endsWith("(--preset rpent --vla)")));
	// RPent on RoboTwin and on the RLinf Franka from a request.
	let p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false, false] },
	});
	await p.run("RPent on RoboTwin");
	assert.match(p.confirms.at(-1) ?? "", /^flags: --preset rpent; planner openai\/gpt-6$/m);
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["Franka (RLinf)", "here", "openai/gpt-6", ""], confirm: [false] },
	});
	await p.run("reproduce RPent on franka");
	assert.match(p.confirms.at(-1) ?? "", /^flags: --preset rpent --vla; planner openai\/gpt-6$/m);
	assert.match(p.confirms.at(-1) ?? "", /--operator=true --preset rpent --vla --max-turns 40/);
	// ManiSkill: a warning, then the grid.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\x1b"]] },
	});
	await p.run("RPent on ManiSkill");
	assert.ok(p.notes.includes("RPent's setting is not on ManiSkill 3; pick a cell"), p.notes.join("|"));
});

test("a robot's own name is no work: CaP-X on RoboTwin is CaP-X; multi-turn narrows CaP-X to its M tiers, single-turn to its S tiers", async () => {
	let r = parseRequest("CaP-X on RoboTwin", ROBOTS, []);
	assert.equal(r.work?.id, "capx");
	assert.deepEqual(r.unknown, []);
	for (const [text, robotId] of [
		["CaP-X on ManiSkill", "maniskill"],
		["CaP-X on RoboCasa365", "robocasa"],
		["capx libero-pro", "libero-pro"],
	]) {
		r = parseRequest(text, ROBOTS, []);
		assert.equal(r.work?.id, "capx", text);
		assert.deepEqual(
			r.robots?.map((x) => x.id),
			[robotId],
			text,
		);
	}
	// A robot named alone is the robot, not the paper that introduced it.
	assert.equal(parseRequest("RoboTwin", ROBOTS, []).work, undefined);
	assert.equal(parseRequest("LIBERO-PRO units", ROBOTS, []).a, "A1C");
	// The paper of a benchmark that is no robot here still counts.
	assert.ok(parseRequest("Code as Policies", ROBOTS, []).work);
	r = parseRequest("CaP-X multi-turn on RoboTwin", ROBOTS, []);
	assert.deepEqual([r.work?.id, r.turns, r.unknown], ["capx", "multi", []]);
	assert.equal(parseRequest("CaP-X single turn", ROBOTS, []).turns, "single");
	// The flow: CaP-X on RoboTwin asks no warning; multi-turn is O2 × A3, where RoboTwin's low-level primitives run M4 only.
	let p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\x1b[B", "\r"]] },
	});
	await p.run("CaP-X multi-turn on RoboTwin");
	assert.deepEqual(p.notes, ["Nothing written or installed."]);
	assert.ok(!p.asked.some((a) => a.startsWith("Cell for")), "multi-turn tiers are O2 × A3 only");
	const tiers = p.grids[0].filter((l) => /capx-/.test(l)).map((l) => /capx-(\w+)/.exec(l)?.[1]);
	assert.deepEqual(tiers, ["M4"]);
	assert.match(p.confirms.at(-1) ?? "", /^RoboTwin 2 \(sim\): O2 × A3 .*, CaP-X M4's setting$/m);
	// Single-turn on Robosuite: both cells keep an S tier, so the cell is asked; O2 × A3 lists S2–S4.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["O2 × A3", "here", "openai/gpt-6", ""], confirm: [false], keys: [["\r"]] },
	});
	await p.run("CaP-X single-turn on robosuite");
	assert.deepEqual(
		p.grids[0].filter((l) => /capx-/.test(l)).map((l) => /capx-(\w+)/.exec(l)?.[1]),
		["S2", "S3", "S4"],
	);
});

test("VLA only, no agent: the VLA-alone baseline on every robot that runs one, outside the grid; elsewhere unsupported, never A1 Model", async () => {
	for (const text of [
		"VLA only, no agent",
		"OpenVLA alone on maniskill",
		"vla-only baseline",
		"the VLA without an agent",
	]) {
		const r = parseRequest(text, ROBOTS, []);
		assert.equal(r.vlaAlone, true, text);
		assert.equal(r.a, undefined, `${text}: not A1 Model`);
		assert.deepEqual(r.unknown, [], text);
	}
	assert.equal(parseRequest("the agent drives the VLA", ROBOTS, []).a, "A1M");
	// Every robot whose code runs --vla-alone has its flags here (../src/infra/setup/caps.ts reads the code).
	assert.deepEqual(
		ROBOTS.filter((r) => r.vlaAlone).map((r) => r.id),
		ROBOTS.filter((r) => robotCaps(r).vlaAlone).map((r) => r.id),
	);
	// Unnamed: the robots that run a VLA alone are the menu; no grid, no planner.
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	let p = fakePi({ cwd, answers: { select: ["ManiSkill", "here", ""], confirm: [true] } });
	await p.run("VLA only, no agent");
	assert.deepEqual(p.offered["Robot or benchmark"], [
		"LIBERO",
		"LIBERO-PRO",
		"LIBERO-plus",
		"ManiSkill 3",
		"Metaworld",
		"RoboCasa365",
		"RoboTwin 2",
		"RoboDojo",
	]);
	assert.deepEqual(p.asked, [
		"Robot or benchmark",
		"Where do the services run?",
		"Experiment directory",
		"Set up pi-embodied?",
	]);
	const summary = p.confirms.at(-1) ?? "";
	assert.match(summary, /^ManiSkill 3 \(sim\): the VLA alone, no agent /m);
	assert.match(
		summary,
		/^flags: --arm widowx250s --vla-adapter openvla --vla-alone openvla; planner none \(the VLA alone\)$/m,
	);
	assert.match(summary, /^needs: NVIDIA GPU \(Vulkan\); the OpenVLA server at services\.openvla/m);
	assert.match(
		summary,
		/pi --task PutCarrotOnPlateInScene-v1 --seed 0 --arm widowx250s --vla-adapter openvla --vla-alone openvla/,
	);
	const text = p.sent[0]?.text ?? "";
	assert.match(text, /^- planner: none; the VLA runs alone \(`--vla-alone`\) and no model is asked/m);
	const settings = JSON.parse(readFileSync(join(cwd, "embodied-maniskill/.pi/settings.json"), "utf8"));
	assert.equal(settings.defaultModel, undefined);
	const record = JSON.parse(readFileSync(join(cwd, "embodied-maniskill", RECORD), "utf8"));
	assert.deepEqual([record.cell, record.vlaAlone, record.planner], [undefined, true, "vla"]);
	assert.equal(record.task, "--task PutCarrotOnPlateInScene-v1 --seed 0");
	assert.match(experimentLabel(readExperiment(join(cwd, "embodied-maniskill"), ROBOTS)!), /the VLA alone/);
	// The robot's own VLA skill: Pi0.5 on LIBERO-PRO, no adapter; its weights are offered (the VLA runs on them).
	const pro = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	p = fakePi({ cwd: pro, answers: { select: ["here", ""], confirm: [true, true] } });
	await p.run("VLA only, no agent, on LIBERO-PRO");
	assert.deepEqual(p.asked, [
		"Where do the services run?",
		"Model weights",
		"Experiment directory",
		"Set up pi-embodied?",
	]);
	assert.match(p.confirms[0] ?? "", /^Also download Pi0\.5 LIBERO SFT .*\(recommended: the VLA runs on them\)$/);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --vla-alone pi0; planner none \(the VLA alone\)$/m);
	assert.match(p.confirms.at(-1) ?? "", /^needs: NVIDIA GPU \(MuJoCo EGL\); the Pi0\.5 server at services\.vla/m);
	assert.match(p.confirms.at(-1) ?? "", /setup\.sh libero-pro --weights/);
	assert.match(p.confirms.at(-1) ?? "", /pi --libero-type pro --suite libero_10 --task 0 --seed 0 --vla-alone pi0/);
	const proRecord = JSON.parse(readFileSync(join(pro, "embodied-libero-pro", RECORD), "utf8"));
	assert.deepEqual([proRecord.vlaAlone, proRecord.flags, proRecord.planner], [true, "--vla-alone pi0", "vla"]);
	// RoboCasa and RoboTwin: their own skills' names. RoboDojo: the XPolicyLab policy served at --xpolicy.
	for (const [name, flags] of [
		["RoboCasa", "--vla-alone rldx"],
		["RoboTwin", "--vla-alone lingbot"],
		["RoboDojo", "--xpolicy ws://127.0.0.1:19000 --vla-alone xpolicy"],
	]) {
		p = fakePi({
			cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
			answers: { select: ["here", ""], confirm: [false, false] },
		});
		await p.run(`the VLA alone on ${name}`);
		assert.ok((p.confirms.at(-1) ?? "").split("\n").includes(`flags: ${flags}; planner none (the VLA alone)`), name);
	}
	assert.match(p.confirms.at(-1) ?? "", /^needs: .*; an XPolicyLab policy server at ws:\/\/127\.0\.0\.1:19000/m);
	// Metaworld: Pi0.5 serves only its VLA alone, so the weights offered are that run's, and no other cell asks for them.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", ""], confirm: [true, false] },
	});
	await p.run("VLA only, no agent, on Metaworld");
	assert.match(
		p.confirms[0] ?? "",
		/^Also download Pi0\.5 MetaWorld SFT \(RLinf, ~7\.5 GB\) and its server's dependencies\? \(recommended: the VLA runs on them\)$/,
	);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --vla-alone pi0; planner none \(the VLA alone\)$/m);
	assert.match(p.confirms.at(-1) ?? "", /setup\.sh metaworld --weights/);
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\r"]] },
	});
	await p.run("metaworld O1 A2");
	assert.ok(!p.asked.includes("Model weights"), "Metaworld's tools need no weights");
	// A robot with no VLA integrated: refused, nothing asked or written.
	p = fakePi({ cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")), answers: {} });
	await p.run("VLA only, no agent, on Robosuite");
	assert.match(
		p.notes[0] ?? "",
		/^the VLA alone \(no agent\) is not supported on Robosuite: only LIBERO, LIBERO-PRO, LIBERO-plus, ManiSkill 3, Metaworld, RoboCasa365, RoboTwin 2, RoboDojo run a VLA alone \(--vla-alone\); no VLA is integrated for the others\./,
	);
	assert.deepEqual([p.asked, p.sent], [[], []]);
});

test("planners: a registry model by its display name, the student on every surface the setting has, Flash's replay where the robot has a hook", async () => {
	const fable = { ref: "anthropic/claude-fable-5-1", name: "Claude Fable 5.1" };
	const models = [fable, { ref: "anthropic/claude-fable-5", name: "Claude Fable 5" }, "openai/gpt-6-astra"];
	let r = parseRequest("Show-Harness on LIBERO-PRO with Claude Fable 5.1", ROBOTS, models);
	assert.deepEqual([r.model, r.unknown], ["anthropic/claude-fable-5-1", []]);
	assert.equal(parseRequest("claude fable 5 on robotwin", ROBOTS, models).model, "anthropic/claude-fable-5");
	assert.equal(parseRequest("gpt-6-astra", ROBOTS, models).model, "openai/gpt-6-astra");
	r = parseRequest("libero-pro tools, fine-tuned student", ROBOTS, models);
	assert.deepEqual([r.model, r.a, r.unknown], ["finetuned", "A2", []]);
	assert.equal(parseRequest("flash replay on libero-pro", ROBOTS, models).model, "flash/replay");
	// The display name reaches the wizard from the registry.
	let p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		available: [{ provider: "anthropic", id: "claude-fable-5-1", name: "Claude Fable 5.1" } as never],
		answers: { select: ["here", ""], confirm: [false] },
	});
	await p.run("Show-Harness on LIBERO-PRO with Claude Fable 5.1");
	assert.ok(!p.asked.includes(PLANNER));
	assert.match(p.confirms.at(-1) ?? "", /^flags: --preset showharness; planner anthropic\/claude-fable-5-1$/m);
	// The student on LIBERO-PRO's A2 and A3: an adapter the user serves on that surface.
	for (const [text, surface, flags] of [
		["libero-pro O1 A2 with the fine-tuned student", "tools", ""],
		["libero-pro O2 A3 fine-tuned student", "code", "--code=true --detections "],
	] as const) {
		p = fakePi({
			cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
			answers: { select: ["here", ""], confirm: [false, true], keys: [["\r"]] },
		});
		await p.run(text);
		assert.ok(!p.asked.includes(PLANNER), text);
		assert.match(
			p.confirms.at(-1) ?? "",
			new RegExp(
				`^flags: ${flags}--model finetuned/local --ft-model <served adapter> --ft-surface ${surface}; planner fine-tuned \\(${surface} surface\\)`,
				"m",
			),
		);
		const sent = p.sent[0]?.text ?? "";
		assert.match(sent, new RegExp(`^- adapter: none is released for the ${surface} surface`, "m"));
		assert.doesNotMatch(sent, /FT_ADAPTER=/);
	}
	// The student with Show-Harness's preset: the released units adapter.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", ""], confirm: [false] },
	});
	await p.run("Show-Harness on maniskill, fine-tuned student");
	assert.match(
		p.confirms.at(-1) ?? "",
		/^flags: --preset showharness --model finetuned\/qwen3_5_2b_showharness_sim; planner fine-tuned: /m,
	);
	// Not with MotorMind's preset, which plans with its own roles: a notice, then the menu without it.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false] },
	});
	await p.run("MotorMind on LIBERO-PRO with the fine-tuned student");
	assert.ok(p.notes.includes("the fine-tuned student is not offered for this setting on LIBERO-PRO; pick a planner"));
	assert.deepEqual(p.offered[PLANNER], [rec("anthropic/claude-opus-5"), "openai/gpt-6", ...HOSTED]);
	// Flash on LIBERO-PRO's A2: --model flash/replay, no default model; the handoff says a plan must exist.
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	p = fakePi({
		cwd,
		answers: { select: ["here", "flash/replay", ""], confirm: [false, true], keys: [["\r"]] },
	});
	await p.run("libero-pro O1 A2");
	assert.match(p.confirms.at(-1) ?? "", /^flags: --model flash\/replay; planner flash\/replay$/m);
	assert.match(p.sent[0]?.text ?? "", /^- plan: flash\/replay needs a recorded plan before an episode/m);
	const settings = JSON.parse(readFileSync(join(cwd, "embodied-libero-pro/.pi/settings.json"), "utf8"));
	assert.deepEqual([settings.defaultProvider, settings.defaultModel], [undefined, undefined]);
	// No Flash hook on Metaworld: neither offered nor taken from a request.
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false] },
	});
	await p.run("metaworld O1 A2 flash replay");
	assert.ok(p.notes.includes("flash/replay is not offered for this setting on Metaworld; pick a planner"));
	assert.deepEqual(p.offered[PLANNER], [
		rec("anthropic/claude-opus-5"),
		"openai/gpt-6",
		...HOSTED,
		FINETUNED_AS.tools,
	]);
});

/** A model answering from a script: each call takes the next reply (an Error is thrown); the prompts it got are kept. */
function scripted(replies: (unknown | Error)[]) {
	const prompts: { system: string; user: string }[] = [];
	const ask: Ask = async (system, user) => {
		prompts.push({ system, user });
		const r = replies.shift();
		if (r instanceof Error) throw r;
		return typeof r === "string" ? r : `Here it is:\n\`\`\`json\n${JSON.stringify(r ?? {})}\n\`\`\``;
	};
	return { ask, prompts };
}
async function withAsk(ask: Ask, fn: () => Promise<void>) {
	agentDeps.ask = ask;
	try {
		await fn();
	} finally {
		agentDeps.ask = undefined;
	}
}
const errorsOf = <T>(c: { value: T } | { errors: string[] }) => ("errors" in c ? c.errors : []);

test("the readers' catalog is generated from the code: cells, runnable settings, what each robot mounts, its flags and eval.sh", () => {
	const cat = setupCatalog(ROBOTS, [{ ref: "openai/gpt-6", name: "GPT-6" }]);
	const file = new URL("./fixtures/setup-catalog.json", import.meta.url);
	const text = `${JSON.stringify(cat, null, "\t")}\n`;
	if (process.env.UPDATE_SNAPSHOTS) writeFileSync(file, text);
	assert.equal(
		text,
		readFileSync(file, "utf8"),
		"the catalog changed with the code: check the diff, then UPDATE_SNAPSHOTS=1 node --test test/setup.test.ts",
	);
	assert.deepEqual(
		cat.robots.map((r) => r.id),
		ROBOTS.map((r) => r.id),
	);
	const ms = cat.robots.find((r) => r.id === "maniskill")!;
	assert.deepEqual(ms.vla_alone, {
		mounted: true,
		setup_flags: "--arm widowx250s --vla-adapter openvla --vla-alone openvla",
		example_task: "--task PutCarrotOnPlateInScene-v1 --seed 0",
	});
	assert.deepEqual(ms.cells.find((c) => c.cell === "O2/A3")?.settings, ["capx-S3", "capx-S4", "capx-M4"]);
	assert.deepEqual(ms.mounts.vla_adapters, ["openvla"]);
	assert.equal(cat.robots.find((r) => r.id === "humanclaw")?.vla_alone.mounted, false);
	assert.deepEqual(cat.robots.find((r) => r.id === "robotwin")?.mounts.vla_skills, ["lingbot_act"]);
	// The batch script's usage and the variables its header names; the flags a robot takes, not another robot's.
	assert.match(evalUsage(robot("libero-pro"))?.usage[0] ?? "", /^eval\.sh <out-dir> <suite> <tasks> <seeds>/);
	assert.equal(evalUsage(robot("libero-pro"))?.script, "packages/embodied/src/robots/libero/eval.sh");
	assert.ok(evalUsage(robot("robocasa"))?.env.includes("TASKS"));
	assert.equal(evalUsage(robot("franka")), undefined, "a real arm has no batch script");
	assert.ok(knownFlags(robot("libero-pro")).has("libero-type"));
	assert.ok(!knownFlags(robot("maniskill")).has("libero-type"));
	for (const f of ["preset", "tier", "units", "model", "thinking", "max-turns"])
		assert.ok(knownFlags(robot("maniskill")).has(f), f);
	// The run catalog: the experiments here, ready or not, their recorded setting and planner, the GPUs, the budget.
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	installed(cwd, "embodied-robotwin", "robotwin", { paper: "capx-M4", cell: cell("O2", "A3"), flags: "--tier M4" });
	const rc = runCatalog(
		scanExperiments(cwd, ROBOTS),
		[{ ref: "openai/gpt-6" }],
		[{ index: 0, label: "GPU 0" }],
		BUDGET,
	);
	assert.deepEqual(rc.experiments[0].setting, {
		cell: "O2/A3",
		vla_alone: false,
		preset_or_tier: "capx-M4",
		flags: "--tier M4",
	});
	assert.equal(rc.experiments[0].ready, true);
	assert.equal(rc.experiments[0].planner, "openai/gpt-6");
	assert.equal(rc.default_budget, BUDGET);
});

test("a setup reading is checked before anything is asked: unknown ids, settings a robot refuses, cells it lacks, planners it cannot use", () => {
	const x = { robots: ROBOTS, planners: ["openai/gpt-6", "anthropic/claude-fable-5-1"] };
	const first = (j: Record<string, unknown>) => errorsOf(checkSetup(j, x))[0] ?? "";
	assert.match(first({ robots: ["aloha"] }), /^robots: aloha is not a robot id of the catalog$/);
	assert.match(
		first({ robots: ["maniskill"], setting: "rpent" }),
		/^setting rpent on maniskill: --preset rpent calls a VLA skill/,
	);
	assert.match(
		first({ robots: ["genesis"], setting: "motormind" }),
		/^setting motormind on genesis: --preset motormind runs MotorMind mode \(--mm\), which genesis does not mount$/,
	);
	// A setting the robot would run but whose cell it does not have: HumanCLAW has no O2.
	assert.match(
		first({ robots: ["humanclaw"], setting: "openeta" }),
		/^setting: openeta is not offered on humanclaw \(its robots: any with the cell\)$/,
	);
	// The tier's API level is missing on the robot: refused as the robot refuses it at start.
	assert.match(
		first({ robots: ["robotwin"], setting: "capx-M2" }),
		/capx-M2 on robotwin: .*needs high-tier code primitives/,
	);
	assert.match(first({ robots: ["libero"], setting: "nope" }), /nope is not a setting id/);
	assert.match(first({ robots: ["piper"], vla_alone: true }), /piper does not run a VLA alone/);
	assert.match(first({ robots: ["ur5e"], cell: "O3/A3" }), /^cell: ur5e has no O3\/A3$/);
	assert.match(first({ robots: ["libero-pro"], setting: "capx-S1", cell: "O2/A3" }), /not capx-S1's cell \(O3\/A3\)/);
	assert.match(first({ robots: ["metaworld"], cell: "O1/A2", planner: "flash/replay" }), /cannot replay a plan/);
	assert.match(
		first({ robots: ["libero-pro"], setting: "motormind", planner: "finetuned" }),
		/student does not plan motormind/,
	);
	assert.match(first({ planner: "openai/gpt-9" }), /openai\/gpt-9 is not a planner of the catalog/);
	assert.match(first({ thinking: "ultra" }), /^thinking: ultra is not one of/);
	assert.match(first({ host: "me@gpu; rm -rf /" }), /is not an ssh host/);
	// A reading that passes becomes the wizard's request.
	const ok = checkSetup(
		{
			robots: ["robotwin"],
			setting: "capx-M4",
			planner: "anthropic/claude-fable-5-1",
			thinking: "high",
			where: "remote",
			host: "me@gpu",
			unsupported: ["3 seeds per task is a run, not a setup"],
		},
		x,
	);
	assert.ok("value" in ok, errorsOf(ok).join("; "));
	if (!("value" in ok)) return;
	const r = ok.value;
	assert.deepEqual(
		[r.robots?.map((b) => b.id), r.kind, r.paper?.id, r.model, r.thinking, r.where, r.host, r.unsupported],
		[
			["robotwin"],
			"sim",
			"capx-M4",
			"anthropic/claude-fable-5-1",
			"high",
			"remote",
			"me@gpu",
			["3 seeds per task is a run, not a setup"],
		],
	);
	// A family of robots narrows the menu; the VLA alone needs no cell.
	const fam = checkSetup({ robots: ["franka", "franka-polymetis", "dual-franka"] }, x);
	assert.deepEqual("value" in fam ? fam.value.robots?.length : 0, 3);
	const alone = checkSetup({ robots: ["maniskill"], vla_alone: true }, x);
	assert.equal("value" in alone && alone.value.vlaAlone, true);
});

test("a run reading is checked: the experiment, flags the robot takes, the batch script's arguments, the GPU, the tier still runnable", () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	const lib = installed(cwd, "embodied-libero-pro", "libero-pro", {
		paper: "rpent",
		cell: cell("O1", "A2"),
		flags: "--preset rpent",
		task: "--libero-type pro --suite libero_10 --task 0 --seed 0",
	});
	const tw = installed(cwd, "embodied-robotwin", "robotwin", {
		paper: "capx-M4",
		cell: cell("O2", "A3"),
		flags: "--tier M4",
	});
	const x = { experiments: scanExperiments(cwd, ROBOTS), planners: ["openai/gpt-6"], gpus: [0, 1] };
	const first = (j: Record<string, unknown>) => errorsOf(checkRun(j, x))[0] ?? "";
	assert.match(first({ experiment: "/nowhere" }), /^experiment: \/nowhere is not an experiment of the catalog/);
	assert.match(first({ task_flags: "--task 1" }), /task_flags needs the experiment: name one/);
	assert.match(
		first({ experiment: lib, task_flags: "--suite libero_10 --arm panda" }),
		/--arm is not a flag libero-pro takes/,
	);
	assert.match(first({ experiment: lib, task_flags: "--seed 1; rm -rf /" }), /cannot carry safely/);
	assert.match(first({ experiment: tw, task_flags: "--tier M2" }), /needs high-tier code primitives/);
	assert.match(first({ experiment: lib, gpu: 7 }), /^gpu: 7 is not one of 0, 1$/);
	assert.match(first({ experiment: lib, mode: "batch" }), /mode "batch" needs a batch/);
	const batch = (b: Record<string, unknown>) => first({ experiment: lib, mode: "batch", batch: b });
	assert.match(batch({ args: ["out/x", "libero_10", "0-9", "1-3"] }), /must be the output directory runs\/<name>/);
	assert.match(batch({ args: ["runs/../x"] }), /must be the output directory/);
	assert.match(batch({ args: ["runs/x", "$(id)"] }), /is not a plain argument/);
	assert.match(batch({ args: ["runs/x"], env: { FOO: "1" } }), /FOO is not a variable .*libero\/eval\.sh reads/);
	assert.match(batch({ args: ["runs/x"], flags: "--made-up" }), /--made-up is not a flag libero-pro takes/);
	assert.match(first({ experiment: lib, planner: "openai/gpt-9" }), /not a planner/);
	const ok = checkRun(
		{
			experiment: lib,
			mode: "batch",
			batch: { args: ["runs/rpent-pro", "libero_10", "0-9", "1-3"], env: {}, flags: "--thinking low" },
			planner: "openai/gpt-6",
			gpu: 1,
			max_turns: 30,
		},
		x,
	);
	assert.ok("value" in ok, errorsOf(ok).join("; "));
	if ("value" in ok)
		assert.deepEqual([ok.value.dir, ok.value.how, ok.value.gpu, ok.value.maxTurns], [lib, "batch", 1, 30]);
});

test("interpret: a failed answer goes back with the reasons; a reply without JSON is one; the rounds are capped", async () => {
	const check = (j: Record<string, unknown>) =>
		j.n === 2 ? { value: j.n } : { errors: [`n is ${String(j.n)}, not 2`] };
	let m = scripted([{ n: 1 }, { n: 2 }]);
	let r = await interpret(m.ask, "SYS", "REQ", check);
	assert.deepEqual([r.value, r.rounds], [2, 2]);
	assert.match(m.prompts[1].user, /^REQ\n\nYour previous answer:\n[\s\S]*It failed these checks:\n- n is 1, not 2\n/);
	m = scripted(["no json here", { n: 3 }, { n: 4 }, { n: 2 }]);
	r = await interpret(m.ask, "SYS", "REQ", check);
	assert.deepEqual([r.value, r.rounds, r.errors], [undefined, 3, ["n is 4, not 2"]]);
	assert.match(m.prompts[1].user, /- the reply holds no JSON object/);
	assert.equal(m.prompts.length, 3, "a first answer and two corrections");
});

test("/embodied-setup <request> with a model: its reading against the catalog settles the setup; a failed check is sent back; Chinese works", async () => {
	const m = scripted([
		{ robots: ["maniskill"], setting: "rpent", planner: "openai/gpt-6", where: "here" },
		{
			robots: ["robotwin"],
			setting: "rpent",
			planner: "openai/gpt-6",
			thinking: "high",
			where: "here",
			unsupported: [],
		},
	]);
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	const p = fakePi({ cwd, answers: { select: [""], confirm: [false, true] } });
	await withAsk(m.ask, () => p.run("在 RoboTwin 上复现 RPent，规划器用 gpt-6，思考深度高，本机"));
	assert.equal(m.prompts.length, 2);
	assert.match(m.prompts[0].system, /never the A1M cell/);
	assert.match(m.prompts[0].user, /^Catalog:\n\{"grid"/);
	assert.match(m.prompts[0].user, /\n\nRequest:\n在 RoboTwin 上复现 RPent/);
	assert.match(m.prompts[1].user, /- setting rpent on maniskill: --preset rpent calls a VLA skill/);
	assert.ok(p.notes.includes("reading the request with anthropic/claude-opus-5"));
	assert.ok(!p.asked.includes(PLANNER) && !p.asked.includes("Where do the services run?"));
	const summary = p.confirms.at(-1) ?? "";
	assert.match(
		summary,
		/^from your request: robot RoboTwin 2; RPent's setting \(O1 × A2, --preset rpent\); planner openai\/gpt-6; thinking high; services here$/m,
	);
	assert.match(summary, /^flags: --preset rpent; planner openai\/gpt-6 \(thinking high\)$/m);
	const settings = JSON.parse(readFileSync(join(cwd, "embodied-robotwin/.pi/settings.json"), "utf8"));
	assert.deepEqual([settings.defaultModel, settings.defaultThinkingLevel], ["gpt-6", "high"]);
});

test("/embodied-setup <request>: what the model says is not supported is shown; a reading that never passes, or no answer, falls back to keywords", async () => {
	// Unsupported: a notice and a line in the confirmation; the rest is asked.
	let m = scripted([{ robots: ["libero-pro"], unsupported: ["Aloha is not a robot here"] }]);
	let p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false], keys: [["\r"], ["\r"]] },
	});
	await withAsk(m.ask, () => p.run("libero-pro, and Aloha too"));
	assert.ok(p.notes.includes("not supported: Aloha is not a robot here"));
	assert.match(p.confirms.at(-1) ?? "", /^not supported: Aloha is not a robot here$/m);
	// Three failed readings: the keywords read the same text.
	m = scripted([{ robots: ["aloha"] }, { robots: ["aloha"] }, { robots: ["aloha"] }]);
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false] },
	});
	await withAsk(m.ask, () => p.run("reproduce RPent on LIBERO-PRO"));
	assert.ok(
		p.notes.some((n) =>
			/^anthropic\/claude-opus-5's reading failed the checks 3 times \(robots: aloha is not a robot id of the catalog\); reading the request by keywords instead$/.test(
				n,
			),
		),
		p.notes.join("|"),
	);
	assert.match(p.confirms.at(-1) ?? "", /^from your request: robot LIBERO-PRO; RPent's setting/m);
	// The model unreachable: the same.
	m = scripted([new Error("connect ECONNREFUSED")]);
	p = fakePi({
		cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")),
		answers: { select: ["here", "openai/gpt-6", ""], confirm: [false] },
	});
	await withAsk(m.ask, () => p.run("reproduce RPent on LIBERO-PRO"));
	assert.ok(
		p.notes.some((n) =>
			/could not read the request \(connect ECONNREFUSED\); reading it by keywords instead$/.test(n),
		),
	);
	assert.match(p.confirms.at(-1) ?? "", /RPent's setting/);
	// No text: nothing is sent to the model.
	m = scripted([]);
	p = fakePi({ cwd: mkdtempSync(join(tmpdir(), "pi-embodied-exp-")), answers: {} });
	await withAsk(m.ask, () => p.run(""));
	assert.equal(m.prompts.length, 0);
});

test("/embodied-run <request> with a model: a batch over tasks and seeds with the robot's eval.sh; an unknown experiment is sent back; a planner override", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	const lib = installed(cwd, "embodied-libero-pro", "libero-pro", {
		paper: "rpent",
		cell: cell("O1", "A2"),
		flags: "--preset rpent",
		task: "--libero-type pro --suite libero_10 --task 0 --seed 0",
	});
	installed(cwd, "embodied-robotwin", "robotwin", { paper: "capx-M4", cell: cell("O2", "A3"), flags: "--tier M4" });
	runDeps.gpus = () => [
		{ index: 0, freeMiB: 40000, totalMiB: 46000 },
		{ index: 1, freeMiB: 30000, totalMiB: 46000 },
	];
	runDeps.freePort = async () => 8765;
	runDeps.env = {};
	// The batch: libero_10 tasks 0-9, seeds 1-3, on GPU 1, with Claude Fable instead of the recorded planner.
	let m = scripted([
		{
			experiment: lib,
			mode: "batch",
			batch: { args: ["runs/rpent-pro", "libero_10", "0-9", "1-3"], env: {}, flags: "" },
			planner: "anthropic/claude-fable-5-1",
			gpu: 1,
			max_turns: 30,
		},
	]);
	let p = fakePi({
		cwd,
		available: [
			{ provider: "openai", id: "gpt-6" },
			{ provider: "anthropic", id: "claude-fable-5-1", name: "Claude Fable 5.1" } as never,
		],
		answers: { select: [BATCH_HOW.print], confirm: [true] },
	});
	await withAsk(
		m.ask,
		() =>
			p.commands
				.get(RUN)
				?.handler("RPent on LIBERO-PRO, libero_10 tasks 0-9, 3 seeds each, Claude Fable 5.1, GPU 1", p.ctx) ??
			Promise.resolve(),
	);
	assert.match(m.prompts[0].system, /eval\.usage/);
	assert.match(m.prompts[0].user, /"script":"packages\/embodied\/src\/robots\/libero\/eval\.sh"/);
	assert.match(m.prompts[0].user, /"gpus":\[\{"index":0/);
	assert.ok(!p.asked.includes("Experiment") && !p.asked.includes("GPU") && !p.asked.includes("Task flags"));
	const command = `cd ${cwd} && source ${cwd}/services/.venv-libero-pro/pi-embodied.env && CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 packages/embodied/src/robots/libero/eval.sh runs/rpent-pro libero_10 0-9 1-3 --preset rpent --model anthropic/claude-fable-5-1 --max-turns 30 --time-limit 900`;
	assert.match(p.confirms.at(-1) ?? "", new RegExp(`^${command.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&")}$`, "m"));
	assert.deepEqual(p.offered["How to run the batch?"], [rec(BATCH_HOW.headless), BATCH_HOW.print]);
	assert.equal(p.notes.at(-1), command);
	// Headless: the agent runs the batch and reports its summary.
	m = scripted([
		{
			experiment: lib,
			mode: "batch",
			batch: { args: ["runs/rpent-pro", "libero_10", "0-9", "1-3"], env: {}, flags: "" },
			gpu: 0,
		},
	]);
	p = fakePi({ cwd, answers: { select: ["", BATCH_HOW.headless], confirm: [true] } });
	await withAsk(m.ask, () => p.commands.get(RUN)?.handler("rpent batch", p.ctx) ?? Promise.resolve());
	assert.match(
		p.sent[0]?.text ?? "",
		/^\/skill:embodied-quickstart Run this batch now, in the background, and report its summary: `cd .*eval\.sh runs\/rpent-pro libero_10 0-9 1-3 --preset rpent --model openai\/gpt-6 --max-turns 40 --time-limit 900`$/m,
	);
	// An unknown experiment goes back; the corrected one episode, printed, with the planner replaced.
	m = scripted([
		{ experiment: "/elsewhere", mode: "print" },
		{
			experiment: lib,
			mode: "print",
			task_flags: "--libero-type pro --suite libero_10 --task 3 --seed 2",
			planner: "anthropic/claude-fable-5-1",
			gpu: 0,
		},
	]);
	p = fakePi({
		cwd,
		available: [{ provider: "anthropic", id: "claude-fable-5-1", name: "Claude Fable 5.1" } as never],
		answers: { select: ["", ""], confirm: [true] },
	});
	await withAsk(
		m.ask,
		() =>
			p.commands
				.get(RUN)
				?.handler("task 3 seed 2 on the LIBERO-PRO one with Claude Fable 5.1, just print it", p.ctx) ??
			Promise.resolve(),
	);
	assert.match(m.prompts[1].user, /- experiment: \/elsewhere is not an experiment of the catalog/);
	assert.ok(!p.asked.includes("How to run?"));
	assert.match(
		p.notes.at(-1) ?? "",
		/pi --libero-type pro --suite libero_10 --task 3 --seed 2 --preset rpent --model anthropic\/claude-fable-5-1 --max-turns 40/,
	);
});

test("the model's questions: only open fields, only options that pass the checks, the recommended one first", () => {
	const x = { robots: ROBOTS, planners: ["openai/gpt-6", "anthropic/claude-opus-5"] };
	const check = (j: Record<string, unknown>) => checkSetup(j, x);
	const json = {
		robots: ["robotwin"],
		where: "here",
		questions: [
			{
				field: "planner",
				header: "Planner",
				question: "Which model plans?",
				options: [
					{ value: "invented/model", label: "Invented" },
					{ value: "anthropic/claude-opus-5", label: "Claude Opus 5" },
					{ value: "openai/gpt-6", label: "GPT-6 (Recommended)", description: "this session's model" },
				],
			},
			// Settled by the answer: not asked again.
			{
				field: "where",
				header: "Where",
				question: "Where?",
				options: [
					{ value: "here", label: "here" },
					{ value: "remote", label: "remote" },
				],
			},
			// One option left after the checks: dropped, the menus ask it.
			{
				field: "setting",
				header: "Setting",
				question: "Which setting?",
				options: [
					{ value: "nope", label: "Nope" },
					{ value: "rpent", label: "RPent" },
				],
			},
			// Not a field the model may ask about.
			{
				field: "summary",
				header: "S",
				question: "?",
				options: [
					{ value: "a", label: "a" },
					{ value: "b", label: "b" },
				],
			},
		],
	};
	const qs = openQuestions(json, check, SETUP_QUESTIONS);
	assert.deepEqual(
		qs.map((q) => [q.field, q.options.map((o) => o.label), q.values]),
		[["planner", ["GPT-6 (Recommended)", "Claude Opus 5"], ["openai/gpt-6", "anthropic/claude-opus-5"]]],
	);
	// The user's choice goes in and is checked; a typed answer the checks refuse is left out.
	let r = withAnswers(json, qs, [{ header: "Planner", selected: ["Claude Opus 5"] }], check);
	assert.equal(r.value?.model, "anthropic/claude-opus-5");
	assert.deepEqual(r.refused, []);
	r = withAnswers(json, qs, [{ header: "Planner", selected: [], typed: "made/up" }], check);
	assert.equal(r.value?.model, undefined);
	assert.match(r.refused[0] ?? "", /^planner made\/up: planner: made\/up is not a planner of the catalog$/);
});

test("/embodied-setup <request> with a model: what the request leaves open the model asks as choices, the rest is the usual menus; cancelling stops", async () => {
	const reading = {
		robots: ["robotwin"],
		setting: "rpent",
		where: "here",
		questions: [
			{
				field: "planner",
				header: "Planner",
				question: "Which model plans each step?",
				options: [
					{
						value: "openai/gpt-6",
						label: "GPT-6 (Recommended)",
						description: "this session's model, sees images",
					},
					{ value: "anthropic/claude-opus-5", label: "Claude Opus 5" },
				],
			},
			{
				field: "thinking",
				header: "Thinking",
				question: "How much should it think per step?",
				options: [
					{ value: "low", label: "Low (Recommended)", description: "fast turns" },
					{ value: "high", label: "High" },
				],
			},
		],
	};
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-exp-"));
	let p = fakePi({
		cwd,
		available: [
			{ provider: "openai", id: "gpt-6" },
			{ provider: "anthropic", id: "claude-opus-5" },
		],
		answers: { select: ["Claude Opus 5", "Low", ""], confirm: [false] },
	});
	await withAsk(scripted([reading]).ask, () => p.run("RPent on RoboTwin here"));
	assert.deepEqual(p.offered["Planner: Which model plans each step?"], [
		"GPT-6 (Recommended) — this session's model, sees images",
		"Claude Opus 5",
		TYPE_OWN,
	]);
	assert.ok(!p.asked.includes(PLANNER), "the planner menu is not asked again");
	assert.deepEqual(p.asked.slice(-2), ["Experiment directory", "Set up pi-embodied?"]);
	assert.match(p.confirms.at(-1) ?? "", /^flags: --preset rpent; planner anthropic\/claude-opus-5 \(thinking low\)$/m);
	// Cancelled: nothing is asked after it, nothing is written.
	p = fakePi({
		cwd,
		available: [
			{ provider: "openai", id: "gpt-6" },
			{ provider: "anthropic", id: "claude-opus-5" },
		],
		answers: { select: [] },
	});
	await withAsk(scripted([reading]).ask, () => p.run("RPent on RoboTwin here"));
	assert.deepEqual(p.asked, ["Planner: Which model plans each step?"]);
	assert.equal(p.notes.at(-1), "Nothing written or installed.");
});

test("/embodied-run <request> with a model: an experiment the request does not single out is the model's question", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "pi-embodied-run-"));
	const pro = installed(cwd, "embodied-libero-pro", "libero-pro", {
		paper: "rpent",
		cell: cell("O1", "A2"),
		flags: "--preset rpent",
		task: "--libero-type pro --suite libero_10 --task 0 --seed 0",
	});
	const tw = installed(cwd, "embodied-robotwin", "robotwin", {
		paper: "capx-M4",
		cell: cell("O2", "A3"),
		flags: "--tier M4",
	});
	runDeps.gpus = () => [];
	runDeps.freePort = async () => 8765;
	runDeps.env = {};
	const m = scripted([
		{
			mode: "print",
			questions: [
				{
					field: "experiment",
					header: "Experiment",
					question: "Which installed experiment?",
					options: [
						{ value: pro, label: "LIBERO-PRO RPent (Recommended)", description: "matches 'the libero one'" },
						{ value: tw, label: "RoboTwin CaP-X M4" },
						{ value: "/elsewhere", label: "Not installed" },
					],
				},
			],
		},
	]);
	const p = fakePi({ cwd, answers: { select: ["RoboTwin", "", "", RUN_HOW.print], confirm: [true] } });
	await withAsk(m.ask, () => p.commands.get(RUN)?.handler("run one episode, print it", p.ctx) ?? Promise.resolve());
	assert.deepEqual(
		p.offered["Experiment: Which installed experiment?"]?.length,
		3,
		"the uninstalled option is dropped",
	);
	assert.ok(!p.asked.includes("Experiment"), "the experiment menu is not asked again");
	assert.match(p.notes.at(-1) ?? "", new RegExp(`^cd ${tw} && .*--tier M4`));
	runDeps.freePort = freePort;
});
