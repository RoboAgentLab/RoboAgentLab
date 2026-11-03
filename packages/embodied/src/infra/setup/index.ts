/**
 * Onboarding: the package's only default extension. It points a new user to `/embodied-setup` once
 * per project, and `/embodied-setup` asks for the robot, its cell in the survey's observation ×
 * execution grid (./axes.ts: O1–O3 × A1–A3), optionally a paper's setting in that cell, where the
 * services run and the planner, writes the experiment directory's `.pi/settings.json`, and hands
 * the install to the agent (the embodied-quickstart skill), which runs services/setup.sh and the
 * preflight with its own tools once the user has confirmed the plan (pi does not gate each tool
 * call). The install and the experiment are separate: the agent installs and checks, and ends with
 * the launch command the user runs afterwards; it starts no episode.
 *
 * A robot extension replaces the coding tools, so robots never load by default: the experiment
 * directory's settings load one robot (and the dashboard) through `extensions`, and a delta entry
 * for this package (`autoload: false`, `-src/infra/setup/index.ts`) drops this extension there. Package
 * filters only narrow what the manifest declares (docs/packages.md), which is why the robots are
 * listed as extension paths and not as a filter of this package.
 *
 * The notice is shown once per project: a `pi-embodied-setup-notice` session entry records it, and
 * any session of the project (the session directory) that holds one suppresses it.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
	type Axes,
	type Cell,
	cellFlags,
	cellKey,
	cellLabel,
	cellName,
	EXECUTION,
	gridLines,
	gridMove,
	keyOf,
	OBSERVATION,
	PAPERS,
	type Paint,
	type Paper,
	type Pick,
	papersFor,
	pickEntries,
	pickerLines,
	pickMatches,
	supported,
	supportedCells,
	typed,
} from "./axes.ts";
import { parseRequest, requestCell } from "./request.ts";
import {
	type Experiment,
	episodeFlags,
	experimentLabel,
	gpuLabel,
	gpus,
	headlessCommand,
	RECORD,
	runCommand,
	type SetupRecord,
	scanExperiments,
} from "./run.ts";

/** The package root and the services checkout next to it (absent in an npm install). */
export const PKG = fileURLToPath(new URL("../../..", import.meta.url)).replace(/\/$/, "");
const SERVICES = join(PKG, "../../services");
const REPO = "https://github.com/RoboAgentLab/RoboAgentLab.git";
const SKILL = join(PKG, "skills/embodied-quickstart/SKILL.md");
export const NOTICE_ENTRY = "pi-embodied-setup-notice";
export const COMMAND = "embodied-setup";
/** The same command under the skill's name, for `/embodied-quickstart <request>`. */
export const ALIAS = "embodied-quickstart";
/** Start an episode of an installed experiment. */
export const RUN = "embodied-run";
export const RUN_HOW = {
	here: "here: pi takes this terminal until the episode ends (trust prompt, then type Solve the task.)",
	headless: "headless: the agent runs pi -a -p now, in the background, and reports the result",
	print: "print: show the command, I run it myself",
} as const;
export const NOTICE =
	"pi-embodied: run /embodied-setup to pick a robot, its observation × execution cell and where the services run; the agent then installs and checks them.";
/** Without a model the agent cannot run the install, and the planner modes have nothing to plan with. */
export const NO_MODEL =
	"/embodied-setup: no model is configured. The agent that runs the install (and the planner) needs one: run /login, or add a provider to ~/.pi/agent/models.json (pi's docs/models.md), then /embodied-setup again.";
/** The quickstart's episode budget: an interactive episode has none by default and can run (and bill) for an hour. */
export const BUDGET = "--max-turns 40 --time-limit 900";
export const GRID = "Observation × Execution";
export const PAPER = "Work to reproduce (optional)";
export const NO_PAPER = "none: the cell's plain flags";
/** The planner option that runs the robot's fine-tuned adapter instead of a model of the registry (A1 Code only). */
export const FINETUNED = "fine-tuned: a small VLM + LoRA picks each unit (served by vLLM)";

export type Kind = "sim" | "real";
export const KINDS: Record<Kind, string> = {
	sim: "sim: a simulator or benchmark on a GPU",
	real: "real: a physical robot, an operator at the e-stop",
};
export type Robot = {
	/** services/setup.sh target. */
	id: string;
	/** Simulated or a physical robot: the first menu of /embodied-setup. */
	kind: Kind;
	/** The name shown in the robot menu; what it needs is in `needs` (the confirmation, not the menu). */
	label: string;
	/** The robot extension, relative to the package. */
	extension: string;
	needs: string;
	/** The levels of the grid this robot has, and their flags (./axes.ts). */
	axes: Axes;
	/** Example task flags for the launch command. */
	task: string;
	/** What `setup.sh --weights` fetches (the VLA of A1 Model and A2, SAM3 of O2). */
	weights?: string;
	/** Fine-tuned planner (A1 Code): model and flags. */
	finetuned?: string;
	/** Fine-tuned planner: the released adapter `setup.sh finetuned` fetches (FT_ADAPTER); unset when the user serves their own. */
	adapter?: string;
};

/** A simulator with ground truth, units and code mode; A1 Model needs a VLA skill (LIBERO's Pi0.5: `LIBERO`). */
const SIM: Axes = { O1: "", O2: "--detections", O3: "--privileged", A1C: "--units=true", A2: "", A3: "--code=true" };
const LIBERO: Axes = { ...SIM, A1M: "--require-skills pi0" };
/** A simulator whose tools are a VLA (RLDX-1, LingBot): no units. */
const VLA_SIM = (skill: string): Axes => ({
	O1: "",
	O2: "--detections",
	O3: "--privileged",
	A1M: `--require-skills ${skill}`,
	A2: "",
	A3: "--code=true",
});
/** A real Franka: SAM3 through its own --segment, Pi0.5 through --vla, no ground truth. */
const ARM: Axes = { O1: "", O2: "--segment", A1C: "--units=true", A1M: "--vla", A2: "", A3: "--code=true" };
const SIM_FT = "--model finetuned/qwen3_5_2b_showharness_sim";
const LIBERO_FT = "--model finetuned/local --ft-model <served adapter> --ft-prompt v5-libero";
export const ROBOTS: Robot[] = [
	{
		id: "libero",
		kind: "sim",
		label: "LIBERO",
		extension: "src/robots/libero/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL)",
		axes: LIBERO,
		task: "--libero-type standard --suite libero_10 --task 0 --seed 0",
		weights: "Pi0.5 LIBERO SFT + SAM3 (gated: HF_TOKEN)",
		finetuned: LIBERO_FT,
	},
	{
		id: "libero-pro",
		kind: "sim",
		label: "LIBERO-PRO",
		extension: "src/robots/libero/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL)",
		axes: LIBERO,
		task: "--libero-type pro --suite libero_10 --task 0 --seed 0",
		weights: "Pi0.5 LIBERO SFT + SAM3 (gated: HF_TOKEN)",
		finetuned: LIBERO_FT,
	},
	{
		id: "libero-plus",
		kind: "sim",
		label: "LIBERO-plus",
		extension: "src/robots/libero/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL), ~9.5 GB scene assets, system ImageMagick (libmagickwand)",
		axes: LIBERO,
		task: "--libero-type plus --suite libero_10 --task 0 --seed 0",
		weights: "Pi0.5 LIBERO SFT + SAM3 (gated: HF_TOKEN)",
		finetuned: LIBERO_FT,
	},
	{
		id: "maniskill",
		kind: "sim",
		label: "ManiSkill 3",
		extension: "src/robots/maniskill/index.ts",
		needs: "NVIDIA GPU (Vulkan)",
		axes: SIM,
		task: "--task PickCube-v1 --seed 0",
		finetuned: SIM_FT,
		adapter: "qwen3_5_2b_sim",
	},
	{
		id: "metaworld",
		kind: "sim",
		label: "Metaworld",
		extension: "src/robots/metaworld/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL)",
		axes: SIM,
		task: "--task reach-v3 --seed 0",
	},
	{
		id: "robosuite",
		kind: "sim",
		label: "Robosuite",
		extension: "src/robots/robosuite/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL), SAM3 for segment; the CaP-X tasks",
		axes: SIM,
		task: "--task Lift --seed 0",
	},
	{
		id: "genesis",
		kind: "sim",
		label: "Genesis",
		extension: "src/robots/genesis/index.ts",
		needs: "NVIDIA GPU (a cu128 torch on RTX 50xx); a Franka, cube_pick",
		axes: SIM,
		task: "--task cube_pick --seed 0",
	},
	{
		id: "robocasa",
		kind: "sim",
		label: "RoboCasa365",
		extension: "src/robots/robocasa/index.ts",
		needs: "NVIDIA GPU, ~10 GB kitchen assets",
		axes: VLA_SIM("rldx"),
		task: "--task OpenDrawer --split target --seed 0",
		weights: "RLDX-1-FT-RC365 VLA",
	},
	{
		id: "robotwin",
		kind: "sim",
		label: "RoboTwin 2",
		extension: "src/robots/robotwin/index.ts",
		needs: "NVIDIA GPU (SAPIEN/Vulkan), cuRobo build; two arms",
		axes: VLA_SIM("lingbot"),
		task: "--task beat_block_hammer --task-config demo_randomized --seed 0",
		weights: "LingBot-VLA RoboTwin EEF",
	},
	{
		id: "robolab",
		kind: "sim",
		label: "RoboLab",
		extension: "src/robots/robolab/index.ts",
		needs: "RTX GPU, Isaac Sim 6.1 / Isaac Lab 3.0",
		axes: SIM,
		task: "--task BananaInBowlTask --seed 0",
		finetuned: SIM_FT,
		adapter: "qwen3_5_2b_sim",
	},
	{
		id: "robodojo",
		kind: "sim",
		label: "RoboDojo",
		extension: "src/robots/robodojo/index.ts",
		needs: "RTX GPU, Isaac Sim 6.1 / Isaac Lab 3.0, ~41 GB assets; two arms",
		axes: SIM,
		task: "--task stack_bowls --seed 0",
	},
	{
		id: "behavior",
		kind: "sim",
		label: "BEHAVIOR-1K",
		extension: "src/robots/behavior/index.ts",
		needs: "RTX GPU, Isaac Sim 6.1 (OmniGibson), a BEHAVIOR-1K checkout and its data (license key); the R1Pro robot",
		axes: SIM,
		task: "--task turning_on_radio --seed 0",
	},
	{
		id: "humanclaw",
		kind: "sim",
		label: "HumanCLAW",
		extension: "src/robots/humanclaw/index.ts",
		needs: "NVIDIA GPU, a Habitat-Sim build (nvcc), the gated HSSD datasets (HF_TOKEN); a humanoid in HSSD homes",
		// The survey files HumanCLAW under A1 Model: its units are run by the motion model; no SAM3.
		axes: { O1: "", O3: "--privileged", A1M: "" },
		task: "--units=both --humanclaw-mode pi --episode one",
	},
	{
		id: "franka",
		kind: "real",
		label: "Franka (RLinf)",
		extension: "src/robots/franka/index.ts",
		needs: "real Franka, RLinf controller + Ray, an operator at the e-stop",
		axes: ARM,
		task: "--robot-config <yaml> --task 0 --operator=true",
	},
	{
		id: "franka-polymetis",
		kind: "real",
		label: "Franka (Polymetis)",
		extension: "src/robots/franka/index.ts",
		needs: "real Franka on a Polymetis NUC, an operator at the e-stop",
		axes: ARM,
		task: "--backend polymetis --robot-config <yaml> --operator=true",
	},
	{
		id: "dual-franka",
		kind: "real",
		label: "Dual Franka",
		extension: "src/robots/dual_franka/index.ts",
		needs: "two real Frankas, RLinf + Ray, an operator at the e-stop",
		axes: ARM,
		task: "--robot-config <yaml> --operator=true",
	},
	{
		id: "piper",
		kind: "real",
		label: "AgileX Piper",
		extension: "src/robots/piper/index.ts",
		needs: "Piper arm, ROS Noetic, an operator at the e-stop",
		axes: { O1: "", O2: "--detections", A1C: "--units=true", A2: "", A3: "--code=true" },
		task: "--robot-config <yaml> --task banana_plate --operator=true",
		finetuned: "--model finetuned/qwen3_5_2b_showharness_ft --ft-prompt v4-piper",
		adapter: "qwen3_5_2b",
	},
	{
		id: "ur5e",
		kind: "real",
		label: "UR5e",
		extension: "src/robots/ur5e/index.ts",
		needs: "UR5e with a Robotiq gripper, cameras, an operator at the e-stop",
		axes: { O1: "", O2: "--segment", A1C: "--units=true", A2: "", A3: "--code=true" },
		task: "--arm <controller serial> --robot-config <yaml> --task <task> --operator=true",
	},
];

type Settings = Record<string, unknown> & { packages?: unknown[]; extensions?: unknown[] };
export type Choice = {
	robot: Robot;
	cell: Cell;
	/** A ported work's setting, run as its preset or tier. */
	paper?: Paper;
	remote?: { host: string; checkout: string };
	services: string;
	weights: boolean;
	/** The planner of the registry, or none when `finetuned`. */
	model?: string;
	/** The robot's fine-tuned adapter is the planner (A1 Code). */
	finetuned: boolean;
	dir: string;
};

/** A robot extension is active when `/robot-task` (registered by every robot's base) exists. */
const robotActive = (pi: ExtensionAPI) => pi.getCommands().some((c) => c.name === "robot-task");

function noticeSeen(ctx: ExtensionContext): boolean {
	if (ctx.sessionManager.getEntries().some((e) => e.type === "custom" && e.customType === NOTICE_ENTRY)) return true;
	const mark = `"customType":"${NOTICE_ENTRY}"`;
	try {
		const dir = ctx.sessionManager.getSessionDir();
		return readdirSync(dir).some((f) => f.endsWith(".jsonl") && readFileSync(join(dir, f), "utf8").includes(mark));
	} catch {
		return false;
	}
}

/** How this package is declared (npm/git as configured, else its absolute path), for the delta entry. */
function packageSource(pi: ExtensionAPI): string {
	const info = pi.getCommands().find((c) => c.name === COMMAND)?.sourceInfo;
	if (info?.origin === "package" && /^(npm:|git:|https?:|ssh:|git@)/.test(info.source)) return info.source;
	return PKG;
}

/** The setting's flags: the ported work's preset or tier, else the cell's, then the fine-tuned planner's. */
export function settingFlags(c: Choice): string {
	const cell = c.paper ? c.paper.flag : cellFlags(c.robot.axes, c.cell);
	return [cell, c.finetuned ? c.robot.finetuned : undefined].filter(Boolean).join(" ");
}

/** The launch flags for a choice (after `pi`), with the quickstart's episode budget. */
export function launchFlags(c: Choice): string {
	return [c.robot.task, settingFlags(c), BUDGET, "--dashboard=true"].filter(Boolean).join(" ");
}

/** The experiment directory's settings: this package without onboarding, the robot, the dashboard, the planner. */
export function experimentSettings(c: Choice, source: string, root: string, existing: Settings = {}): Settings {
	const src = (p: string) => `${root}/${p}`;
	const extensions = [src(c.robot.extension), src("src/capabilities/dashboard/index.ts")];
	if (c.finetuned) extensions.push(src("src/modes/finetuned/index.ts"));
	const same = (p: unknown) => {
		const s = typeof p === "string" ? p : (p as { source?: unknown })?.source;
		return s === source || s === root;
	};
	const out: Settings = {
		...existing,
		packages: [
			...(existing.packages ?? []).filter((p) => !same(p)),
			{ source, autoload: false, extensions: ["-src/infra/setup/index.ts"] },
		],
		extensions: [
			...(existing.extensions ?? []).filter((e) => typeof e !== "string" || !e.startsWith(`${root}/src/`)),
			...extensions,
		],
	};
	if (c.model) {
		const i = c.model.indexOf("/");
		out.defaultProvider = c.model.slice(0, i);
		out.defaultModel = c.model.slice(i + 1);
	}
	return out;
}

/** The command that starts an episode: the experiment directory, the environment setup.sh wrote, pi. */
export function launchCommand(c: Choice): string {
	return `cd ${c.dir} && source ${c.services}/.venv-${c.robot.id}/pi-embodied.env && pi ${launchFlags(c)}`;
}

/** A model the agent can run on: pi's placeholder `unknown/unknown` (no model configured) is none. */
const realModel = (m: { provider: string; id: string } | undefined) =>
	m && !(m.provider === "unknown" && m.id === "unknown") ? `${m.provider}/${m.id}` : undefined;

/** The setting in one line: the cell with the survey's sections and names, the ported work, or the work to reproduce. */
export const settingLine = (c: Choice) => {
	const O = OBSERVATION[c.cell.o];
	const A = EXECUTION[c.cell.a];
	return `${cellName(c.cell)} (${O.section} ${O.name} × ${A.section} ${A.name})${c.paper ? `, ${c.paper.name}'s setting` : ""}`;
};

function handoff(c: Choice, settingsText: string): string {
	const setup = `${c.services}/setup.sh ${c.robot.id}${c.weights ? " --weights" : ""}`;
	const where = c.remote
		? `remote: run everything on \`ssh ${c.remote.host}\` in the checkout ${c.remote.checkout} (clone ${REPO} there if missing); the robot runs where pi runs, so pi and the experiment directory go on the box too`
		: existsSync(c.services)
			? `local, services at ${c.services}`
			: `local; no services checkout next to the package: clone ${REPO} to ${dirname(c.services)}`;
	const settings = c.remote
		? `write this to <experiment dir on the box>/.pi/settings.json (the package paths become ${c.remote.checkout}/packages/embodied):\n${settingsText}`
		: `${join(c.dir, ".pi/settings.json")} is written`;
	// O2: the perception server the flags attach to.
	const perception =
		c.cell.o === "O2"
			? [
					`- perception (O2): a SAM3 server at the deployment's services.sam3 must be running for the episodes (\`--serve-models sam3\` per episode, or the robot's serve.sh; SAM3 is gated: HF_TOKEN)`,
				]
			: [];
	// Fine-tuned planner: the adapter and its base model, and the vLLM server services.finetuned (default :8010) names.
	const ft = c.robot.adapter ? `FT_ADAPTER=${c.robot.adapter} ${c.services}/setup.sh finetuned` : undefined;
	const finetuned = c.finetuned
		? [
				ft
					? `- adapter: \`${ft}\` (finetuned/download.py fetches the adapter and its base model, pinned and verified)`
					: "- adapter: none is released for this robot; ask me which adapter to serve and where its files are",
				`- serve: \`MODEL=<base dir> LORA=<name>=<adapter dir> VLLM_VENV=<venv with vllm> bash ${c.services}/pi_embodied_services/finetuned/serve.sh\` (${ft ? "setup.sh finetuned prints the exact command; " : ""}services.finetuned defaults to http://127.0.0.1:8010/v1); it must be running for the episodes`,
			]
		: [];
	return [
		`/skill:embodied-quickstart Install and verify pi-embodied with the choices I confirmed in /embodied-setup (skill file: ${SKILL}):`,
		`- robot: ${c.robot.label} (${c.robot.kind}); setup target \`${c.robot.id}\`; needs ${c.robot.needs}`,
		`- setting: ${settingLine(c)}; flags \`${settingFlags(c) || "none"}\``,
		`- services: ${where}`,
		`- install: \`${setup}\`${c.weights ? ` (weights: ${c.robot.weights})` : ""}`,
		...perception,
		c.model ? `- planner: ${c.model}` : `- planner: ${FINETUNED}`,
		...finetuned,
		`- experiment dir: ${c.dir}; ${settings}`,
		`- launch: \`${launchCommand(c)}\` (I start episodes with it myself, afterwards)`,
		"- trust: the first pi in the experiment dir asks whether to trust the project (its .pi/settings.json loads the robot extension); I answer yes there, or the robot does not load (/trust saves the decision)",
		"Run the install, then the preflight, fix what fails, and end with the launch command. Do not start an episode. Ask me before sudo, anything touching real hardware, or downloads not listed here.",
	].join("\n");
}

/** The grid: a custom component in the TUI, a list of the supported cells elsewhere. */
async function pickCell(ctx: ExtensionCommandContext, robot: Robot): Promise<Cell | undefined> {
	const cells = supportedCells(robot.axes);
	const title = `${GRID} for ${robot.label}`;
	if (ctx.mode !== "tui") {
		const opts = cells.map(cellLabel);
		return cells[opts.indexOf((await ctx.ui.select(title, opts)) ?? "")];
	}
	return ctx.ui.custom<Cell | undefined>((_tui, theme, keybindings, done) => {
		let cur = cells[0];
		const paint: Paint = {
			accent: (s) => theme.fg("accent", s),
			dim: (s) => theme.fg("dim", s),
			bold: (s) => theme.bold(s),
		};
		return {
			render: () => gridLines(title, robot.axes, cur, paint),
			invalidate: () => {},
			handleInput: (data: string) => {
				const key = keyFrom(data, keybindings, true);
				if (key === "enter") done(cur);
				else if (key === "escape") done(undefined);
				else if (key) cur = gridMove(robot.axes, cur, key);
			},
		};
	});
}

/** The select keybindings and the arrows, as one key; `vi`: h/j/k/l/q too (not where text is typed). */
function keyFrom(data: string, keybindings: { matches(data: string, id: string): boolean }, vi: boolean) {
	return keybindings.matches(data, "tui.select.up")
		? "up"
		: keybindings.matches(data, "tui.select.down")
			? "down"
			: keybindings.matches(data, "tui.select.confirm")
				? "enter"
				: keybindings.matches(data, "tui.select.cancel")
					? "escape"
					: keyOf(data, vi);
}

type Picked = { paper?: Paper };
/**
 * The work to reproduce: the cell's ported settings, a searchable list (typing filters by the short name)
 * in the TUI, a select elsewhere. Cancelled: undefined.
 */
async function pickWork(ctx: ExtensionCommandContext, papers: Paper[]): Promise<Picked | undefined> {
	const entries = pickEntries(papers, NO_PAPER);
	if (ctx.mode !== "tui") {
		const picked = await ctx.ui.select(
			PAPER,
			entries.map((e) => e.text),
		);
		const e = entries.find((x) => x.text === picked);
		return e ? { paper: e.paper } : undefined;
	}
	return ctx.ui.custom<Picked | undefined>((_tui, theme, keybindings, done) => {
		let q = "";
		let idx = 0;
		const paint: Paint = {
			accent: (s) => theme.fg("accent", s),
			dim: (s) => theme.fg("dim", s),
			bold: (s) => theme.bold(s),
		};
		return {
			render: () => pickerLines(PAPER, entries, q, idx, paint),
			invalidate: () => {},
			handleInput: (data: string) => {
				const shown: Pick[] = pickMatches(entries, q);
				const key = keyFrom(data, keybindings, false);
				if (key === "enter") {
					if (shown[idx]) done({ paper: shown[idx].paper });
				} else if (key === "escape") done(undefined);
				else if (key === "up") idx = Math.max(0, idx - 1);
				else if (key === "down") idx = Math.min(shown.length - 1, idx + 1);
				else {
					const t = typed(data);
					if (t === "erase") q = q.slice(0, -1);
					else if (t) q += t.char;
					else return;
					idx = 0;
				}
			},
		};
	});
}

async function setup(pi: ExtensionAPI, ctx: ExtensionCommandContext, args = ""): Promise<void> {
	if (robotActive(pi)) {
		ctx.ui.notify("A robot is already loaded here; run /embodied-setup from a directory without one.", "warning");
		return;
	}
	if (!ctx.hasUI) {
		ctx.ui.notify("/embodied-setup needs an interactive session", "error");
		return;
	}
	const ui = ctx.ui;
	const current = realModel(ctx.model);
	const available = ctx.modelRegistry.getAvailable().map((m) => `${m.provider}/${m.id}`);
	if (!available.length) {
		ui.notify(NO_MODEL, "error");
		return;
	}
	// A request settles what it names; the rest is asked. A name nothing matches may be something not supported yet.
	const req = parseRequest(args, ROBOTS, available);
	if (req.unknown.length)
		ui.notify(
			`not recognized: ${req.unknown.join(", ")}; no robot, ported setting, survey work, level or planner by that name (maybe not supported yet). The rest is asked.`,
			"warning",
		);
	let kind = req.kind;
	if (!kind) {
		const kindOpts = Object.values(KINDS);
		kind = (Object.keys(KINDS) as Kind[])[
			kindOpts.indexOf((await ui.select("Simulation or real robot?", kindOpts)) ?? "")
		];
		if (!kind) return;
	}
	// One named robot is taken; several named (franka) narrow the menu to them.
	const robots = req.robots ?? ROBOTS.filter((r) => r.kind === kind);
	let robot: Robot | undefined = robots.length === 1 ? robots[0] : undefined;
	if (!robot) {
		const robotOpts = robots.map((r) => r.label);
		robot = robots[robotOpts.indexOf((await ui.select("Robot or benchmark", robotOpts)) ?? "")];
	}
	if (!robot) return;
	// The cell: the named setting's, else one of a named work's cells on this robot (its ported settings, when it
	// has some, narrow the picker; a work that is not ported is a notice), else the named levels', else the grid.
	let cell: Cell | undefined;
	let paper: Paper | undefined;
	if (req.paper && papersFor(robot, req.paper.cell).includes(req.paper)) {
		cell = req.paper.cell;
		paper = req.paper;
	} else if (req.paper) {
		ui.notify(`${req.paper.name}'s setting is not on ${robot.label}; pick a cell`, "warning");
	}
	const ports = req.work ? PAPERS.filter((p) => p.work === req.work?.id) : [];
	if (!cell && req.work) {
		const cells = supportedCells(robot.axes).filter(
			(c) =>
				req.work?.cells.includes(cellKey(c)) &&
				(!ports.length || ports.some((p) => p.cell.o === c.o && p.cell.a === c.a)),
		);
		if (!ports.length)
			ui.notify(
				`${req.work.name} is not ported yet (the survey files it in ${req.work.cells.map((k) => k.replace("/", " × ")).join(", ")}); pick a cell, the agent starts from its flags`,
				"warning",
			);
		else if (cells.length === 1) cell = cells[0];
		else if (cells.length > 1) {
			const opts = cells.map(cellLabel);
			cell = cells[opts.indexOf((await ui.select(`Cell for ${req.work.name}`, opts)) ?? "")];
			if (!cell) return;
		} else ui.notify(`${robot.label} has none of ${req.work.name}'s cells; pick a cell`, "warning");
	}
	const named = requestCell(req);
	if (!cell && named && supported(robot.axes, named)) cell = named;
	if (!cell) cell = await pickCell(ctx, robot);
	if (!cell) return;
	// The work to reproduce: one of the cell's ported settings; a named work narrows them to its own.
	if (!paper) {
		const papers = papersFor(robot, cell).filter((p) => !ports.length || p.work === req.work?.id);
		if (papers.length) {
			const picked = await pickWork(ctx, papers);
			if (!picked) return;
			paper = picked.paper;
		}
	}

	const where =
		req.where === "remote"
			? "a remote"
			: req.where === "here"
				? "here"
				: await ui.select("Where do the services run?", ["here (this machine)", "a remote GPU box over ssh"]);
	if (!where) return;
	let remote: Choice["remote"];
	let services = SERVICES;
	if (where.startsWith("a remote")) {
		const host = req.host ?? (await ui.input("ssh host", "user@gpu-box"))?.trim();
		if (!host) return;
		const checkout = (await ui.input("pi-embodied checkout on the box", "~/pi-embodied"))?.trim() || "~/pi-embodied";
		remote = { host, checkout };
		services = `${checkout}/services`;
	} else if (!existsSync(SERVICES)) {
		const checkout = (
			await ui.input(`pi-embodied checkout (cloned from ${REPO} if missing)`, "~/pi-embodied")
		)?.trim();
		services = `${(checkout || "~/pi-embodied").replace(/^~(?=\/|$)/, homedir())}/services`;
	}

	// The planner: a model of the registry, or (A1 Code, no preset) the robot's fine-tuned adapter.
	const planners = [...new Set([...(current ? [current] : []), ...available])];
	if (cell.a === "A1C" && robot.finetuned && !paper) planners.push(FINETUNED);
	const planner = req.model ?? (await ui.select("Planner model", planners));
	if (!planner) return;
	const finetuned = planner === FINETUNED;
	const model = finetuned ? undefined : planner;
	if (!finetuned && !model?.includes("/")) return;
	// The weights setup.sh --weights fetches serve the VLA of A1 Model and A2, and O2's SAM3.
	const weights =
		robot.weights && (cell.a === "A1M" || cell.a === "A2" || cell.o === "O2")
			? await ui.confirm("Model weights", `Also download ${robot.weights}?`)
			: false;
	const fallback = join(ctx.cwd, `embodied-${robot.id}`);
	// The title names the default: an empty input field shows no placeholder in every TUI.
	const answer = (await ui.input(`Experiment directory (empty: ${fallback})`, fallback))?.trim();
	const dir = resolve(ctx.cwd, (answer || fallback).replace(/^~(?=\/|$)/, homedir()));
	const choice: Choice = { robot, cell, paper, remote, services, weights, model, finetuned, dir };

	const file = join(dir, ".pi", "settings.json");
	let existing: Settings = {};
	if (!remote && existsSync(file)) {
		try {
			existing = JSON.parse(readFileSync(file, "utf8")) as Settings;
		} catch (e) {
			ui.notify(`${file} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`, "error");
			return;
		}
	}
	const root = remote ? `${remote.checkout}/packages/embodied` : PKG;
	const source = remote ? root : packageSource(pi);
	const settings = experimentSettings(choice, source, root, existing);
	const text = `${JSON.stringify(settings, null, "\t")}\n`;
	const summary = [
		...(req.settled.length ? [`from your request: ${req.settled.join("; ")}`] : []),
		...(req.unknown.length ? [`not recognized (maybe not supported yet): ${req.unknown.join(", ")}`] : []),
		`${robot.label} (${robot.kind}): ${settingLine(choice)}`,
		`flags: ${settingFlags(choice) || "none (the robot's defaults)"}; planner ${model ?? FINETUNED}`,
		`needs: ${robot.needs}`,
		remote ? `services on ${remote.host}:${services}` : `services at ${services}`,
		`install: ${services}/setup.sh ${robot.id}${weights ? " --weights" : ""}${finetuned ? ", the adapter download and the vLLM server" : ""}; the agent runs it after this confirmation (pi does not ask per step), then the preflight; no episode is started`,
		remote ? "settings: the agent writes them on the box" : `writes ${file}`,
		`launch (you run it afterwards): ${launchCommand(choice)}`,
	].join("\n");
	if (!(await ui.confirm("Set up pi-embodied?", summary))) {
		ui.notify("Nothing written or installed.", "info");
		return;
	}
	if (!remote) {
		mkdirSync(dirname(file), { recursive: true });
		writeFileSync(file, text);
		// What /embodied-run reads back: the robot, the setting and its flags.
		const record: SetupRecord = {
			robot: robot.id,
			cell,
			paper: paper?.id,
			flags: settingFlags(choice),
			task: robot.task,
			services,
			planner: model ?? "finetuned",
			created: new Date().toISOString(),
		};
		writeFileSync(join(dir, RECORD), `${JSON.stringify(record, null, "\t")}\n`);
		ui.notify(`Wrote ${file}`, "info");
	}
	pi.sendUserMessage(handoff(choice, text), {
		expandPromptTemplates: true,
		...(ctx.isIdle() ? {} : { deliverAs: "followUp" as const }),
	});
}

/**
 * /embodied-run [request]: an episode of an installed experiment. The experiment directories here (this
 * one and its subdirectories) are listed; a request naming a robot picks among them. The setting is the
 * recorded one, else the grid and the picker as in /embodied-setup. Then the task flags, a GPU, the budget,
 * and how: in this terminal (pi takes it over until the episode ends), headless by the agent, or printed.
 */
/** What /embodied-run reads off the machine; a test replaces it. */
export const runDeps = { gpus };

async function run(pi: ExtensionAPI, ctx: ExtensionCommandContext, args = ""): Promise<void> {
	if (robotActive(pi)) {
		ctx.ui.notify("A robot is already loaded here: this is an episode session; type Solve the task.", "warning");
		return;
	}
	if (!ctx.hasUI) {
		ctx.ui.notify("/embodied-run needs an interactive session", "error");
		return;
	}
	const ui = ctx.ui;
	const all = scanExperiments(ctx.cwd, ROBOTS);
	if (!all.length) {
		ui.notify(
			`no experiment here: no .pi/settings.json loading a robot in ${ctx.cwd} or its subdirectories; run /embodied-setup first`,
			"error",
		);
		return;
	}
	const req = parseRequest(
		args,
		ROBOTS,
		ctx.modelRegistry.getAvailable().map((m) => `${m.provider}/${m.id}`),
	);
	const named = req.robots ? all.filter((e) => req.robots?.some((r) => e.robots.includes(r))) : all;
	const pool = named.length ? named : all;
	if (req.robots && !named.length)
		ui.notify(`no experiment here for ${req.robots.map((r) => r.label).join(" / ")}; these are installed`, "warning");
	let exp: Experiment<Robot> | undefined = pool.length === 1 ? pool[0] : undefined;
	if (!exp) {
		const opts = pool.map(experimentLabel);
		exp = pool[opts.indexOf((await ui.select("Experiment", opts)) ?? "")];
	}
	if (!exp) return;
	if (!exp.env) {
		ui.notify(
			`${exp.robot.label} is not installed: ${exp.services}/.venv-${exp.robot.id}/pi-embodied.env is missing; run /embodied-setup`,
			"error",
		);
		return;
	}
	// The setting: recorded, else chosen as in /embodied-setup (a request's setting or levels count).
	let setting = exp.record?.flags;
	let settingName = exp.record ? `${cellName(exp.record.cell)}${exp.record.paper ? `, ${exp.record.paper}` : ""}` : "";
	if (setting === undefined || req.paper || requestCell(req)) {
		const robot = exp.robot;
		let cell: Cell | undefined;
		let paper: Paper | undefined;
		if (req.paper && papersFor(robot, req.paper.cell).includes(req.paper)) {
			cell = req.paper.cell;
			paper = req.paper;
		} else {
			const n = requestCell(req);
			cell = n && supported(robot.axes, n) ? n : await pickCell(ctx, robot);
			if (!cell) return;
			const papers = papersFor(robot, cell);
			if (papers.length) {
				const picked = await pickWork(ctx, papers);
				if (!picked) return;
				paper = picked.paper;
			}
		}
		setting = paper ? paper.flag : cellFlags(robot.axes, cell);
		settingName = `${cellName(cell)}${paper ? `, ${paper.id}` : ""}`;
	}
	const task = (
		await ui.input(`Task flags (empty: ${exp.record?.task ?? exp.robot.task})`, exp.record?.task ?? exp.robot.task)
	)?.trim();
	if (task === undefined) return;
	const taskFlags = task || exp.record?.task || exp.robot.task;
	// A GPU, the freest first; none when nvidia-smi says nothing.
	let gpu: number | undefined;
	const devices = runDeps.gpus();
	if (devices.length > 1) {
		const opts = [...devices.map(gpuLabel), "any: no CUDA_VISIBLE_DEVICES"];
		const g = await ui.select("GPU", opts);
		if (g === undefined) return;
		gpu = devices[opts.indexOf(g)]?.index;
	} else if (devices.length === 1) gpu = devices[0].index;
	const budget = (await ui.input(`Episode budget (empty: ${BUDGET})`, BUDGET))?.trim();
	if (budget === undefined) return;
	const flags = episodeFlags(taskFlags, setting, budget || BUDGET);
	const command = runCommand(exp, flags, gpu);
	const hows =
		ctx.mode === "tui" ? [RUN_HOW.here, RUN_HOW.headless, RUN_HOW.print] : [RUN_HOW.headless, RUN_HOW.print];
	const how = await ui.select("How to run?", hows);
	if (!how) return;
	const summary = [
		`${exp.robot.label}: ${settingName}; planner ${exp.planner ?? exp.record?.planner ?? "the session's model"}`,
		`task: ${taskFlags}`,
		gpu === undefined ? "GPU: any" : `GPU ${gpu}`,
		how.split(":")[0] === "headless" ? headlessCommand(exp, flags, gpu) : command,
	].join("\n");
	if (!(await ui.confirm("Start the episode?", summary))) {
		ui.notify("Nothing started.", "info");
		return;
	}
	if (how === RUN_HOW.print) {
		ui.notify(command, "info");
		return;
	}
	if (how === RUN_HOW.headless) {
		pi.sendUserMessage(
			[
				`/skill:embodied-quickstart Run one episode now, in the background, and report its result: \`${headlessCommand(exp, flags, gpu)}\``,
				"- it prints the dashboard URL at startup and the robot's result at the end; when it is over, read the session's robot_result (terminated is the environment's verdict, claimed the agent's) and tell me both, with the dashboard URL and the session file",
				"- do not install or change anything; if it fails to start, show me the error and stop",
			].join("\n"),
			{ expandPromptTemplates: true, ...(ctx.isIdle() ? {} : { deliverAs: "followUp" as const }) },
		);
		return;
	}
	// Here: the TUI stops, pi runs the episode in this terminal, the TUI returns when it exits.
	const status = await ui.custom<number | null>((tui, _theme, _kb, done) => {
		tui.stop?.();
		process.stdout.write("\x1b[2J\x1b[H");
		const r = spawnSync(process.env.SHELL || "/bin/bash", ["-lc", command], { stdio: "inherit", env: process.env });
		tui.start?.();
		tui.requestRender?.(true);
		done(r.status);
		return { render: () => [], invalidate: () => {} };
	});
	ui.notify(
		status === 0 ? `episode session ended (${exp.dir})` : `pi exited with ${status ?? "a signal"}: ${command}`,
		status === 0 ? "info" : "warning",
	);
}

export default function embodiedSetup(pi: ExtensionAPI) {
	pi.registerCommand(RUN, {
		description:
			"Start an episode of an installed experiment: the directory, its recorded setting, task flags, a GPU, then in this terminal, headless by the agent, or printed",
		handler: (args, ctx) => run(pi, ctx, args),
	});
	pi.registerCommand(COMMAND, {
		description:
			"Set up pi-embodied: pick a robot, its observation × execution cell, where services run and the planner; the agent installs. A request after it (robot, work, cell, planner, host) settles what it names",
		handler: (args, ctx) => setup(pi, ctx, args),
	});
	pi.registerCommand(ALIAS, {
		description: "The same as /embodied-setup: `/embodied-quickstart reproduce RPent on LIBERO-PRO`",
		handler: (args, ctx) => setup(pi, ctx, args),
	});
	pi.on("session_start", (_event, ctx) => {
		if (ctx.mode !== "tui" || !ctx.hasUI || robotActive(pi) || noticeSeen(ctx)) return;
		ctx.ui.notify(NOTICE, "info");
		pi.appendEntry(NOTICE_ENTRY, { shown: new Date().toISOString() });
	});
}
