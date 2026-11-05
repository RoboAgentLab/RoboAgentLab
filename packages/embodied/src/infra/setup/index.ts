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
import { deployment, deploymentName } from "../config.ts";
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
	type Panel,
	type Paper,
	type Pick,
	paperFlag,
	paperTurns,
	pickEntries,
	pickerLines,
	pickMatches,
	supported,
	supportedCells,
	typed,
	VISER,
} from "./axes.ts";
import { settingsFor } from "./caps.ts";
import { FINETUNED_REF, FLASH_REF, parseRequest, requestCell } from "./request.ts";
import {
	DASHBOARD_PORT,
	type Experiment,
	episodeFlags,
	experimentLabel,
	freePort,
	gpuLabel,
	gpus,
	headlessCommand,
	matchExperiments,
	namesInstall,
	panelFlags,
	panelUrls,
	RECORD,
	runCommand,
	runOptions,
	type SetupRecord,
	scanExperiments,
	setupRequest,
	tunnelHint,
	withRunOptions,
} from "./run.ts";
import { SELFHOST, type SelfhostSpec, selfhostHandoff, selfhostLabel, selfhostOf, selfhostUp } from "./selfhost.ts";

/** The package root and the services checkout next to it (absent in an npm install). */
export const PKG = fileURLToPath(new URL("../../..", import.meta.url)).replace(/\/$/, "");
const SERVICES = join(PKG, "../../services");
const REPO = "https://github.com/RoboAgentLab/RoboAgentLab.git";
const SKILL = join(PKG, "skills/embodied-quickstart/SKILL.md");
export const NOTICE_ENTRY = "pi-embodied-setup-notice";
export const COMMAND = "embodied-setup";
/** Start an episode of an installed experiment. */
export const RUN = "embodied-run";
export const RUN_HOW = {
	here: "here: pi takes this terminal and starts the episode at once (the trust prompt first, the first time)",
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
/** The budgets /embodied-run offers, the quickstart's first. */
export const BUDGETS = [
	{ value: BUDGET, label: `40 turns or 15 minutes: ${BUDGET}` },
	{
		value: "--max-turns 10 --time-limit 300",
		label: "10 turns or 5 minutes, a quick check: --max-turns 10 --time-limit 300",
	},
	{
		value: "--max-turns 100 --time-limit 2700",
		label: "100 turns or 45 minutes, a long task: --max-turns 100 --time-limit 2700",
	},
];
export const GRID = "Observation × Execution";
export const PAPER = "Work to reproduce (optional)";
export const NO_PAPER = "none: the cell's plain flags";
/** The planner option that runs the robot's fine-tuned adapter instead of a model of the registry (A1 Code only). */
export const FINETUNED = "fine-tuned: a small VLM + LoRA picks each unit (served by vLLM)";
/** The action surface the fine-tuned student speaks (../../modes/finetuned `--ft-surface`): A1 Code's units, A1 Model's and A2's tools, A3's code. */
export type Surface = "units" | "tools" | "code";
export const FINETUNED_AS: Record<Surface, string> = {
	units: FINETUNED,
	tools: "fine-tuned (tools surface): a small VLM + LoRA picks each tool call; no released adapter, you serve one",
	code: "fine-tuned (code surface): a small VLM + LoRA writes each program; no released adapter, you serve one",
};
/** The no-model planner: Flash replays a recorded plan (../../capabilities/flash), on a robot with a Flash hook. */
export const FLASH = `${FLASH_REF}: no model; replays a recorded plan (a solved episode's, or --flash-plan <file>)`;
/** The settings whose own planner loop the student and Flash do not stand in for: MotorMind's roles, HumanCLAW's paper mode. */
const OWN_LOOP = new Set(["motormind2026", "humanclaw"]);

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
	/** Live panels the robot can serve besides the dashboard (the Viser 3D view: LIBERO, the Frankas). */
	panels?: Panel[];
	/** Flash mounts (`--model flash/replay`): the robot has a hook that replays its recorded plans. */
	flash?: boolean;
	/** The VLA-alone baseline (`--vla-alone`, ../../planner/vla-alone.ts): its flags, an example task, what it needs. */
	vlaAlone?: { flags: string; task: string; needs: string };
};

/** A simulator with ground truth, units and code mode; A1 Model needs a VLA skill (LIBERO's Pi0.5: `LIBERO`). */
const SIM: Axes = { O1: "", O2: "--detections", O3: "--privileged", A1C: "--units=true", A2: "", A3: "--code=true" };
const LIBERO: Axes = { ...SIM, A1M: "--require-skills pi0" };
/** A simulator with a VLA skill of its own (RLDX-1, LingBot) besides units and code mode. */
const VLA_SIM = (skill: string): Axes => ({ ...SIM, A1M: `--require-skills ${skill}` });
/** A real Franka: SAM3 through its own --segment, Pi0.5 through --vla (RLinf only), no ground truth. */
const ARM: Axes = { O1: "", O2: "--segment", A1C: "--units=true", A1M: "--vla", A2: "", A3: "--code=true" };
/** The Polymetis backend has no VLA action space (its env server's has_vla is false): no A1 Model. */
const POLYMETIS: Axes = { O1: "", O2: "--segment", A1C: "--units=true", A2: "", A3: "--code=true" };
const SIM_FT = "--model finetuned/qwen3_5_2b_showharness_sim";
const LIBERO_FT = "--model finetuned/local --ft-model <served adapter> --ft-prompt v5-libero";
export const ROBOTS: Robot[] = [
	{
		id: "libero",
		kind: "sim",
		flash: true,
		label: "LIBERO",
		panels: [VISER],
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
		flash: true,
		label: "LIBERO-PRO",
		panels: [VISER],
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
		flash: true,
		label: "LIBERO-plus",
		panels: [VISER],
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
		flash: true,
		label: "ManiSkill 3",
		extension: "src/robots/maniskill/index.ts",
		needs: "NVIDIA GPU (Vulkan)",
		axes: SIM,
		task: "--task PickCube-v1 --seed 0",
		finetuned: SIM_FT,
		adapter: "qwen3_5_2b_sim",
		vlaAlone: {
			flags: "--arm widowx250s --vla-adapter openvla --vla-alone openvla",
			task: "--task PutCarrotOnPlateInScene-v1 --seed 0",
			needs: "the OpenVLA server at services.openvla (openvla-7b with --suite bridge_orig, ~16 GB of GPU; the services' openvla extra) and the bridge_v2_real2sim assets",
		},
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
		flash: true,
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
		flash: true,
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
		flash: true,
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
		flash: true,
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
		panels: [VISER],
		extension: "src/robots/franka/index.ts",
		needs: "real Franka, RLinf controller + Ray, an operator at the e-stop",
		axes: ARM,
		task: "--robot-config <yaml> --task 0 --operator=true",
	},
	{
		id: "franka-polymetis",
		kind: "real",
		label: "Franka (Polymetis)",
		panels: [VISER],
		extension: "src/robots/franka/index.ts",
		needs: "real Franka on a Polymetis NUC, an operator at the e-stop",
		axes: POLYMETIS,
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
	/** The grid's cell; none for the VLA-alone baseline, which has no agent to place. */
	cell?: Cell;
	/** The VLA runs alone (the robot's `vlaAlone`): no planner, no cell. */
	vlaAlone?: boolean;
	/** A ported work's setting, run as its preset or tier. */
	paper?: Paper;
	remote?: { host: string; checkout: string };
	services: string;
	weights: boolean;
	/** The planner of the registry, or none when `finetuned`. */
	model?: string;
	/** The robot's fine-tuned adapter is the planner, speaking `surface` (units: the released adapters). */
	finetuned: boolean;
	surface?: Surface;
	dir: string;
	/** A self-hosted planner (selfhost/<id>), the deployment's entry for it when there is one, and that deployment's name. */
	selfhost?: {
		spec: SelfhostSpec;
		deploy?: NonNullable<ReturnType<typeof deployment>["selfhost"]>[string];
		deployment: string;
	};
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

/** The fine-tuned student's flags: the robot's released units adapter, else an adapter the user serves on that surface. */
const finetunedFlags = (c: Choice) =>
	(c.surface ?? "units") === "units"
		? c.robot.finetuned
		: `--model finetuned/local --ft-model <served adapter> --ft-surface ${c.surface}`;

/** The setting's flags: the VLA alone's, else the ported work's preset or tier, else the cell's; then a planner's that is no registry model. */
export function settingFlags(c: Choice): string {
	if (c.vlaAlone) return c.robot.vlaAlone?.flags ?? "";
	const cell = c.paper ? paperFlag(c.paper, c.robot.id) : c.cell ? cellFlags(c.robot.axes, c.cell) : "";
	const planner = c.finetuned ? finetunedFlags(c) : c.model === FLASH_REF ? `--model ${FLASH_REF}` : undefined;
	return [cell, planner].filter(Boolean).join(" ");
}

/** The launch flags for a choice (after `pi`), with the quickstart's episode budget. */
export function launchFlags(c: Choice): string {
	const task = c.vlaAlone ? c.robot.vlaAlone?.task : c.robot.task;
	return [task, settingFlags(c), BUDGET, "--dashboard=true"].filter(Boolean).join(" ");
}

/** The planner in a few words: a registry model, the student, Flash, or none (the VLA alone). */
export const plannerName = (c: Choice) =>
	c.vlaAlone ? "none (the VLA alone)" : c.finetuned ? FINETUNED_AS[c.surface ?? "units"] : (c.model ?? FINETUNED);

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
	// Flash's provider is the robot's own (registered when it loads): the launch flags name it.
	if (c.model && c.model !== FLASH_REF) {
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
	if (!c.cell) return "the VLA alone, no agent (the VLA-alone baseline, outside the observation × execution grid)";
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
		c.cell?.o === "O2"
			? [
					`- perception (O2): a SAM3 server at the deployment's services.sam3 must be running for the episodes (\`--serve-models sam3\` per episode, or the robot's serve.sh; SAM3 is gated: HF_TOKEN)`,
				]
			: [];
	// Fine-tuned planner: the adapter and its base model, and the vLLM server services.finetuned (default :8010) names.
	const units = (c.surface ?? "units") === "units";
	const ft = c.robot.adapter && units ? `FT_ADAPTER=${c.robot.adapter} ${c.services}/setup.sh finetuned` : undefined;
	const finetuned = c.finetuned
		? [
				ft
					? `- adapter: \`${ft}\` (finetuned/download.py fetches the adapter and its base model, pinned and verified)`
					: units
						? "- adapter: none is released for this robot; ask me which adapter to serve and where its files are"
						: `- adapter: none is released for the ${c.surface} surface (sessions_to_sft.ts --surface ${c.surface} makes training pairs from teacher sessions); ask me which adapter to serve and where its files are`,
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
		...plannerLines(c),
		...(c.selfhost ? selfhostHandoff(c.selfhost.spec, c.selfhost.deploy, c.selfhost.deployment) : []),
		...finetuned,
		`- experiment dir: ${c.dir}; ${settings}`,
		`- launch: \`${launchCommand(c)}\` (I start episodes with it myself, afterwards)`,
		"- trust: the first pi in the experiment dir asks whether to trust the project (its .pi/settings.json loads the robot extension); I answer yes there, or the robot does not load (/trust saves the decision)",
		"Run the install, then the preflight, fix what fails, and end with the launch command. Do not start an episode. Ask me before sudo, anything touching real hardware, or downloads not listed here.",
	].join("\n");
}

/** The handoff's planner lines: what plans, and what it needs before an episode. */
function plannerLines(c: Choice): string[] {
	if (c.vlaAlone)
		return [
			`- planner: none; the VLA runs alone (\`--vla-alone\`) and no model is asked; it needs ${c.robot.vlaAlone?.needs}`,
		];
	if (c.model === FLASH_REF)
		return [
			`- planner: ${FLASH}`,
			"- plan: flash/replay needs a recorded plan before an episode: the robot's plan directories (dirs.flash_plans, else the memory's flash/ and task_only/) or --flash-plan <file>; capabilities/flash/generate.ts writes one from a solved session; tell me when there is none yet",
		];
	return [`- planner: ${plannerName(c)}`];
}

/** The fine-tuned student's surface for a setting, when it can plan it: none for a setting with its own loop. */
export function surfaceFor(robot: Robot, cell: Cell, paper?: Paper): Surface | undefined {
	if (robot.id === "humanclaw" || (paper && OWN_LOOP.has(paper.work))) return undefined;
	if (cell.a === "A1C") return robot.finetuned && (!paper || paper.id === "showharness") ? "units" : undefined;
	return cell.a === "A3" ? "code" : "tools";
}
/** Whether Flash can replay a plan for a setting: the robot has a Flash hook and the setting no loop of its own. */
export const flashFor = (robot: Robot, paper?: Paper) => Boolean(robot.flash) && !(paper && OWN_LOOP.has(paper.work));

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

export const PLANNER = "Planner model";
/**
 * A searchable list: typing filters by id or text (case-insensitive), the arrows move, enter picks, escape
 * cancels (undefined). A custom component in the TUI (its `options` are the entries' texts); a select elsewhere.
 */
async function searchPick<T extends Pick>(
	ctx: ExtensionCommandContext,
	title: string,
	entries: T[],
): Promise<T | undefined> {
	if (ctx.mode !== "tui") {
		const picked = await ctx.ui.select(
			title,
			entries.map((e) => e.text),
		);
		return entries.find((x) => x.text === picked);
	}
	return ctx.ui.custom<T | undefined>((_tui, theme, keybindings, done) => {
		let q = "";
		let idx = 0;
		const paint: Paint = {
			accent: (s) => theme.fg("accent", s),
			dim: (s) => theme.fg("dim", s),
			bold: (s) => theme.bold(s),
		};
		return {
			options: entries.map((e) => e.text),
			render: () => pickerLines(title, entries, q, idx, paint),
			invalidate: () => {},
			handleInput: (data: string) => {
				const shown = pickMatches(entries, q) as T[];
				const key = keyFrom(data, keybindings, false);
				if (key === "enter") {
					if (shown[idx]) done(shown[idx]);
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

/** The mark of the option a menu suggests, listed first: taking it is never wrong for a first run. */
export const RECOMMENDED = " (recommended)";
export const rec = (s: string) => `${s}${RECOMMENDED}`;
/** The menu option that asks for a value instead of picking one. */
export const OTHER = "other: type it";
/**
 * A value chosen from a menu: the suggested ones (the first marked recommended unless `recommend` is
 * false), and OTHER, which asks for it as text. Undefined when cancelled.
 */
async function chooseValue(
	ctx: ExtensionCommandContext,
	title: string,
	values: { value: string; label?: string }[],
	recommend = true,
): Promise<string | undefined> {
	const labels = values.map((v, i) => (i === 0 && recommend ? rec(v.label ?? v.value) : (v.label ?? v.value)));
	const picked = await ctx.ui.select(title, [...labels, OTHER]);
	if (picked === undefined) return undefined;
	if (picked === OTHER) return (await ctx.ui.input(title, values[0]?.value ?? ""))?.trim() || undefined;
	return values[labels.indexOf(picked)]?.value;
}
/** The hosts of ~/.ssh/config (no patterns): the remote boxes the user already reaches. */
export function sshHosts(file = join(homedir(), ".ssh", "config")): string[] {
	try {
		const out = new Set<string>();
		for (const m of readFileSync(file, "utf8").matchAll(/^\s*Host\s+(.+)$/gim))
			for (const h of m[1].trim().split(/\s+/)) if (!/[*?!]/.test(h)) out.add(h);
		return [...out];
	} catch {
		return [];
	}
}
/** What /embodied-setup reads off the machine; a test replaces it. */
export const setupDeps = { sshHosts: () => sshHosts() };

type Picked = { paper?: Paper };
/** The work to reproduce: one of the cell's ported settings, or none. Cancelled: undefined. */
async function pickWork(ctx: ExtensionCommandContext, papers: Paper[], robot = ""): Promise<Picked | undefined> {
	const e = await searchPick(ctx, PAPER, pickEntries(papers, NO_PAPER, robot));
	return e ? { paper: e.paper } : undefined;
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
	// The registry's models with their display names: a request may name one either way (Claude Fable 5.1).
	const refs = ctx.modelRegistry.getAvailable().map((m) => ({ ref: `${m.provider}/${m.id}`, name: m.name }));
	const available = refs.map((m) => m.ref);
	if (!available.length) {
		ui.notify(NO_MODEL, "error");
		return;
	}
	// A request settles what it names; the rest is asked. A name nothing matches may be something not supported yet.
	// What only an episode uses (task 3, gpu 5, headless) is /embodied-run's: the same text works there.
	const req = parseRequest(runOptions(args).rest, ROBOTS, [...refs, ...SELFHOST.map((x) => `selfhost/${x.id}`)]);
	if (req.unknown.length)
		ui.notify(
			`not recognized: ${req.unknown.join(", ")}; no robot, ported setting, survey work, level or planner by that name (maybe not supported yet). The rest is asked.`,
			"warning",
		);
	// The VLA alone: only robots that run --vla-alone; a named robot that does not is refused, not misread as A1 Model.
	const alone = req.vlaAlone ? (req.robots ?? ROBOTS).filter((r) => r.vlaAlone) : undefined;
	if (alone && !alone.length) {
		ui.notify(
			`the VLA alone (no agent) is not supported on ${req.robots?.map((r) => r.label).join(" / ")}: only ${ROBOTS.filter(
				(r) => r.vlaAlone,
			)
				.map((r) => r.label)
				.join(
					", ",
				)} runs a VLA alone (--vla-alone: OpenVLA on the Bridge WidowX). Nothing set up; for the agent driving the robot's VLA, ask for A1 Model.`,
			"error",
		);
		return;
	}
	let kind = req.kind ?? (alone?.every((r) => r.kind === alone[0].kind) ? alone[0].kind : undefined);
	if (!kind) {
		const kindOpts = Object.values(KINDS);
		kind = (Object.keys(KINDS) as Kind[])[
			kindOpts.indexOf((await ui.select("Simulation or real robot?", kindOpts)) ?? "")
		];
		if (!kind) return;
	}
	// One named robot is taken; several named (franka) narrow the menu to them.
	const robots = alone ?? req.robots ?? ROBOTS.filter((r) => r.kind === kind);
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
	if (req.paper && settingsFor(robot, req.paper.cell).includes(req.paper)) {
		cell = req.paper.cell;
		paper = req.paper;
	} else if (req.paper) {
		ui.notify(`${req.paper.name}'s setting is not on ${robot.label}; pick a cell`, "warning");
	}
	// Single or multi-turn narrows a work's ported settings (CaP-X multi-turn: M1–M4), when any are left.
	const byTurns = (ps: Paper[]) => {
		const kept = ps.filter((p) => !req.turns || paperTurns(p) === req.turns);
		return kept.length ? kept : ps;
	};
	const ports = req.work ? byTurns(PAPERS.filter((p) => p.work === req.work?.id)) : [];
	if (!cell && req.work && !req.vlaAlone) {
		const cells = supportedCells(robot.axes).filter(
			(c) =>
				req.work?.cells.includes(cellKey(c)) &&
				(!ports.length || settingsFor(robot, c).some((p) => ports.includes(p))),
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
	const vlaAlone = Boolean(alone);
	if (!vlaAlone) {
		if (!cell && named && supported(robot.axes, named)) cell = named;
		if (!cell) cell = await pickCell(ctx, robot);
		if (!cell) return;
		// The work to reproduce: one of the cell's ported settings; a named work narrows them to its own.
		if (!paper) {
			const papers = byTurns(settingsFor(robot, cell).filter((p) => !ports.length || ports.includes(p)));
			if (papers.length) {
				const picked = await pickWork(ctx, papers, robot.id);
				if (!picked) return;
				paper = picked.paper;
			}
		}
	}

	const where =
		req.where === "remote"
			? "a remote"
			: req.where === "here"
				? "here"
				: await ui.select("Where do the services run?", [rec("here (this machine)"), "a remote GPU box over ssh"]);
	if (!where) return;
	let remote: Choice["remote"];
	let services = SERVICES;
	if (where.startsWith("a remote")) {
		// The hosts of the user's ssh config are the choice; none: typed.
		const hosts = setupDeps.sshHosts();
		const host =
			req.host ??
			(hosts.length
				? await chooseValue(
						ctx,
						"ssh host",
						hosts.map((h) => ({ value: h })),
						false,
					)
				: (await ui.input("ssh host", "user@gpu-box"))?.trim());
		if (!host) return;
		const checkout = await chooseValue(ctx, "pi-embodied checkout on the box", [{ value: "~/pi-embodied" }]);
		if (!checkout) return;
		remote = { host, checkout };
		services = `${checkout}/services`;
	} else if (!existsSync(SERVICES)) {
		const checkout = await chooseValue(ctx, `pi-embodied checkout (cloned from ${REPO} if missing)`, [
			{ value: "~/pi-embodied" },
		]);
		if (!checkout) return;
		services = `${checkout.replace(/^~(?=\/|$)/, homedir())}/services`;
	}

	// The planner: a model of the registry, a self-hosted planner the agent starts (selfhost/<id>, offered
	// whether or not models.json has it yet), the fine-tuned student on the cell's surface (A1 Code: the
	// robot's released units adapter, alone or with Show-Harness's preset; A1 Model and A2: tools; A3:
	// code), or Flash's replay of a recorded plan. The VLA alone has none.
	const registry = [...new Set([...(current ? [current] : []), ...available])].filter((m) => !selfhostOf(m));
	const hosted = SELFHOST.map(selfhostLabel);
	const planners = [...registry, ...hosted];
	const surface = cell ? surfaceFor(robot, cell, paper) : undefined;
	if (surface) planners.push(FINETUNED_AS[surface]);
	if (cell && flashFor(robot, paper)) planners.push(FLASH);
	// A planner that is no registry model, named in the request: taken where this setting offers it.
	const asked =
		req.model === FINETUNED_REF
			? surface && FINETUNED_AS[surface]
			: req.model === FLASH_REF
				? planners.includes(FLASH) && FLASH
				: req.model;
	if (req.model && !asked && !vlaAlone)
		ui.notify(
			`${req.model === FINETUNED_REF ? "the fine-tuned student" : FLASH_REF} is not offered for this setting on ${robot.label}; pick a planner`,
			"warning",
		);
	// Searchable: the registry can list many models; typing filters them. This session's model is the suggestion.
	const picked = vlaAlone
		? ""
		: asked ||
			(
				await searchPick(
					ctx,
					PLANNER,
					planners.map((m) => ({ id: m, text: m === current ? rec(m) : m })),
				)
			)?.id;
	if (picked === undefined || (!vlaAlone && !picked)) return;
	const hostedSpec = vlaAlone ? undefined : (SELFHOST[hosted.indexOf(picked)] ?? selfhostOf(picked));
	const planner = hostedSpec ? `selfhost/${hostedSpec.id}` : picked;
	const ftSurface = (Object.keys(FINETUNED_AS) as Surface[]).find((k) => FINETUNED_AS[k] === planner);
	const finetuned = ftSurface !== undefined;
	const model = finetuned || vlaAlone ? undefined : planner === FLASH ? FLASH_REF : planner;
	if (!finetuned && !vlaAlone && !model?.includes("/")) return;
	// The weights setup.sh --weights fetches serve the VLA of A1 Model and A2, and O2's SAM3.
	const weights =
		robot.weights && cell && (cell.a === "A1M" || cell.a === "A2" || cell.o === "O2")
			? await ui.confirm(
					"Model weights",
					`Also download ${robot.weights}? (recommended: this cell's tools need them)`,
				)
			: false;
	const fallback = join(ctx.cwd, `embodied-${robot.id}`);
	const answer = await chooseValue(ctx, "Experiment directory", [{ value: fallback }]);
	if (!answer) return;
	const dir = resolve(ctx.cwd, answer.replace(/^~(?=\/|$)/, homedir()));
	const selfhost = hostedSpec
		? { spec: hostedSpec, deploy: deployment(pi).selfhost?.[hostedSpec.id], deployment: deploymentName(pi) }
		: undefined;
	const choice: Choice = {
		robot,
		cell,
		vlaAlone,
		paper,
		remote,
		services,
		weights,
		model,
		finetuned,
		surface: ftSurface,
		dir,
		selfhost,
	};

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
		`flags: ${settingFlags(choice) || "none (the robot's defaults)"}; planner ${plannerName(choice)}`,
		...(selfhost
			? [
					selfhost.deploy
						? `planner server: ${selfhost.spec.engine} · ${selfhost.spec.model} at ${selfhost.deploy.base_url}; the agent starts it if it is down${selfhost.deploy.run_on ? ` (its start runs on ${selfhost.deploy.run_on}: you run that command)` : ""} and registers it in models.json`
						: `planner server: ${selfhost.spec.engine} · ${selfhost.spec.model} is not in the deployment yet; the agent sets it up on ${selfhost.spec.hardware}, asking you which box`,
				]
			: []),
		`needs: ${robot.needs}${vlaAlone ? `; ${robot.vlaAlone?.needs}` : ""}`,
		remote ? `services on ${remote.host}:${services}` : `services at ${services}`,
		`install: ${services}/setup.sh ${robot.id}${weights ? " --weights" : ""}${finetuned ? ", the adapter download and the vLLM server" : ""}; the agent runs it after this confirmation (pi does not ask per step), then the preflight; no episode is started`,
		remote ? "settings: the agent writes them on the box" : `writes ${file}`,
		`launch (you run it afterwards): ${launchCommand(choice)}`,
		`or, from ${ctx.cwd}: /${RUN}${args.trim() ? ` ${args.trim()}` : ""}`,
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
			...(vlaAlone ? { vlaAlone: true } : {}),
			paper: paper?.id,
			flags: settingFlags(choice),
			task: (vlaAlone ? robot.vlaAlone?.task : undefined) ?? robot.task,
			services,
			planner: vlaAlone ? "vla" : (model ?? "finetuned"),
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
 * one and its subdirectories) are listed; a request is read as /embodied-setup reads it, and the
 * experiments whose robot and recorded setting it names are the choice (none: the setup command is
 * offered with the same request). The setting is the recorded one, else (a hand-made experiment) the
 * grid and the picker as in /embodied-setup. Then the task flags, a GPU, the budget, and how: in this
 * terminal (pi takes it over until the episode ends), headless by the agent, or printed; a request's
 * task 3, seed 1, libero_90, gpu 5, 20 turns, headless or print answer those instead.
 */
/** The answer to /embodied-run's offer that hands its request to /embodied-setup. */
export const SETUP_NOW = "set it up now with /embodied-setup";
/**
 * Nothing installed matches: say why, and offer /embodied-setup with the same request (the rest is asked
 * there). Setup installs and records; the episode is then this command again.
 */
async function offerSetup(pi: ExtensionAPI, ctx: ExtensionCommandContext, why: string, request: string): Promise<void> {
	const command = `/${COMMAND}${request ? ` ${request}` : ""}`;
	ctx.ui.notify(`${why}\nto install it: ${command}`, "warning");
	const picked = await ctx.ui.select("Not installed", [rec(`${SETUP_NOW}: ${command}`), "no"]);
	if (picked?.startsWith(SETUP_NOW)) await setup(pi, ctx, request);
}

/** What /embodied-run reads off the machine; a test replaces it. */
export const runDeps = { gpus, freePort, env: process.env as NodeJS.ProcessEnv, selfhostUp };
export const PANELS = "Panels";

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
	// The request: what it says about the episode (task, GPU, turns, how) is split off; the rest names what is installed.
	const text = args.trim();
	const { rest, opts } = runOptions(text);
	const req = parseRequest(rest, ROBOTS, [
		...ctx.modelRegistry.getAvailable().map((m) => ({ ref: `${m.provider}/${m.id}`, name: m.name })),
		...SELFHOST.map((x) => `selfhost/${x.id}`),
	]);
	if (req.unknown.length)
		ui.notify(
			`not recognized: ${req.unknown.join(", ")}; no robot, ported setting, survey work, level or planner by that name`,
			"warning",
		);
	if (req.where === "remote")
		ui.notify(
			`/${RUN} starts episodes on this machine${req.host ? `; for ${req.host}, run it there` : ""}`,
			"warning",
		);
	const all = scanExperiments(ctx.cwd, ROBOTS);
	if (!all.length) {
		const why = `no experiment here: no .pi/settings.json loading a robot in ${ctx.cwd} or its subdirectories`;
		if (rest) await offerSetup(pi, ctx, why, text);
		else ui.notify(`${why}; run /embodied-setup first`, "error");
		return;
	}
	// The experiments the request describes; when it names robot or setting and none is installed, setup is next.
	const pool = namesInstall(req) ? matchExperiments(all, req) : all;
	if (!pool.length) {
		await offerSetup(
			pi,
			ctx,
			`nothing installed here is ${req.settled.filter((x) => !x.startsWith("planner")).join("; ")}; installed: ${all.map(experimentLabel).join("; ")}`,
			text,
		);
		return;
	}
	let exp: Experiment<Robot> | undefined = pool.length === 1 ? pool[0] : undefined;
	if (!exp) {
		const labels = pool.map(experimentLabel);
		exp = pool[labels.indexOf((await ui.select("Experiment", labels)) ?? "")];
	}
	if (!exp) return;
	if (!exp.env) {
		await offerSetup(
			pi,
			ctx,
			`${exp.robot.label} is not installed: ${exp.services}/.venv-${exp.robot.id}/pi-embodied.env is missing`,
			namesInstall(req) ? text : setupRequest(exp),
		);
		return;
	}
	// The setting is the recorded one; only a hand-made experiment (no record) has it chosen here, as in /embodied-setup.
	let setting = exp.record?.flags;
	let settingName = exp.record
		? `${exp.record.cell ? cellName(exp.record.cell) : "the VLA alone"}${exp.record.paper ? `, ${exp.record.paper}` : ""}`
		: "";
	let settingPaper: Paper | undefined;
	if (setting === undefined) {
		const robot = exp.robot;
		let cell: Cell | undefined;
		let paper: Paper | undefined;
		if (req.paper && settingsFor(robot, req.paper.cell).includes(req.paper)) {
			cell = req.paper.cell;
			paper = req.paper;
		} else {
			const n = requestCell(req);
			cell = n && supported(robot.axes, n) ? n : await pickCell(ctx, robot);
			if (!cell) return;
			const papers = settingsFor(robot, cell);
			if (papers.length) {
				const picked = await pickWork(ctx, papers, robot.id);
				if (!picked) return;
				paper = picked.paper;
			}
		}
		setting = paper ? paperFlag(paper, robot.id) : cellFlags(robot.axes, cell);
		settingName = `${cellName(cell)}${paper ? `, ${paper.id}` : ""}`;
		settingPaper = paper;
	}
	// A planner the request names replaces the experiment's for this episode.
	if (req.model && req.model !== (exp.planner ?? exp.record?.planner))
		setting = `${setting} --model ${req.model}`.trim();
	// The task: the request's task, seed and suite on the recorded flags; else asked.
	const recordedTask = exp.record?.task ?? exp.robot.task;
	let taskFlags: string;
	if (opts.task !== undefined || opts.seed !== undefined || opts.suite !== undefined)
		taskFlags = withRunOptions(recordedTask, opts);
	else {
		const task = await chooseValue(ctx, "Task flags", [
			{ value: recordedTask, label: `the recorded task: ${recordedTask}` },
		]);
		if (task === undefined) return;
		taskFlags = task;
	}
	// A GPU: the request's, else the freest first; none when nvidia-smi says nothing.
	let gpu: number | undefined = opts.gpu;
	const devices = gpu === undefined ? runDeps.gpus() : [];
	if (devices.length > 1) {
		const choices = [
			...devices.map((d, i) => (i === 0 ? rec(gpuLabel(d)) : gpuLabel(d))),
			"any: no CUDA_VISIBLE_DEVICES",
		];
		const g = await ui.select("GPU", choices);
		if (g === undefined) return;
		gpu = devices[choices.indexOf(g)]?.index;
	} else if (devices.length === 1) gpu = devices[0].index;
	const budget =
		opts.turns !== undefined
			? BUDGET.replace(/--max-turns \d+/, `--max-turns ${opts.turns}`)
			: await chooseValue(ctx, "Episode budget", BUDGETS);
	if (budget === undefined) return;
	// Panels: the work's own (the ported setting's, when the robot serves them), else the robot's, with the dashboard; or the dashboard alone.
	const paperOf = exp.record?.paper ? PAPERS.find((p) => p.id === exp.record?.paper) : undefined;
	const chosen = settingPaper ?? paperOf;
	const own = (chosen?.panels ?? []).filter((p) => exp.robot.panels?.some((q) => q.flags === p.flags));
	const robotPanels = exp.robot.panels ?? [];
	let panels: Panel[] = [];
	if (own.length || robotPanels.length) {
		const withOwn = own.length
			? `${chosen?.name}'s own: ${own.map((p) => p.name).join(", ")}, with the dashboard`
			: `the robot's: ${robotPanels.map((p) => p.name).join(", ")}, with the dashboard`;
		const picked = await ui.select(PANELS, [rec(withOwn), "the dashboard only"]);
		if (!picked) return;
		if (picked === rec(withOwn)) panels = own.length ? own : robotPanels;
	}
	const port = await runDeps.freePort(DASHBOARD_PORT);
	const flags = episodeFlags(taskFlags, setting, budget || BUDGET, panelFlags(port, panels));
	const command = runCommand(exp, flags, gpu);
	const urls = panelUrls(port, panels);
	const tunnel = tunnelHint([port, ...panels.map((p) => p.port)], runDeps.env);
	const hows =
		ctx.mode === "tui" ? [RUN_HOW.here, RUN_HOW.headless, RUN_HOW.print] : [RUN_HOW.headless, RUN_HOW.print];
	// The first way is the suggestion: in this terminal, else headless.
	const howPicked = opts.how ? RUN_HOW[opts.how] : await ui.select("How to run?", [rec(hows[0]), ...hows.slice(1)]);
	const how = howPicked?.replace(RECOMMENDED, "");
	if (!how) return;
	// A self-hosted planner must answer before an episode is worth starting.
	const plannerId = req.model ?? exp.planner ?? exp.record?.planner;
	const hostedPlanner = selfhostOf(plannerId);
	const hostedDeploy = hostedPlanner ? deployment(pi).selfhost?.[hostedPlanner.id] : undefined;
	const hostedLine = !hostedPlanner
		? []
		: !hostedDeploy
			? [
					`planner server: ${hostedPlanner.engine} · ${hostedPlanner.model} is not in the deployment; run /embodied-setup ${setupRequest(exp)} ${plannerId} to set it up`,
				]
			: (await runDeps.selfhostUp(hostedDeploy.base_url))
				? [`planner server: ${hostedPlanner.engine} answers at ${hostedDeploy.base_url}`]
				: [
						`planner server: ${hostedPlanner.engine} does NOT answer at ${hostedDeploy.base_url}; start it first${hostedDeploy.run_on ? ` on ${hostedDeploy.run_on}` : ""}: ${hostedDeploy.start ?? "(no start command in the deployment)"}`,
					];
	const summary = [
		`${exp.robot.label}: ${settingName}; planner ${plannerId ?? "the session's model"}`,
		...hostedLine,
		`task: ${taskFlags}`,
		gpu === undefined ? "GPU: any" : `GPU ${gpu}`,
		`panels on this machine: ${urls.join("; ")}`,
		...(tunnel
			? [`from your own machine, first: ${tunnel} (or a LocalForward of the same ports in its ssh config)`]
			: []),
		how.split(":")[0] === "headless" ? headlessCommand(exp, flags, gpu) : command,
	].join("\n");
	if (!(await ui.confirm("Start the episode?", summary))) {
		ui.notify("Nothing started.", "info");
		return;
	}
	if (how === RUN_HOW.print) {
		ui.notify(tunnel ? `${command}\n${tunnel}` : command, "info");
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
			"Start an episode of an installed experiment. A request after it is read as /embodied-setup reads it (robot, work, cell, planner) to find the experiment, plus task / seed / suite / gpu / turns / headless / print; not installed: /embodied-setup is offered with the same request",
		handler: (args, ctx) => run(pi, ctx, args),
	});
	pi.registerCommand(COMMAND, {
		description:
			"Set up pi-embodied: pick a robot, its observation × execution cell, where services run and the planner; the agent installs. A request after it (robot, work, cell, planner, host) settles what it names",
		handler: (args, ctx) => setup(pi, ctx, args),
	});
	pi.on("session_start", (_event, ctx) => {
		if (ctx.mode !== "tui" || !ctx.hasUI || robotActive(pi) || noticeSeen(ctx)) return;
		ctx.ui.notify(NOTICE, "info");
		pi.appendEntry(NOTICE_ENTRY, { shown: new Date().toISOString() });
	});
}
