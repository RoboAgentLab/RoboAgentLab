/**
 * What the robots can run, generated from the code at call time, for the model that reads a free-text
 * `/embodied-setup` or `/embodied-run` request (./interpret.ts). Nothing here is written by hand: the
 * robots' levels and settings come from ./index.ts `ROBOTS` and ./axes.ts `PAPERS`, the preset and tier
 * tables from ../tiers.ts, what a robot mounts from its extension's spec (`units:`, `motormind:`,
 * `flash:`, ... at the spec's indentation, and a `vlaAlone(pi` call) and its primitive manifest, the
 * flags it takes from every literal `registerFlag` its extension and the shared modules make, and the
 * batch command from the header of its `eval.sh`.
 *
 * The same functions back the validation (./interpret.ts): a model's answer is checked against what
 * this module reads, never against a list of its own.
 */

import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { PRESETS, TIERS } from "../tiers.ts";
import {
	AS,
	type Axes,
	type Cell,
	cellFlags,
	cellKey,
	EXECUTION,
	OBSERVATION,
	OS,
	PAPERS,
	type Paper,
	paperTurns,
	supported,
} from "./axes.ts";
import { PKG, robotCaps, robotDir, settingsFor, source } from "./caps.ts";

/** What the catalog needs of a robot entry (./index.ts `Robot`). */
export type CatalogRobot = {
	id: string;
	label: string;
	kind: "sim" | "real";
	extension: string;
	axes: Axes;
	task: string;
	finetuned?: string;
	flash?: boolean;
	vlaAlone?: { flags: string; task: string; needs: string; weights?: string };
};

/** pi's own command-line flags a launch may carry besides the extensions' (../../../../coding-agent/src/cli/args.ts). */
const PI_FLAGS = ["model", "provider", "thinking", "models"];
function walk(dir: string, out: string[] = []): string[] {
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) walk(p, out);
		else if (name.endsWith(".ts")) out.push(p);
	}
	return out;
}
let flagFiles: { rel: string; names: string[] }[] | undefined;
/** Every literal flag name a source file registers: `registerFlag("x"`, a `const X = "x"` it registers, `optionalTools(pi, "x"`. */
function registered(src: string): string[] {
	const names = [...src.matchAll(/registerFlag\(\s*"([a-z0-9-]+)"/g)].map((m) => m[1]);
	for (const m of src.matchAll(/registerFlag\(\s*([A-Z_][A-Z0-9_]*)\b/g)) {
		const v = new RegExp(`const ${m[1]} = "([a-z0-9-]+)"`).exec(src)?.[1];
		if (v) names.push(v);
	}
	names.push(...[...src.matchAll(/optionalTools(?:<[^>]*>)?\(\s*pi,\s*"([a-z0-9-]+)"/g)].map((m) => m[1]));
	return names;
}
/** The flags a robot's launch may name: its own extension's and every shared module's (not another robot's), and pi's. */
export function knownFlags(r: CatalogRobot): Set<string> {
	flagFiles ??= walk(join(PKG, "src")).map((f) => ({
		rel: relative(join(PKG, "src"), f),
		names: registered(source(f)),
	}));
	const own = `robots/${robotDir(r)}/`;
	const out = new Set(PI_FLAGS);
	for (const f of flagFiles)
		if (!f.rel.startsWith("robots/") || f.rel.startsWith(own)) for (const n of f.names) out.add(n);
	return out;
}

/** The robot's batch script (repo-relative), its usage lines and the environment variables its header names. */
export type EvalUsage = { script: string; usage: string[]; env: string[] };
export function evalUsage(r: CatalogRobot): EvalUsage | undefined {
	const rel = `src/robots/${robotDir(r)}/eval.sh`;
	const src = source(join(PKG, rel));
	if (!src) return undefined;
	const header: string[] = [];
	for (const line of src.split("\n").slice(1)) {
		if (!line.startsWith("#")) break;
		header.push(line.replace(/^#\s?/, ""));
	}
	const usage = header
		.filter((l) => /^\s*eval\.sh\b/.test(l) || /\b[A-Z][A-Z0-9_]{2,}=\S/.test(l))
		.map((l) => l.trim());
	const env = [...new Set(header.flatMap((l) => [...l.matchAll(/\b([A-Z][A-Z0-9_]{2,})=\S/g)].map((m) => m[1])))];
	return { script: `packages/embodied/${rel}`, usage, env };
}

/** The fine-tuned student's action surface (../../modes/finetuned `--ft-surface`). */
export type Surface = "units" | "tools" | "code";
/** The settings whose own planner loop the student and Flash do not stand in for: MotorMind's roles, HumanCLAW's paper mode. */
const OWN_LOOP = new Set(["motormind2026", "humanclaw"]);
/** The student's surface for a setting, when it can plan it: A1 Code's units (the robot's released adapter, alone or with Show-Harness), A1 Model's and A2's tools, A3's code. */
export function surfaceFor(robot: CatalogRobot, cell: Cell, paper?: Paper): Surface | undefined {
	if (robot.id === "humanclaw" || (paper && OWN_LOOP.has(paper.work))) return undefined;
	if (cell.a === "A1C") return robot.finetuned && (!paper || paper.id === "showharness") ? "units" : undefined;
	return cell.a === "A3" ? "code" : "tools";
}
/** Whether Flash can replay a plan for a setting: the robot has a Flash hook and the setting no loop of its own. */
export const flashFor = (robot: CatalogRobot, paper?: Paper) =>
	Boolean(robot.flash) && !(paper && OWN_LOOP.has(paper.work));

/** The thinking levels pi takes (`--thinking`). */
export const THINKING = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/** One robot in the catalog: its levels and cells, the ported settings per cell, what it mounts, its batch script. */
function robotEntry(r: CatalogRobot) {
	const caps = robotCaps(r);
	const cells = OS.flatMap((o) => AS.map((a) => ({ o, a }) as Cell)).filter((c) => supported(r.axes, c));
	const ev = evalUsage(r);
	return {
		id: r.id,
		label: r.label,
		kind: r.kind,
		cells: cells.map((c) => ({
			cell: cellKey(c),
			flags: cellFlags(r.axes, c),
			settings: settingsFor(r, c).map((p) => p.id),
		})),
		mounts: {
			units: caps.units,
			code: caps.code,
			code_tiers: caps.codeTiers,
			vdm: caps.vdm,
			ground_truth: caps.groundTruth,
			memory: caps.memory,
			motormind: caps.motormind,
			xpolicy: caps.xpolicy,
			flash: caps.flash,
			vla_skills: caps.vlaSkills,
			vla_adapters: caps.vlaAdapters,
		},
		vla_alone: caps.vlaAlone
			? { mounted: true, setup_flags: r.vlaAlone?.flags ?? null, example_task: r.vlaAlone?.task ?? null }
			: { mounted: false },
		fine_tuned_units_adapter: r.finetuned ?? null,
		example_task_flags: r.task,
		eval: ev ? { script: ev.script, usage: ev.usage, env: ev.env } : null,
	};
}

/** The ported settings: id, name, cell, flags (per robot when one adds to them), the robots, single or multi-turn. */
const settings = () =>
	PAPERS.map((p) => ({
		id: p.id,
		name: p.name,
		cell: cellKey(p.cell),
		flags: p.flag,
		...(p.with ? { robot_flags: p.with } : {}),
		robots: p.robots ?? "any robot with the cell",
		turns: paperTurns(p),
		note: p.note,
	}));

/** What /embodied-setup's reader gets: the grid, every robot, the settings, presets and tiers, the planners. */
export function setupCatalog(robots: CatalogRobot[], planners: { ref: string; name?: string }[], session?: string) {
	return {
		grid: {
			observation: Object.fromEntries(OS.map((o) => [o, OBSERVATION[o].name])),
			execution: Object.fromEntries(AS.map((a) => [a, EXECUTION[a].name])),
		},
		robots: robots.map(robotEntry),
		settings: settings(),
		presets: Object.fromEntries(
			Object.entries(PRESETS).map(([k, v]) => [k, { needs: v.needs, expands_to: v.flags }]),
		),
		capx_tiers: Object.fromEntries(Object.entries(TIERS).map(([k, v]) => [`capx-${k}`, v])),
		planners: {
			registry: planners,
			...(session ? { session } : {}),
			special: {
				finetuned:
					"the fine-tuned student: units in A1 Code (robots with a fine_tuned_units_adapter, alone or with showharness), tools in A1 Model and A2, code in A3; not with motormind or humanclaw",
				"flash/replay":
					"replays a recorded plan, no model: robots that mount flash; not with motormind or humanclaw",
			},
		},
		thinking_levels: THINKING,
	};
}

/** An installed experiment as the run reader sees it (./run.ts `Experiment`). */
export type CatalogExperiment = {
	dir: string;
	robot: CatalogRobot;
	record?: { cell?: Cell; paper?: string; flags: string; task: string; planner?: string; vlaAlone?: boolean };
	planner?: string;
	env?: string;
};

/** What /embodied-run's reader gets: the experiments here, each robot's task flags and batch script, the GPUs, the budget. */
export function runCatalog(
	experiments: CatalogExperiment[],
	planners: { ref: string; name?: string }[],
	gpus: { index: number; label: string }[],
	budget: string,
) {
	return {
		experiments: experiments.map((e) => {
			const ev = evalUsage(e.robot);
			return {
				dir: e.dir,
				robot: e.robot.id,
				label: e.robot.label,
				ready: Boolean(e.env),
				setting: e.record
					? {
							cell: e.record.cell ? cellKey(e.record.cell) : null,
							vla_alone: Boolean(e.record.vlaAlone),
							preset_or_tier: e.record.paper ?? null,
							flags: e.record.flags,
						}
					: null,
				planner: e.planner ?? e.record?.planner ?? null,
				task_flags: e.record?.task ?? e.robot.task,
				eval: ev ? { script: ev.script, usage: ev.usage, env: ev.env } : null,
			};
		}),
		planners,
		gpus,
		default_budget: budget,
		thinking_levels: THINKING,
	};
}
