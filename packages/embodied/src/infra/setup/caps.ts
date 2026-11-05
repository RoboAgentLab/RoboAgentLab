/**
 * What a robot can run, read off its code: the modules its extension's spec mounts (`units:`,
 * `motormind:`, `flash:`, ... at the spec's indentation, and a `vlaAlone(pi` call) and its primitive
 * manifest (the VLA skills, the code tiers), the same reading ../../robot.ts makes for ../tiers.ts at
 * start. /embodied-setup offers a ported setting only where the robot would not refuse it: a CaP-X tier
 * needs its API level in the manifest (S1, S2, M1-M3 the high level, which a low/raw-only robot lacks).
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadManifest } from "../../primitives/manifest.ts";
import { adaptersFor } from "../../primitives/vla-adapters.ts";
import { argvFlags } from "../params.ts";
import { type Caps, choose, conflicts, unsupported, VLA_SKILLS } from "../tiers.ts";
import { type Axes, type Cell, type Paper, paperFlag, papersFor } from "./axes.ts";

/** The package root (`packages/embodied`). */
export const PKG = fileURLToPath(new URL("../../..", import.meta.url)).replace(/\/$/, "");

/** What is read of a robot entry (./index.ts `Robot`). */
export type CapsRobot = { id: string; extension: string; axes: Axes };

/** The robot's directory under src/robots (LIBERO's three benchmarks share one). */
export const robotDir = (r: { id: string; extension: string }) => dirname(r.extension).split("/").pop() ?? r.id;

const sources = new Map<string, string>();
/** A source file's text, read once. */
export const source = (file: string) => {
	let s = sources.get(file);
	if (s === undefined) {
		s = existsSync(file) ? readFileSync(file, "utf8") : "";
		sources.set(file, s);
	}
	return s;
};
/** Whether the robot's spec names a module (`units: {`, `flash: hook`, `memory,`): two tabs in, as defineRobot's argument. */
const mounts = (src: string, key: string) => new RegExp(`^\\t\\t${key}(: |,)`, "m").test(src);

/** ../tiers.ts `Caps` with the code tiers as a list, plus Flash, the VLA skills and adapters, and the VLA alone. */
export type RobotCaps = Omit<Caps, "codeTiers"> & {
	codeTiers: string[];
	flash: boolean;
	/** The VLA skills its manifest declares (RPent's `vla`). */
	vlaSkills: string[];
	/** The third-party VLA adapters it can mount (`--vla-adapter <name>`). */
	vlaAdapters: string[];
	/** It runs a VLA alone (`--vla-alone`, ../../planner/vla-alone.ts). */
	vlaAlone: boolean;
};
const capsCache = new Map<string, RobotCaps>();
export function robotCaps(r: { id: string; extension: string }): RobotCaps {
	const hit = capsCache.get(r.extension);
	if (hit) return hit;
	const src = source(join(PKG, r.extension));
	const manifestName = /^\t\tmanifest: "(\w+)"/m.exec(src)?.[1];
	const manifest = manifestName ? loadManifest(manifestName) : undefined;
	const has = (k: string) => mounts(src, k);
	const caps: RobotCaps = {
		robot: robotDir(r),
		code: has("code"),
		units: has("units"),
		motormind: has("motormind"),
		vdm: has("vdm"),
		groundTruth: has("groundTruth"),
		memory: has("memory"),
		xpolicy: has("xpolicy"),
		vla: manifest?.primitives.some((e) => VLA_SKILLS.has(e.name)) ?? false,
		codeTiers: ["high", "low", "raw"].filter((t) =>
			manifest?.primitives.some((e) => e.side !== "ts" && e.doc.code && e.tier === t),
		),
		flash: has("flash"),
		vlaSkills: manifest?.primitives.filter((e) => VLA_SKILLS.has(e.name)).map((e) => e.name) ?? [],
		vlaAdapters: manifestName ? adaptersFor(manifestName).map((a) => a.flag) : [],
		vlaAlone: /\bvlaAlone\(pi\b/.test(src),
	};
	capsCache.set(r.extension, caps);
	return caps;
}
/** ../tiers.ts `Caps` for `unsupported`: the same booleans, the code tiers as its function. */
export const tierCaps = (c: RobotCaps): Caps => ({ ...c, codeTiers: () => c.codeTiers });

/** Why the preset or tier among `flags` cannot run on the robot, as the robot refuses it at start (../tiers.ts). */
export function tierProblems(robot: { id: string; extension: string }, flags: string): string[] {
	const given = argvFlags(flags.split(/\s+/).filter(Boolean));
	const c = choose(given.get("tier"), given.get("preset"));
	if (!c) return [];
	if ("error" in c) return [c.error];
	const why = unsupported(c, tierCaps(robotCaps(robot)));
	return [...conflicts(c, given), ...(why ? [why] : [])];
}

/** The ported settings of a cell the robot runs: offered there (./axes.ts `papersFor`) and not refused at start. */
export const settingsFor = (robot: CapsRobot, c: Cell): Paper[] =>
	papersFor(robot, c).filter((p) => !tierProblems(robot, paperFlag(p, robot.id)).length);
