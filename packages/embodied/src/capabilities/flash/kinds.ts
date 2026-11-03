/**
 * The replay plan kinds besides a robot's tool-call plan: what `flash/replay` plays when the plan
 * file holds an action-unit sequence (A1) or a program (A3), so the no-model planner drives every
 * surface the LLM planners drive (./index.ts dispatches on the loaded program's `kind`).
 *
 * - `units` (A1): `{"kind": "units", "units": [{"unit": "MV_UP"}, {"unit": "GRASP"}, ...]}` (one
 *   `act` call per entry, with its `n`, `arm`, `other` and `param` as recorded), sent in order as
 *   recorded. A pure replay: units are robot-frame moves with nothing to re-anchor, so the sequence
 *   is meaningful on the recorded scene (the recorded seed) only; the robot's `act` runs each unit
 *   through its own safety checks, and the episode ends at the robot's success, the sequence's end,
 *   or the first failed call.
 * - `program` (A3): a Python file (`<name>_program.py`, header comments `# key: value` kept as the
 *   oracles' are) or `{"kind": "program", "code": "..."}`. Replay = the saved program executed once
 *   through `run_code` (code mode's tier, whitelist and limits, `--code=true`), then `finish` with
 *   the robot's success.
 *
 * `readKind` judges a file by its content, not its name: a JSON object with `units`, a JSON object
 * with `code`, a `.py` file, or a JSONL of `act` lines. Anything else is the robot's own plan
 * (./recipe.ts, ../../robots/libero/flash.ts), left to its hook.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import type { FlashReply, FlashRobot } from "./index.ts";

type Json = Record<string, unknown>;
/** One recorded `act` call. */
export type UnitStep = { unit: string; n?: number; arm?: string; other?: string; param?: string | number };
export type UnitsProgram = { name: string; kind: "units"; units: UnitStep[]; source?: string };
export type CodeProgram = { name: string; kind: "program"; code: string; sha256: string; source?: string };
export type KindProgram = UnitsProgram | CodeProgram;

/** The file names a robot's plan directories are searched for, per kind, before its own plan. */
export const kindFiles = (name: string) => [`${name}_units.json`, `${name}_program.py`, `${name}_program.json`];

const STEP_KEYS = new Set(["unit", "n", "arm", "other", "param"]);
/** One unit entry as recorded (wire input: a plan file); `act`'s own schema checks the rest at replay. */
function unitStep(v: unknown, where: string): UnitStep {
	const o = (typeof v === "string" ? { unit: v } : v) as Json | null;
	if (!o || typeof o !== "object" || typeof o.unit !== "string" || !o.unit)
		throw new Error(`${where}: a unit entry is {"unit": "<name>", ...} or "<name>"`);
	const step: UnitStep = { unit: o.unit };
	for (const k of Object.keys(o)) {
		if (k === "action") continue;
		if (!STEP_KEYS.has(k)) throw new Error(`${where}: unknown unit field ${k}`);
		if (k !== "unit") (step as Json)[k] = o[k];
	}
	return step;
}

const program = (name: string, code: string, source?: string): CodeProgram => ({
	name,
	kind: "program",
	code,
	sha256: createHash("sha256").update(code).digest("hex"),
	...(source ? { source } : {}),
});

/**
 * The units or program plan `path` holds, or undefined when its content is neither (a tool-call
 * plan or recipe of the robot's own kind). Throws on a file that claims a kind and is malformed.
 */
export function readKind(path: string, name: string): KindProgram | undefined {
	const text = readFileSync(path, "utf8");
	if (path.endsWith(".py")) return program(name, text, path);
	const trimmed = text.trim();
	if (!trimmed) return undefined;
	// One JSON object: a units plan, a program, or (a tool plan) none of ours. A JSONL starts with "{" too
	// and fails to parse as a whole: it falls through to the line parser.
	let doc: unknown;
	try {
		doc = JSON.parse(trimmed);
	} catch {
		doc = undefined;
	}
	if (doc && typeof doc === "object" && !Array.isArray(doc)) {
		const o = doc as Json;
		if (Array.isArray(o.units))
			return {
				name,
				kind: "units",
				units: o.units.map((u, i) => unitStep(u, `${path} units[${i}]`)),
				...(typeof o.source === "string" ? { source: o.source } : {}),
			};
		if (typeof o.code === "string") return program(name, o.code, typeof o.source === "string" ? o.source : path);
		return undefined;
	}
	// A JSONL whose every line is an `act` call (what memory records of a units episode).
	const lines = trimmed.split("\n").filter((l) => l.trim());
	const parsed: Json[] = [];
	for (const line of lines) {
		try {
			parsed.push(JSON.parse(line) as Json);
		} catch {
			return undefined;
		}
	}
	if (!parsed.length || !parsed.every((l) => l.action === "act")) return undefined;
	return { name, kind: "units", units: parsed.map((l, i) => unitStep(l, `${path}:${i + 1}`)) };
}

/** Write a units plan: `{name, kind, source, units}`. */
export const unitsPlanJson = (p: UnitsProgram) =>
	`${JSON.stringify({ name: p.name, kind: "units", ...(p.source ? { source: p.source } : {}), units: p.units }, null, 2)}\n`;

/**
 * Replay a unit sequence: one `act` per entry, in order, until the robot reports the episode over or
 * a call fails (`robot.move` throws). The outcome is the robot's own success signal.
 */
export async function replayUnits(
	p: UnitsProgram,
	robot: FlashRobot,
	over: (latest: FlashReply) => boolean,
): Promise<{ sent: number }> {
	let sent = 0;
	robot.note(`pure replay: ${p.units.length} units as recorded (nothing is re-anchored)`);
	for (const u of p.units) {
		if (sent && over(robot.latest())) break;
		await robot.move({ name: "act", arguments: { ...u } });
		sent++;
	}
	return { sent };
}

/** Replay a program: `run_code` once with the saved program; the outcome is the robot's own success signal. */
export async function replayProgram(p: CodeProgram, robot: FlashRobot): Promise<{ status: string }> {
	robot.note(`running the saved program once through run_code (sha256 ${p.sha256.slice(0, 12)})`);
	const reply = await robot.move({ name: "run_code", arguments: { code: p.code } });
	const status = String(reply.details?.status ?? reply.json.status ?? "unknown");
	return { status };
}
