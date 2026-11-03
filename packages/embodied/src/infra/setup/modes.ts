/**
 * What a robot's own spec says it can do, for /embodied-setup: the modes it supports and the levels
 * of the observation × execution grid (./axes.ts) it has. The wizard's robot table holds only what a
 * spec cannot say (labels, setup targets, tasks, weights, the VLA skill flag); everything else is read
 * here, so a robot that gains a mode is offered it without editing the wizard.
 *
 * A robot extension builds its `RobotSpec` inside its default export, so the spec is read the way
 * test/readme-modules.test.ts reads it: the extension runs against a stub `pi`, and the flags the base
 * and its mounted modules register show which keys the spec has (`units`, `code`, `flash`,
 * `motormind`, `groundTruth` register `--units`, `--code`, `--flash-reanchor`, `--mm`, `--privileged`).
 * Nothing starts: `start` is never called, so no env server and no GPU. The robot modules are loaded on
 * first use, not at import: this file is part of the default extension every pi session loads.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Axes } from "./axes.ts";

export type Mode = "tools" | "units" | "code" | "finetuned" | "flash" | "motormind";

/** What the table says about a robot that its spec cannot. */
export type SpecInput = {
	/** The robot extension, relative to the package. */
	extension: string;
	/** The flag of A1 Model's learned executor (a VLA skill); absent: the robot has no such level. */
	a1m?: string;
	/** The survey files the robot under A1 Model only: its units are run by a learned motion model. */
	a1ModelOnly?: boolean;
};

/** The flags each robot extension registers, by extension path (one extension may serve several robots). */
const registered = new Map<string, Promise<Set<string>>>();

/** The flags the extension registers against a stub `pi`, without starting the robot. */
function flagsOf(extension: string): Promise<Set<string>> {
	let flags = registered.get(extension);
	if (!flags) {
		flags = load(extension);
		registered.set(extension, flags);
	}
	return flags;
}

async function load(extension: string): Promise<Set<string>> {
	// A computed path: the robots are loaded when /embodied-setup runs, not when pi starts.
	const mod = (await import(new URL(`../../../${extension}`, import.meta.url).href)) as {
		default: (pi: ExtensionAPI) => void;
	};
	const flags = new Set<string>();
	const stub: Record<string, unknown> = {
		registerFlag: (name: string) => flags.add(name),
		getFlag: () => undefined,
		getActiveTools: () => [],
		events: { emit: () => {}, on: () => () => {} },
	};
	// Every other pi method is a no-op; the cast stands for that stub (a robot extension needs only these).
	mod.default(new Proxy(stub, { get: (t, k: string) => t[k] ?? (() => {}) }) as unknown as ExtensionAPI);
	return flags;
}

/**
 * The modes a robot supports, in this order: `tools` always; `units` when its spec has `units`; `code`
 * when it has `code`; `finetuned` when it has `units` (the student speaks the robot's units, tools or
 * code on any robot that has them, `--ft-surface`; a released adapter only makes it ready to run, and
 * the wizard offers it where the table names the adapter's flags); `flash` when it has `flash`;
 * `motormind` when it has `motormind`.
 */
export async function modesOf(robot: SpecInput): Promise<Mode[]> {
	const flags = await flagsOf(robot.extension);
	const modes: Mode[] = ["tools"];
	if (flags.has("units")) modes.push("units");
	if (flags.has("code")) modes.push("code");
	if (flags.has("units")) modes.push("finetuned");
	if (flags.has("flash-reanchor")) modes.push("flash");
	if (flags.has("mm")) modes.push("motormind");
	return modes;
}

/**
 * The grid levels a robot has, with the flag each adds (./axes.ts): O1 always; O2 as `--segment` for a
 * robot with its own segmentation flag, else `--detections`, none without SAM3; O3 when the spec has a
 * ground truth; A1 Code when it has `units`; A2 always; A3 when it has `code`; A1 Model when the table
 * names its executor. A robot filed under A1 Model only has no A1 Code, A2 or A3.
 */
export async function axesOf(robot: SpecInput): Promise<Axes> {
	const flags = await flagsOf(robot.extension);
	const axes: Axes = { O1: "" };
	if (flags.has("segment")) axes.O2 = "--segment";
	else if (flags.has("detections")) axes.O2 = "--detections";
	if (flags.has("privileged")) axes.O3 = "--privileged";
	if (!robot.a1ModelOnly) {
		if (flags.has("units")) axes.A1C = "--units=true";
		axes.A2 = "";
		if (flags.has("code")) axes.A3 = "--code=true";
	}
	if (robot.a1m !== undefined) axes.A1M = robot.a1m;
	return axes;
}
