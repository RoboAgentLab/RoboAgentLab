/**
 * Which skills a robot session lists (src/infra/tiers.ts `Preset.skills`): the package manifest
 * loads only embodied-quickstart, and OpenETA's task skills (skills/openeta/, adapted from its
 * agent/skills/*.md) reach a session through the robot's `resources_discover` answer only under
 * --preset openeta. The names come from pi's own skill loader over the manifest's entries plus the
 * paths the robot answers with, as the resource loader merges them.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { type ExtensionAPI, loadSkills } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { PRESETS, SKILLS } from "../src/infra/tiers.ts";
import { deployFlags } from "./helpers/deployment.ts";

// The toy robot's primitive manifest (./fixtures/manifests/toy.json).
process.env.PI_EMBODIED_MANIFESTS = new URL("./fixtures/manifests/", import.meta.url).pathname;

import { defineRobot } from "../src/robot.ts";

type Handler = (event: any, ctx: any) => unknown;
const PKG = new URL("../", import.meta.url).pathname;
const OPENETA = ["embodied-object-memory", "embodied-pick", "embodied-place", "embodied-push-pull", "embodied-stack"];

/** A stub pi that runs handlers in registration order and reads flags from `flagValues`. */
function fakePi(flagValues: Record<string, unknown> = {}) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: () => {},
		registerCommand: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = { hasUI: false, ui: { notify: () => {} }, shutdown: () => {}, sessionManager: { getBranch: () => [] } };
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) {
			const r: any = await fn({ type: name, ...event }, ctx);
			if (r !== undefined) result = r;
		}
		return result;
	}
	return { pi, emit };
}

/**
 * The skill names a session with `flags` lists (sorted): the manifest's skills, then what the robot's
 * resources_discover adds. The toy robot mounts VDM unless `vdm` is false (then it cannot serve --preset openeta).
 */
async function listed(flags: Record<string, unknown>, vdm = true): Promise<string[]> {
	const f = fakePi(flags);
	const stderr: string[] = [];
	const error = console.error;
	console.error = (...a: unknown[]) => stderr.push(a.map(String).join(" "));
	try {
		defineRobot(f.pi, {
			name: "toy",
			manifest: "toy",
			task: [],
			keepImages: 2,
			start: async () => ["move_to", "segment"],
			result: () => ({}),
			...(vdm ? { vdm: { views: 1 } } : {}),
			finish: {
				description: "finish",
				parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
				result: (p) => ({ content: [{ type: "text", text: p.status }], details: p }),
			},
		});
		await f.emit("session_start");
	} finally {
		console.error = error;
		process.exitCode = undefined;
	}
	const discovered = (await f.emit("resources_discover", { cwd: PKG, reason: "startup" })) as
		| { skillPaths?: string[] }
		| undefined;
	const manifest = JSON.parse(readFileSync(join(PKG, "package.json"), "utf8")) as { pi: { skills: string[] } };
	const skillPaths = [...manifest.pi.skills.map((p) => join(PKG, p)), ...(discovered?.skillPaths ?? [])];
	const loaded = loadSkills({ cwd: PKG, agentDir: join(PKG, "test/fixtures"), skillPaths, includeDefaults: false });
	assert.deepEqual(loaded.diagnostics, [], "every skill path loads cleanly");
	return loaded.skills.map((s) => s.name).sort();
}

test("the openeta preset names the directory of OpenETA's skills, and only it does", () => {
	assert.equal(PRESETS.openeta.skills, "openeta");
	for (const [name, p] of Object.entries(PRESETS)) if (name !== "openeta") assert.equal(p.skills, undefined, name);
	assert.equal(SKILLS, join(PKG, "skills"));
});

test("without the preset a session lists only embodied-quickstart", async () => {
	assert.deepEqual(await listed({}), ["embodied-quickstart"]);
	// Another preset or a tier brings none of OpenETA's.
	assert.deepEqual(await listed({ preset: "showharness" }), ["embodied-quickstart"]);
	assert.deepEqual(await listed({ tier: "M2" }), ["embodied-quickstart"]);
});

test("--preset openeta lists the quickstart and OpenETA's five task skills, their licence lines kept", async () => {
	assert.deepEqual(await listed({ preset: "openeta" }), [...OPENETA, "embodied-quickstart"].sort());
	for (const name of OPENETA) {
		const head = readFileSync(join(SKILLS, "openeta", name, "SKILL.md"), "utf8")
			.split("\n")
			.slice(0, 5);
		assert.equal(head[1], `name: ${name}`);
		assert.match(head[3], /^license: Apache-2\.0 \(adapted from OpenETA agent\/skills\//);
	}
});

test("a refused start (a robot that cannot serve the preset) adds no skills", async () => {
	// --preset openeta needs VDM mounted; without it the robot does not start, and the session stays as it is.
	assert.deepEqual(await listed({ preset: "openeta" }, false), ["embodied-quickstart"]);
});
