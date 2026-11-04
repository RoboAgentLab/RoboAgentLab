/**
 * MotorMind mode on Metaworld (src/robots/metaworld, `motormind` in its spec): the mode's tools on
 * a fake env server speaking services utils/motormind.py's wire format. --preset motormind mounts
 * pure mode on Metaworld (the preset's LIBERO flags are not Metaworld's); mm_act cuts an action by
 * the server's 40 mm tick limit; the episode's env steps and success come from every reply; a
 * rotation comes back refused; home is the episode reset with --mm-home reset; a server started
 * otherwise than the flags fails the start.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import metaworld from "../src/robots/metaworld/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { body, fakeMmServer, type MmServer, mmSession } from "./helpers/motormind-robot.ts";
import { type Call, f32, rgb } from "./sim-stub.ts";

/** Metaworld's own start calls: the meta its cameras are checked against, the reset, the language. */
function base(c: Call): unknown {
	if (c.method === "code.api") return codeApiReply("metaworld", c.kwargs.tier);
	if (c.method === "env.get_env_meta")
		return {
			task: "reach-v3",
			seed: 0,
			metaworld: "3.1.1",
			agentview: "corner4",
			wrist: "gripperPOV",
			view_size: 256,
			workspace: { min: [-0.5, 0.4, 0.05], max: [0.5, 1, 0.5] },
		};
	if (c.method === "env.reset")
		return [
			{ agentview: rgb(4, 4), wrist: rgb(4, 4), tcp_pos: f32([0, 0.6, 0.15]), gripper_width: 0.094, obs: f32([0]) },
			{ success: false },
		];
	if (c.method === "env.get_task_language") return "Move the gripper to the red target ball.";
	return undefined;
}

/** Metaworld's server: four env steps a tick, 40 mm a tick at most, a hand that cannot rotate. */
const METAWORLD: Omit<MmServer, "config"> = {
	stepsPerTick: 4,
	fixedHand: true,
	caps: { max_translation_m: 0.04, tilt_limits_deg: { roll_cw: 0, roll_ccw: 0, pitch_down: 0, pitch_up: 0 } },
};

async function session(
	t: { after: (fn: () => void) => void },
	values: Record<string, unknown>,
	o: Partial<MmServer> & { model?: { provider: string; id: string } } = {},
) {
	const config = { mm_home: String(values["mm-home"] ?? "refuse"), truth: null, ...o.config };
	const env = await fakeMmServer({ ...METAWORLD, ...o, config }, base);
	t.after(env.close);
	const s = await mmSession(t, metaworld, "metaworld", env, { task: "reach-v3", seed: "0", ...values }, o.model);
	return { ...s, env };
}

test("--preset motormind on Metaworld: pure mode; mm_act cuts by the 40 mm tick limit; steps and success come from the server", async (t) => {
	const s = await session(t, { preset: "motormind" }, { successAt: 160 });
	assert.equal(s.unavailable(), false, s.errors.join("\n"));
	assert.deepEqual(s.active(), ["mm_observe", "mm_act", "finish"]);
	assert.ok(!s.loaded.some((n) => n.startsWith("mm_")), "registered at the start that turns the mode on");
	const o = body(await s.call("mm_observe", {}));
	assert.deepEqual(o.pose.position_mm, [450, 0, 300]);
	assert.equal(o.caps.max_translation_m, 0.04);
	const r = body(await s.call("mm_act", { type: "move", direction: "forward", distance_mm: 100 }));
	assert.deepEqual(
		r.segments.map((x: any) => [x.label, x.outcome]),
		[1, 2, 3].map((i) => [`move forward 100.0 mm (base) [${i}/3]`, "done"]),
	);
	// Base +X from where the action started, a third of the way per segment.
	s.env.state.segments.forEach((x: any, k: number) => {
		assert.ok(Math.abs(x.goal_position_m[0] - (0.45 + (0.1 * (k + 1)) / 3)) < 1e-9, JSON.stringify(x));
	});
	// 100 mm at 50 mm/s is 2.5 mm a tick: 14 ticks per 33.3 mm segment, four env steps each; success
	// latched in the last one.
	assert.equal(s.env.state.steps, 3 * 14 * 4);
	const result = await s.finish();
	assert.equal(result.preset, "motormind");
	assert.deepEqual(result.motormind.variant, []);
	assert.deepEqual([result.env_steps, result.success], [168, true]);
	assert.equal(result.motormind.status, "env_success");
	assert.equal(result.motormind.segments, 3);
});

test("a rotation comes back refused and stops the action; nothing else moved", async (t) => {
	const s = await session(t, { preset: "motormind" });
	const r = body(await s.call("mm_act", { type: "rotate", direction: "yaw_left", angle_deg: 60 }));
	assert.deepEqual(
		r.segments.map((x: any) => x.outcome),
		["refused", "skipped", "skipped"],
	);
	assert.match(r.segments[0].message, /cannot rotate/);
	assert.equal(s.env.state.steps, 0);
	assert.equal((await s.finish()).env_steps, 0);
});

test("home is refused by default; --mm-home reset is the episode reset: its steps start over", async (t) => {
	const s = await session(t, { preset: "motormind" });
	const refused = await s.call("mm_act", { type: "home" });
	assert.equal(refused.isError, true);
	assert.match(refused.content[0].text, /this robot has no home/);
	const v = await session(t, { mm: "true", "mm-home": "reset", "memory-profile": "off" });
	assert.equal(v.unavailable(), false, v.errors.join("\n"));
	await v.call("mm_act", { type: "move", direction: "up", distance_mm: 20 });
	assert.equal(v.env.state.steps, 32);
	const home = body(await v.call("mm_act", { type: "home" }));
	assert.deepEqual([home.segments[0].outcome, home.episode_over], ["done", false]);
	const result = await v.finish();
	assert.deepEqual([result.success, result.env_steps], [false, 0]);
	assert.deepEqual(result.motormind.home, { mode: "reset", asked: 1, refused: 0, reset: 1 });
	assert.deepEqual(
		result.motormind.variant.map((x: any) => x.flag),
		["mm-home"],
	);
});

test("an attached server started otherwise than the flags fails the start; off, no mm tool is registered", async (t) => {
	const s = await session(t, { preset: "motormind" }, { config: { mm_home: "reset" } });
	assert.deepEqual(s.active(), []);
	assert.match(s.errors.join("\n"), /runs with mm_home="reset", but the flags ask for mm_home="refuse"/);
	const off = await session(t, {});
	assert.ok(!off.tools().some((n) => n.startsWith("mm_")), off.tools().join(", "));
	assert.ok(off.active().includes("move_delta"));
});
