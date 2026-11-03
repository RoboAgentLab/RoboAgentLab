/**
 * MotorMind mode on robosuite (src/robots/robosuite, `motormind` in its spec): the mode's tools on
 * a fake env server speaking services utils/motormind.py's wire format. --preset motormind mounts
 * pure mode on the one-arm tasks; mm_act cuts an action by OSC_POSE's 50 mm limit, one env step a
 * tick; the episode's env steps, success and success step come from every reply; the two-arm tasks
 * refuse the mode; a server started otherwise than the flags fails the start.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import robosuite from "../src/robots/robosuite/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { body, fakeMmServer, type MmServer, mmSession } from "./helpers/motormind-robot.ts";
import { type Call, f32, rgb } from "./sim-stub.ts";

/** Robosuite's own start calls for `task`: the meta, the reset (robot0's state), the language. */
const base =
	(task: string) =>
	(c: Call): unknown => {
		const two = task.startsWith("TwoArm");
		if (c.method === "code.api") return codeApiReply("robosuite", c.kwargs.tier);
		if (c.method === "env.get_env_meta")
			return {
				task,
				seed: 0,
				arms: two ? ["robot0", "robot1"] : ["robot0"],
				gripper: task !== "Wipe",
				language: "lift the cube",
				box: [],
				z_floor: 0.8,
				table_z: 0.8,
				max_move_m: 0.3,
			};
		if (c.method === "env.reset")
			return [
				{
					agentview: rgb(),
					wrist: rgb(),
					robot0_eef_pos: f32([0, 0, 1]),
					robot0_eef_quat: f32([1, 0, 0, 0]),
					robot0_gripper_width: 0.08,
					robot0_gripper_command: "open",
					success: false,
					success_step: null,
					env_steps: 0,
				},
				{ language: "lift the cube" },
			];
		if (c.method === "env.get_task_language") return "lift the cube";
		return undefined;
	};

/** The server's robot0: one env step a tick, `success_step` in every observation. */
const ROBOSUITE: Omit<MmServer, "config"> = { extra: (s) => ({ success_step: s.successStep }) };

async function session(
	t: { after: (fn: () => void) => void },
	values: Record<string, unknown>,
	o: Partial<MmServer> & { task?: string } = {},
) {
	const task = o.task ?? "Lift";
	const config = { mm_home: String(values["mm-home"] ?? "refuse"), truth: null, ...o.config };
	const env = await fakeMmServer({ ...ROBOSUITE, ...o, config }, base(task));
	t.after(env.close);
	const s = await mmSession(t, robosuite, "robosuite", env, { task, seed: "0", ...values });
	return { ...s, env };
}

test("--preset motormind on Lift: pure mode; 50 mm segments of one env step a tick; success and its step from the server", async (t) => {
	const s = await session(t, { preset: "motormind" }, { successAt: 30 });
	assert.equal(s.unavailable(), false, s.errors.join("\n"));
	assert.deepEqual(s.active(), ["mm_observe", "mm_act", "finish"]);
	const r = body(await s.call("mm_act", { type: "move", direction: "up", distance_mm: 100 }));
	assert.deepEqual(
		r.segments.map((x: any) => [x.label, x.outcome]),
		[
			["move up 100.0 mm (base) [1/2]", "done"],
			["move up 100.0 mm (base) [2/2]", "done"],
		],
	);
	// 50 mm at 50 mm/s, 2.5 mm a tick: 20 ticks a segment, one env step each; success at step 30.
	assert.equal(s.env.state.steps, 40);
	const result = await s.finish();
	assert.deepEqual([result.success, result.success_step, result.env_steps], [true, 30, 40]);
	assert.equal(result.motormind.status, "env_success");
	assert.deepEqual(result.motormind.variant, []);
});

test("the two-arm tasks refuse MotorMind mode at the start; Lift without it registers no mm tool", async (t) => {
	const s = await session(t, { preset: "motormind" }, { task: "TwoArmLift" });
	assert.deepEqual(s.active(), []);
	assert.match(s.errors.join("\n"), /MotorMind mode \(--mm\) drives one arm; TwoArmLift has two/);
	const off = await session(t, {});
	assert.ok(!off.tools().some((n) => n.startsWith("mm_")), off.tools().join(", "));
	assert.ok(off.active().includes("move_to"));
});

test("home is the episode reset with --mm-home reset; a server started otherwise fails the start", async (t) => {
	const v = await session(t, { mm: "true", "mm-home": "reset", "memory-profile": "off" });
	assert.equal(v.unavailable(), false, v.errors.join("\n"));
	await v.call("mm_act", { type: "move", direction: "left", distance_mm: 20 });
	assert.equal(v.env.state.steps, 8);
	const home = body(await v.call("mm_act", { type: "home" }));
	assert.equal(home.segments[0].outcome, "done");
	const result = await v.finish();
	assert.deepEqual([result.env_steps, result.success], [0, false]);
	assert.deepEqual(result.motormind.home, { mode: "reset", asked: 1, refused: 0, reset: 1 });
	const off = await session(t, { preset: "motormind" }, { config: { mm_home: "reset" } });
	assert.deepEqual(off.active(), []);
	assert.match(off.errors.join("\n"), /runs with mm_home="reset", but the flags ask for mm_home="refuse"/);
});
