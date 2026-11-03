/**
 * MotorMind mode on RoboTwin (src/robots/robotwin, `motormind` in its spec): the mode's tools on a
 * fake env server speaking services utils/motormind.py's wire format, every reply carrying the
 * robot state and episode status RoboTwin's tools keep. --preset motormind mounts pure mode (its
 * --memory-profile off is accepted in MotorMind mode only); the result's success, action count and
 * native actions come from the replies; --robotwin-mm-arm names one arm.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import robotwin from "../src/robots/robotwin/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { body, fakeMmServer, type MmServer, mmSession } from "./helpers/motormind-robot.ts";
import { type Call, nd } from "./sim-stub.ts";

const f64 = (v: number[], shape = [v.length]) => nd("float64", shape, Buffer.from(Float64Array.from(v).buffer));
const robotState = () => ({
	left_eef_pose: f64([-0.2, 0, 0.9, 1, 0, 0, 0]),
	right_eef_pose: f64([0.2, 0, 0.9, 1, 0, 0, 0]),
	left_gripper: 1,
	right_gripper: 1,
	qpos_target14: f64(Array(14).fill(0)),
});
const status = (count: number, success: boolean) => ({
	eval_success: success,
	take_action_cnt: count,
	step_lim: 10000,
	actual_seed: 100000,
});

/** RoboTwin's own start calls: meta, the reset with its info, renders and calibration. */
function base(c: Call): unknown {
	const { method, kwargs } = c;
	if (method === "code.api") return codeApiReply("robotwin", kwargs.tier);
	if (method === "env.get_env_meta")
		return { task_name: "beat_block_hammer", task_config: "demo_randomized", seed: 100000 };
	if (method === "env.reset")
		return [{}, { robot_state: robotState(), episode_status: status(0, false), instruction: "beat the block" }];
	if (method === "env.render_camera")
		return [nd("uint8", [2, 2, 3], Buffer.alloc(12)), nd("float32", [2, 2], Buffer.alloc(16))];
	if (method === "env.get_camera_meta")
		return {
			intrinsic_K: f64([1, 0, 1, 0, 1, 1, 0, 0, 1], [3, 3]),
			cam2world_gl: f64([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [4, 4]),
			width: 2,
			height: 2,
		};
	return undefined;
}

/** The server's info in every observation (its `_mm_extra`); one native action a tick. */
const ROBOTWIN: Omit<MmServer, "config"> = {
	views: ["scene", "wrist"],
	extra: (s) => ({ robot_state: robotState(), episode_status: status(s.steps, s.success) }),
};

async function session(
	t: { after: (fn: () => void) => void },
	values: Record<string, unknown>,
	o: Partial<MmServer> = {},
) {
	const config = { mm_home: String(values["mm-home"] ?? "refuse"), truth: null, ...o.config };
	const env = await fakeMmServer({ ...ROBOTWIN, ...o, config }, base);
	t.after(env.close);
	const s = await mmSession(t, robotwin, "robotwin", env, {
		task: "beat_block_hammer",
		"task-config": "demo_randomized",
		seed: "100000",
		lingbot: "off",
		...values,
	});
	process.exitCode = undefined;
	return { ...s, env };
}

test("--preset motormind on RoboTwin: pure mode with memory off; success, action count and native actions from the replies", async (t) => {
	const s = await session(t, { preset: "motormind" }, { successAt: 30 });
	assert.equal(s.unavailable(), false, s.errors.join("\n"));
	assert.deepEqual(s.active(), ["mm_observe", "mm_act", "finish"]);
	const r = body(await s.call("mm_act", { type: "move", direction: "forward", distance_mm: 100 }));
	assert.deepEqual(
		r.segments.map((x: any) => x.outcome),
		["done", "done"],
	);
	const result = await s.finish();
	assert.deepEqual([result.success, result.take_action_cnt, result.native_actions], [true, 40, 40]);
	assert.equal(result.motormind.status, "env_success");
	assert.deepEqual(result.motormind.variant, []);
});

test("--memory-profile off is MotorMind mode's only; --robotwin-mm-arm is left or right", async (t) => {
	const off = await session(t, { "memory-profile": "off" });
	assert.deepEqual(off.active(), []);
	assert.match(off.errors.join("\n"), /--memory-profile off: robotwin's prompt reads memory/);
	const both = await session(t, { preset: "motormind", "robotwin-mm-arm": "both" });
	assert.deepEqual(both.active(), []);
	assert.match(both.errors.join("\n"), /--robotwin-mm-arm must be left or right, not both/);
});
