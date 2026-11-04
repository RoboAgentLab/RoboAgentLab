/**
 * MotorMind mode on ManiSkill (src/robots/maniskill, `motormind` in its spec): the mode's tools on a
 * fake env server speaking services utils/motormind.py's wire format. --preset motormind mounts pure
 * mode on the Panda and the xArm6; mm_act cuts an action by the 50 mm per-tick limit, one control step
 * a tick; the episode's env steps and success come from every reply and the segments' frames go to the
 * video; the other arms and the RLinf rigs refuse the mode before the reset; a server started
 * otherwise than the flags fails the start.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import maniskill, { ROBOTS, type RobotId, VIEW_SETUP } from "../src/robots/maniskill/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { body, fakeMmServer, type MmServer, mmSession } from "./helpers/motormind-robot.ts";
import { type Call, f32, rgb } from "./sim-stub.ts";

/** ManiSkill's own start calls for `robot` on `envId`: the meta, the reset (the arm's state), the language. */
const base =
	(robot: RobotId, envId: string) =>
	(c: Call): unknown => {
		if (c.method === "code.api") return codeApiReply("maniskill", c.kwargs.tier);
		if (c.method === "env.get_env_meta")
			return {
				env_id: envId,
				seed: 0,
				scene: envId.startsWith("Block") ? { table_tex: "006" } : null,
				table_z: 0,
				view_size: 256,
				episode_steps: 50,
				robot,
				...(envId.startsWith("Block")
					? { agentview: "external_cam", wrist_mount: "centered", wrist_rotation: 0, wrist_flip: "both" }
					: robot === "panda"
						? VIEW_SETUP
						: ROBOTS[robot].setup),
			};
		if (c.method === "env.reset")
			return [
				{
					agentview: rgb(),
					wrist: rgb(),
					tcp_pos: f32([0, 0, 0.18]),
					tcp_quat_wxyz: f32([0, 1, 0, 0]),
					gripper_width: 0.08,
					qpos: f32([0, 0]),
				},
				{ success: false },
			];
		if (c.method === "env.get_task_language") return "pick up the red cube";
		return undefined;
	};

async function session(
	t: { after: (fn: () => void) => void },
	values: Record<string, unknown>,
	o: Partial<MmServer> & { robot?: RobotId; envId?: string } = {},
) {
	const robot = o.robot ?? "panda";
	const envId = o.envId ?? "PickCube-v1";
	const config = { mm_home: String(values["mm-home"] ?? "refuse"), truth: null, ...o.config };
	const env = await fakeMmServer({ ...o, config }, base(robot, envId));
	t.after(env.close);
	const s = await mmSession(t, maniskill, "maniskill", env, { task: envId, seed: "0", arm: robot, ...values });
	return { ...s, env };
}

test("--preset motormind on PickCube: pure mode on the Panda and the xArm6; 50 mm segments of one control step a tick; success from the server", async (t) => {
	for (const robot of ["panda", "xarm6_robotiq"] as const) {
		const s = await session(t, { preset: "motormind" }, { robot, successAt: 30 });
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
		// 50 mm at 50 mm/s, 2.5 mm a tick: 20 ticks a segment, one control step each; success at step 30.
		assert.equal(s.env.state.steps, 40);
		const result = await s.finish();
		assert.deepEqual([result.success, result.env_steps], [true, 40]);
		assert.equal(result.motormind.status, "env_success");
		assert.deepEqual(result.motormind.variant, []);
		assert.equal(result.maniskill_robot, robot === "panda" ? undefined : robot);
		const runs = s.env.calls.filter((c) => c.method === "env.mm_run_step");
		assert.equal(runs.length, 2);
		assert.equal(runs[0].kwargs.seg.kind, "translate");
	}
});

test("a turn runs as rotate segments; home is the episode reset with --mm-home reset", async (t) => {
	const s = await session(t, { mm: "true", "mm-home": "reset", "memory-profile": "off" });
	assert.equal(s.unavailable(), false, s.errors.join("\n"));
	// 45 deg is cut by the 0.5 rad per-segment limit.
	const turn = body(await s.call("mm_act", { type: "rotate", direction: "yaw_left", angle_deg: 45 }));
	assert.deepEqual(
		turn.segments.map((x: any) => [x.kind, x.outcome]),
		[
			["rotate", "done"],
			["rotate", "done"],
		],
	);
	await s.call("mm_act", { type: "move", direction: "left", distance_mm: 20 });
	assert.ok(s.env.state.steps > 0);
	const home = body(await s.call("mm_act", { type: "home" }));
	assert.equal(home.segments[0].outcome, "done");
	const result = await s.finish();
	assert.deepEqual([result.env_steps, result.success], [0, false]);
	assert.deepEqual(result.motormind.home, { mode: "reset", asked: 1, refused: 0, reset: 1 });
});

test("the other arms and the RLinf rigs refuse MotorMind mode before the reset; without it no mm tool", async (t) => {
	for (const [robot, envId, why] of [
		["widowxai", "PickCube-v1", /widowxai has no wrist camera/],
		["panda_stick", "PushT-v1", /panda_stick holds a stick/],
		["panda_pair", "TwoRobotPickCube-v1", /panda_pair has two arms/],
		["widowx250s", "PutCarrotOnPlateInScene-v1", /widowx250s runs its scene's 5 Hz controller/],
		["panda", "BlockPAP-v1", /BlockPAP-v1 is an RLinf rig/],
	] as const) {
		const s = await session(t, { preset: "motormind" }, { robot, envId });
		assert.deepEqual(s.active(), [], robot);
		assert.match(s.errors.join("\n"), why);
		assert.ok(!s.env.calls.some((c) => c.method === "env.reset"), `${robot}: refused before the reset`);
	}
	const off = await session(t, {});
	assert.ok(!off.tools().some((n) => n.startsWith("mm_")), off.tools().join(", "));
	assert.ok(off.active().includes("move_delta"));
});

test("a server started with another --mm-home fails the start", async (t) => {
	const s = await session(t, { preset: "motormind" }, { config: { mm_home: "reset" } });
	assert.deepEqual(s.active(), []);
	assert.match(s.errors.join("\n"), /runs with mm_home="reset", but the flags ask for mm_home="refuse"/);
});
