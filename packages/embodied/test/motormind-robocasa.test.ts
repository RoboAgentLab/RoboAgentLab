/**
 * MotorMind mode on RoboCasa (src/robots/robocasa, `motormind` in its spec): the mode's tools on a
 * fake env server speaking services utils/motormind.py's wire format. --preset motormind mounts
 * pure mode (its --memory-profile off is accepted in MotorMind mode only: the robot's own prompt
 * reads memory); mm_act cuts an action by OSC_POSE's 50 mm limit, one env step a tick; the episode's
 * env steps and success come from every reply; home is the episode reset with --mm-home reset.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import robocasa from "../src/robots/robocasa/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { body, fakeMmServer, type MmServer, mmSession } from "./helpers/motormind-robot.ts";
import { type Call, f32, nd } from "./sim-stub.ts";

const SERVICES = new URL("../../../services", import.meta.url).pathname;

/** The robot's own observation (robot0_* only), as the env server's reset returns it. */
const raw = () => ({
	robot0_eef_pos: f32([0.5, 0, 1]),
	robot0_eef_quat: f32([0, 0, 0, 1]),
	robot0_gripper_qpos: f32([0.04, -0.04]),
	robot0_base_pos: f32([0, 0, 0]),
	robot0_base_quat: f32([0, 0, 0, 1]),
	robot0_base_to_eef_pos: f32([0.5, 0, 1]),
	robot0_base_to_eef_quat: f32([0, 0, 0, 1]),
});

/** RoboCasa's own start calls: meta, resets, language, success text, renders, the recorder switch. */
function base(c: Call): unknown {
	const { method, kwargs } = c;
	const size = Number(kwargs.height ?? 256);
	if (method === "code.api") return codeApiReply("robocasa", kwargs.tier);
	if (method === "env.get_env_meta")
		return { task_name: "OpenDrawer", split: "target", seed: 0, scene: null, env_id: null };
	if (method === "env.reset") return raw();
	if (method === "env.get_task_language") return "open the drawer";
	if (method === "env.get_success_criteria_text") return "drawer open";
	if (method === "env.check_success") return false;
	if (method === "env.get_task_progress") return {};
	if (method === "env.render_camera") {
		const img = nd("uint8", [size, size, 3], Buffer.alloc(size * size * 3));
		return kwargs.depth ? [img, nd("float32", [size, size], Buffer.alloc(size * size * 4))] : img;
	}
	if (method === "env.get_camera_transform")
		return nd(
			"float64",
			[4, 4],
			Buffer.from(Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).buffer),
		);
	return undefined;
}

async function session(
	t: { after: (fn: () => void) => void },
	values: Record<string, unknown>,
	o: Partial<MmServer> = {},
) {
	const config = { mm_home: String(values["mm-home"] ?? "refuse"), truth: null, ...o.config };
	const env = await fakeMmServer({ ...o, config }, base);
	t.after(env.close);
	const s = await mmSession(t, robocasa, "robocasa", env, {
		task: "OpenDrawer",
		split: "target",
		seed: "0",
		rldx: "off",
		services: SERVICES,
		...values,
	});
	process.exitCode = undefined;
	return { ...s, env };
}

test("--preset motormind on RoboCasa: pure mode with memory off; 50 mm segments; steps and success from the server", async (t) => {
	const s = await session(t, { preset: "motormind" }, { successAt: 30 });
	assert.equal(s.unavailable(), false, s.errors.join("\n"));
	assert.deepEqual(s.active(), ["mm_observe", "mm_act", "finish"]);
	const r = body(await s.call("mm_act", { type: "move", direction: "forward", distance_mm: 100 }));
	assert.deepEqual(
		r.segments.map((x: any) => x.outcome),
		["done", "done"],
	);
	assert.equal(s.env.state.steps, 40);
	const result = await s.finish();
	assert.deepEqual([result.success, result.env_steps], [true, 40]);
	assert.equal(result.motormind.status, "env_success");
	assert.deepEqual(result.motormind.variant, []);
});

test("--memory-profile off is MotorMind mode's only: the robot's own prompt reads memory", async (t) => {
	const s = await session(t, { "memory-profile": "off" });
	assert.deepEqual(s.active(), []);
	assert.match(s.errors.join("\n"), /--memory-profile off: robocasa's prompt reads memory/);
});

test("home is the episode reset with --mm-home reset: its steps and success start over", async (t) => {
	const v = await session(t, { mm: "true", "mm-home": "reset", "memory-profile": "off" });
	assert.equal(v.unavailable(), false, v.errors.join("\n"));
	await v.call("mm_act", { type: "move", direction: "up", distance_mm: 20 });
	assert.equal(v.env.state.steps, 8);
	assert.equal(body(await v.call("mm_act", { type: "home" })).segments[0].outcome, "done");
	const result = await v.finish();
	assert.deepEqual([result.env_steps, result.success], [0, false]);
	assert.deepEqual(result.motormind.home, { mode: "reset", asked: 1, refused: 0, reset: 1 });
});
