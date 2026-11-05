import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Duplex } from "node:stream";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import robotwin from "../src/robots/robotwin/index.ts";
import { codeApiReply } from "./helpers/code-api.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A stub pi that records flags, tools and entries and runs handlers in registration order. */
function stubPi(values: Record<string, unknown> = {}) {
	values = deployFlags(values);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in values ? values[name] : o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: any) => tools.set(t.name, t),
		registerCommand: () => {},
		registerProvider: () => {},
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getEntries: () => [],
			getSessionDir: () => tmpdir(),
			getSessionFile: () => undefined,
			getSessionId: () => "sess",
		},
	};
	async function emit(name: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(name) ?? []) result = (await fn({ type: name, ...event }, ctx)) ?? result;
		return result;
	}
	const run = async (name: string, params: unknown) =>
		(await tools.get(name).execute("id", params, undefined, undefined, ctx)) as any;
	return { pi, flags, tools, entries, emit, run, active: () => active };
}

const nd = (dtype: string, shape: number[], data: Buffer) => ({ __ndarray__: data.toString("base64"), dtype, shape });
const f64 = (v: number[], shape = [v.length]) => nd("float64", shape, Buffer.from(Float64Array.from(v).buffer));
const info = (count: number, success: boolean) => ({
	robot_state: {
		left_eef_pose: f64([-0.2, 0, 0.9, 1, 0, 0, 0]),
		right_eef_pose: f64([0.2, 0, 0.9, 1, 0, 0, 0]),
		left_gripper: 1,
		right_gripper: 1,
		qpos_target14: f64(Array(14).fill(0)),
	},
	episode_status: { eval_success: success, take_action_cnt: count, step_lim: 10000, actual_seed: 100000 },
});

/** msgpack of LingBot's metadata (fixmap of fixstr keys; fixstr values and one positive fixint). */
function metadata(): Buffer {
	const str = (s: string) => Buffer.concat([Buffer.from([0xa0 | s.length]), Buffer.from(s)]);
	const fields: [string, string | number][] = [
		["runtime", "lingbotvla"],
		["policy_name", "robotwin_eef"],
		["state_layout", "eef16"],
		["action_layout", "eef16"],
		["use_length", 50],
	];
	return Buffer.concat([
		Buffer.from([0x80 | fields.length]),
		...fields.flatMap(([k, v]) => [str(k), typeof v === "number" ? Buffer.from([v]) : str(v)]),
	]);
}

/**
 * msgpack of LingBot's reply to an observation: `{ action: <rows x 16 float32 zeros> }` in openpi's
 * numpy extension (a map of bin keys: __ndarray__, data, dtype, shape).
 */
function actionReply(rows: number): Buffer {
	const bin = (b: Buffer) =>
		Buffer.concat([
			b.length < 0x100 ? Buffer.from([0xc4, b.length]) : Buffer.from([0xc5, b.length >> 8, b.length & 0xff]),
			b,
		]);
	const key = (k: string) => bin(Buffer.from(k));
	return Buffer.concat([
		Buffer.from([0x81, 0xa6]),
		Buffer.from("action"),
		Buffer.from([0x84]),
		key("__ndarray__"),
		Buffer.from([0xc3]),
		key("data"),
		bin(Buffer.alloc(rows * 16 * 4)),
		key("dtype"),
		Buffer.from([0xa3]),
		Buffer.from("<f4"),
		key("shape"),
		Buffer.from([0x92, rows, 16]),
	]);
}

/**
 * A fake RoboTwin env server (the wire protocol of ../src/rpc.ts) that also answers the LingBot
 * WebSocket handshake with the policy metadata; `code.run` reports a program that solved the task
 * in 12 native actions. As LingBot it answers every observation with a 50-action chunk;
 * `env.chunk_step` counts the native actions, and RoboTwin's eval_success holds from action `successAt` on.
 */
async function fakeEnv(o: { successAt?: number } = {}) {
	const calls: { method: string; args: unknown[]; kwargs: Record<string, unknown> }[] = [];
	let count = 0;
	const sockets: Duplex[] = [];
	const server = createServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			const { method, args = [], kwargs = {} } = JSON.parse(body);
			calls.push({ method, args, kwargs });
			let result: unknown = { status: "ok" };
			if (method === "code.api") result = codeApiReply("robotwin", kwargs.tier);
			else if (method === "env.get_env_meta")
				result = { task_name: "beat_block_hammer", task_config: "demo_randomized", seed: 100000 };
			else if (method === "env.reset") result = [{}, { ...info(0, false), instruction: "beat the block" }];
			else if (method === "env.get_task_language") result = "beat the block";
			else if (method === "env.chunk_step") {
				const n = (args[0] as { shape: number[] }).shape[0];
				count += n;
				const zeros = f64(Array(n).fill(0));
				result = [
					{ frames: [] },
					zeros,
					zeros,
					zeros,
					{ ...info(count, count >= (o.successAt ?? Infinity)), executed_actions: n },
				];
			} else if (method === "env.render_camera")
				result = [nd("uint8", [2, 2, 3], Buffer.alloc(12)), nd("float32", [2, 2], Buffer.alloc(16))];
			else if (method === "env.get_camera_meta")
				result = {
					intrinsic_K: f64([1, 0, 1, 0, 1, 1, 0, 0, 1], [3, 3]),
					cam2world_gl: f64([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [4, 4]),
					width: 2,
					height: 2,
				};
			else if (method === "env.move_to" || method === "env.set_gripper")
				result = {
					action_type: "qpos",
					requested_actions: 2,
					executed_actions: 2,
					completed: true,
					requested_steps: 2,
					executed_steps: 2,
					stop_reason: "completed",
					success: true,
					info: { ...info(2, method === "env.set_gripper"), executed_actions: 2 },
					frames: [nd("uint8", [2, 2, 3], Buffer.alloc(12)), nd("uint8", [2, 2, 3], Buffer.alloc(12))],
				};
			else if (method === "code.run")
				result = {
					status: "ran",
					stdout: "",
					stderr: "",
					traceback: null,
					error: null,
					result: null,
					calls: [],
					n_calls: 3,
					move_m: 0.2,
					ms: 5,
					steps: 12,
					info: info(12, true),
					success: true,
					budget_exhausted: false,
					frames: [nd("uint8", [2, 2, 3], Buffer.alloc(12))],
				};
			res.end(JSON.stringify({ ok: true, result }));
		});
	});
	server.on("upgrade", (req, socket) => {
		sockets.push(socket);
		const accept = createHash("sha1")
			.update(`${req.headers["sec-websocket-key"]}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
			.digest("base64");
		socket.write(
			`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
		);
		const payload = metadata();
		socket.write(Buffer.concat([Buffer.from([0x82, payload.length]), payload]));
		socket.on("error", () => {});
		// The client's frames (masked): answer each observation with a chunk, and the close frame at once.
		let pending = Buffer.alloc(0);
		socket.on("data", (d: Buffer) => {
			pending = Buffer.concat([pending, d]);
			while (pending.length >= 2) {
				const op = pending[0] & 0x0f;
				let len = pending[1] & 0x7f;
				let at = 2;
				if (len === 126) [len, at] = [pending.length >= 4 ? pending.readUInt16BE(2) : Infinity, 4];
				else if (len === 127)
					[len, at] = [pending.length >= 10 ? Number(pending.readBigUInt64BE(2)) : Infinity, 10];
				if (pending[1] & 0x80) at += 4;
				if (pending.length < at + len) return;
				pending = pending.subarray(at + len);
				if (op === 0x8) return void socket.end(Buffer.from([0x88, 0]));
				if (op === 0x2) {
					const reply = actionReply(50);
					socket.write(Buffer.concat([Buffer.from([0x82, 126, reply.length >> 8, reply.length & 0xff]), reply]));
				}
			}
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const port = (server.address() as AddressInfo).port;
	return {
		url: `http://127.0.0.1:${port}`,
		ws: `ws://127.0.0.1:${port}`,
		calls,
		close: () => {
			for (const s of sockets) s.destroy();
			server.closeAllConnections();
			server.close();
		},
	};
}

test("--code=true: run_code runs on the env server and its result becomes a recorded state and the success", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	// A local memory corpus: the published one would be fetched from Hugging Face.
	const base = mkdtempSync(join(tmpdir(), "robotwin-code-"));
	t.after(() => rmSync(base, { recursive: true, force: true }));
	const memory = join(base, "robotwin");
	mkdirSync(memory);
	writeFileSync(join(memory, "MEMORY.md"), "# RoboTwin\n");
	const s = stubPi({
		"env-url": env.url,
		lingbot: env.ws,
		code: "true",
		"code-api": "low",
		"memory-profile": "local",
		"memory-dir": memory,
	});
	robotwin(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), ["run_code", "read", "ls", "grep", "find", "write", "finish"]);
	assert.deepEqual(
		env.calls.filter((c) => c.method === "code.api").map((c) => c.kwargs.tier),
		[undefined, "low"],
		"the episode's registry, then code mode's tier",
	);
	await s.emit("agent_start");
	const r = await s.run("run_code", { code: "step(list(get_state()['qpos_target']))" });
	const run = env.calls.find((c) => c.method === "code.run")!;
	assert.equal(run.kwargs.tier, "low");
	assert.equal(run.kwargs.code, "step(list(get_state()['qpos_target']))");
	assert.equal(r.details.status, "ran");
	assert.equal(r.details.eval_success, true, "the success of the run's episode status");
	assert.equal(r.details.step, 1, "a new recorded state after the reset's");
	assert.deepEqual(
		r.content.map((c: { type: string }) => c.type),
		["text", "text", "image", "image", "image"],
	);
	// As the motion tools: nothing runs once the episode is terminal.
	const again = await s.run("run_code", { code: "step(list(get_state()['qpos_target']))" });
	assert.match(again.content[0].text, /^Episode is terminal \(eval_success=true/);
	assert.equal(env.calls.filter((c) => c.method === "code.run").length, 1);
	await s.run("finish", { status: "success", summary: "hammered" });
	await s.emit("agent_end", { messages: [] });
	const result = s.entries.find((e) => e.type === "robot_result")?.data;
	assert.equal(result.success, true);
	assert.equal(result.take_action_cnt, 12);
	assert.equal(result.native_actions, 12);
	assert.equal(result.code, "true");
	assert.equal(result.code_api, "low");
	await s.emit("session_shutdown");
});

test("the motion tools are the env server's methods: manifest schemas, params passed through, info absorbed", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const base = mkdtempSync(join(tmpdir(), "robotwin-motion-"));
	t.after(() => rmSync(base, { recursive: true, force: true }));
	const memory = join(base, "robotwin");
	mkdirSync(memory);
	writeFileSync(join(memory, "MEMORY.md"), "# RoboTwin\n");
	const s = stubPi({ "env-url": env.url, lingbot: env.ws, "memory-profile": "local", "memory-dir": memory });
	robotwin(s.pi);
	const props = (name: string) => s.tools.get(name).parameters.properties;
	assert.deepEqual(props("move_to").arm.enum, ["left", "right"]);
	assert.equal(props("move_to").quat.minItems, 4);
	assert.equal(props("set_gripper").val.maximum, 1);
	assert.deepEqual(s.tools.get("move_to").parameters.required, ["arm", "xyz"]);
	await s.emit("session_start");
	process.exitCode = undefined;
	for (const name of ["move_to", "rotate_wrist", "set_gripper", "release", "lingbot_act"])
		assert.ok(s.active().includes(name), name);
	await s.emit("agent_start");
	const r = await s.run("move_to", { arm: "left", xyz: [-0.2, 0, 0.95], substeps: 5 });
	const call = env.calls.find((c) => c.method === "env.move_to")!;
	assert.deepEqual(call.kwargs, { arm: "left", xyz: [-0.2, 0, 0.95], substeps: 5 }, "no plan or step from pi");
	assert.ok(!env.calls.some((c) => c.method === "env.step" || c.method === "env.plan_arm_path"));
	const view = JSON.parse(r.content[0].text);
	assert.equal(view.result.stop_reason, "completed");
	assert.equal(view.result.info, undefined, "the info becomes the state, not the result");
	assert.equal(view.result.frames, undefined);
	assert.equal(view.episode_status.native_actions, 2);
	// A motion that solved the task: the next one is refused by pi without a call.
	await s.run("set_gripper", { arm: "right", val: 0 });
	const before = env.calls.length;
	const again = await s.run("move_to", { arm: "left", xyz: [-0.2, 0, 0.9] });
	assert.match(again.content[0].text, /^Episode is terminal \(eval_success=true/);
	assert.equal(env.calls.length, before);
	await s.emit("session_shutdown");
});

test("--vla-alone lingbot: the first prompt runs LingBot until eval_success or the step budget, asks no model, and records planner vla", async (t) => {
	const base = mkdtempSync(join(tmpdir(), "robotwin-vla-alone-"));
	t.after(() => rmSync(base, { recursive: true, force: true }));
	const memory = join(base, "robotwin");
	mkdirSync(memory);
	writeFileSync(join(memory, "MEMORY.md"), "# RoboTwin\n");
	const run = async (fake: Parameters<typeof fakeEnv>[0], values: Record<string, unknown> = {}) => {
		const env = await fakeEnv(fake);
		t.after(env.close);
		const s = stubPi({
			"env-url": env.url,
			lingbot: env.ws,
			"memory-profile": "local",
			"memory-dir": memory,
			"vla-alone": "lingbot",
			...values,
		});
		robotwin(s.pi);
		await s.emit("session_start");
		process.exitCode = undefined;
		const log = console.error;
		console.error = () => {};
		let handled: unknown;
		try {
			handled = await s.emit("input", { text: "Solve the task.", source: "interactive" });
			// A second prompt runs nothing more.
			await s.emit("input", { text: "again", source: "interactive" });
			await s.emit("session_shutdown");
		} finally {
			console.error = log;
		}
		const result = s.entries.find((e) => e.type === "robot_result")?.data;
		const chunks = env.calls.filter((c) => c.method === "env.chunk_step").map((c) => (c.args[0] as any).shape[0]);
		return { handled, result, chunks };
	};
	const solved = await run({ successAt: 120 });
	assert.deepEqual(solved.handled, { action: "handled" });
	assert.deepEqual(solved.chunks, [50, 50, 50], "stops at eval_success");
	assert.equal(solved.result.planner, "vla");
	assert.equal(solved.result.success, true);
	assert.equal(solved.result.turns, 0);
	assert.equal(solved.result.env_error, false);
	assert.equal(solved.result.native_actions, 150);
	const { vla_seeds, ...rest } = solved.result.vla_alone;
	assert.deepEqual(rest, {
		skill: "lingbot",
		tool: "lingbot_act",
		instruction: "beat the block",
		// What is left of the env's step limit (its --max-episode-steps).
		steps_budget: 10000,
		env_steps: 150,
		chunks: 3,
		success: true,
	});
	assert.equal(vla_seeds.length, 3);
	// --vla-alone-steps bounds a run that never succeeds; the last chunk is cut to what is left of it.
	const spent = await run({}, { "vla-alone-steps": "120" });
	assert.deepEqual(spent.chunks, [50, 50, 20]);
	assert.equal(spent.result.success, false);
	assert.equal(spent.result.vla_alone.env_steps, 120);
});

test("LingBot is optional: without it lingbot_act stays inactive and the result notes it; --require-skills lingbot refuses", async (t) => {
	const env = await fakeEnv();
	t.after(env.close);
	const base = mkdtempSync(join(tmpdir(), "robotwin-skill-"));
	t.after(() => rmSync(base, { recursive: true, force: true }));
	const memory = join(base, "robotwin");
	mkdirSync(memory);
	writeFileSync(join(memory, "MEMORY.md"), "# RoboTwin\n");
	const values = { "env-url": env.url, "memory-profile": "local", "memory-dir": memory };
	const s = stubPi({ ...values, lingbot: "off" });
	robotwin(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.ok(s.active().includes("move_to"), "the robot started");
	assert.ok(!s.active().includes("lingbot_act"));
	await s.emit("agent_start");
	await s.run("finish", { status: "failure", summary: "x" });
	await s.emit("agent_end", { messages: [] });
	assert.deepEqual(s.entries.find((e) => e.type === "robot_result")?.data.skills_off, { lingbot: "switched off" });
	await s.emit("session_shutdown");

	const r = stubPi({ ...values, lingbot: "ws://127.0.0.1:9", "require-skills": "lingbot" });
	robotwin(r.pi);
	await r.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(r.active(), [], "a run that needs LingBot does not start without it");
	await r.emit("session_shutdown");
});
