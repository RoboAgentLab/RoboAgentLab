import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { generateKind, sessionPrograms, sessionUnits } from "../src/capabilities/flash/generate.ts";
import { type FlashHook, type FlashProgram, flash } from "../src/capabilities/flash/index.ts";
import { type KindProgram, kindFiles, readKind, replayUnits } from "../src/capabilities/flash/kinds.ts";
import { encodePng } from "../src/infra/png.ts";
import finetuned, { STEP_ENTRY } from "../src/modes/finetuned/index.ts";
import { codeSamples, convert, parseVars, toolSamples } from "../src/modes/finetuned/sessions_to_sft.ts";
import {
	FINISH_TOOL,
	NOOP_PROGRAM,
	parseProgram,
	parseToolCall,
	renderTools,
	SURFACE_MAX_TOKENS,
	type ToolSpec,
} from "../src/modes/finetuned/surfaces.ts";
import { loadManifest, toolEntry, toolSchema } from "../src/primitives/manifest.ts";
import { PRIMITIVES_EVENT, type PrimitivesHandle } from "../src/robot.ts";
import { deployFlags } from "./helpers/deployment.ts";

// The toy robot's primitive manifest (./fixtures/manifests/toy.json): move_to (high, tool and code), segment (ts).
process.env.PI_EMBODIED_MANIFESTS = new URL("./fixtures/manifests/", import.meta.url).pathname;
const TOY = loadManifest("toy");

type Handler = (event: any, ctx: any) => unknown;
const png = (w: number, h: number) => encodePng(Buffer.alloc(w * h * 3, 7), w, h);
const AGENT = png(64, 64).toString("base64");
const WRIST = png(32, 32).toString("base64");

/** The tools the toy robot publishes: a read, the manifest's move_to, a module tool (act) with its own schema. */
const OBSERVE: ToolSpec = {
	name: "view_env_state",
	description: "Current state and images.",
	parameters: Type.Object({}),
};
const MOVE_TO: ToolSpec = {
	name: "move_to",
	description: "Servo to xyz.",
	parameters: toolSchema(toolEntry(TOY, "move_to")),
};
const TOOLS = [OBSERVE, MOVE_TO];

// ---------------------------------------------------------------------------
// surfaces.ts: rendering and the strict parsers

test("renderTools lists each tool with its parameters, types, requiredness and descriptions", () => {
	assert.equal(
		renderTools(TOOLS),
		[
			"- view_env_state: Current state and images.",
			"- move_to: Servo to xyz.",
			"    xyz: [number, number, number], required: target [x, y, z] in metres",
			"    gripper: number: -1 opens, +1 closes",
		].join("\n"),
	);
	const enumTool: ToolSpec = {
		name: "grip",
		description: "",
		parameters: Type.Object({
			how: Type.Unsafe<string>({ type: "string", enum: ["open", "close"] }),
			n: Type.Optional(Type.Integer({ minimum: 1, maximum: 5 })),
		}),
	};
	assert.equal(renderTools([enumTool]), '- grip\n    how: "open" | "close", required\n    n: integer (>= 1, <= 5)');
});

test("parseToolCall accepts one checked call in the offered tools and rejects everything else", () => {
	const ok = (raw: string) => parseToolCall(raw, [...TOOLS, FINISH_TOOL]);
	assert.deepEqual(ok('{"tool": "move_to", "arguments": {"xyz": [0.1, 0.2, 0.3]}}'), {
		name: "move_to",
		arguments: { xyz: [0.1, 0.2, 0.3] },
	});
	// A fenced reply, a reasoning block, `name`/`args` spellings and prose around the object all parse.
	assert.deepEqual(ok('```json\n{"name": "view_env_state", "args": {}}\n```'), {
		name: "view_env_state",
		arguments: {},
	});
	assert.equal(ok('<think>look first</think>I will call {"tool": "view_env_state"} now.').name, "view_env_state");
	assert.deepEqual(ok('{"tool":"finish","arguments":{"status":"success","summary":"done"}}').arguments, {
		status: "success",
		summary: "done",
	});
	// Rejected: no JSON, no tool, a tool that is not offered, arguments that fail the schema, non-object arguments.
	for (const [raw, why] of [
		["MV_DOWN", /no JSON object/],
		['{"arguments": {}}', /names no tool/],
		['{"tool": "pi0_pick", "arguments": {"prompt": "the cube"}}', /not one of the offered tools/],
		['{"tool": "move_to", "arguments": {"xyz": [0.1, 0.2]}}', /move_to/],
		['{"tool": "move_to", "arguments": {}}', /move_to/],
		['{"tool": "move_to", "arguments": [1, 2, 3]}', /must be an object/],
	] as const)
		assert.throws(() => ok(raw), why, raw);
});

test("parseProgram takes exactly one fenced block, or the word DONE, and nothing else", () => {
	assert.deepEqual(parseProgram("```python\nr = move_to([0.1, 0.2, 0.3])\nRESULT = r\n```"), {
		kind: "program",
		code: "r = move_to([0.1, 0.2, 0.3])\nRESULT = r\n",
	});
	assert.deepEqual(parseProgram("<think>plan</think>\n```\nprint(1)\n```\n"), { kind: "program", code: "print(1)\n" });
	assert.deepEqual(parseProgram(" DONE. "), { kind: "done" });
	for (const [raw, why] of [
		["move_to([0, 0, 0])", /exactly one fenced code block, found 0/],
		["```python\na\n```\nthen\n```python\nb\n```", /found 2/],
		["```python\n\n```", /empty/],
		["Here is the program:\n```python\nprint(1)\n```", /text outside/],
		['{"tool": "move_to"}', /found 0/],
	] as const)
		assert.throws(() => parseProgram(raw), why, raw);
});

// ---------------------------------------------------------------------------
// the provider on the tools and code surfaces, against a fake OpenAI-compatible endpoint

async function endpoint(replies: string[]) {
	const bodies: any[] = [];
	const server = createServer(async (req: IncomingMessage, res) => {
		let data = "";
		for await (const chunk of req) data += chunk;
		bodies.push(JSON.parse(data));
		const r = replies.shift() ?? "DONE";
		res.writeHead(200, { "content-type": "application/json" });
		res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: r } }] }));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
	return { url, bodies, close: () => new Promise<void>((resolve) => server.close(() => resolve())) };
}

/** A stub pi with the finetuned provider, the toy robot's task entry and its published tools. */
function fakePi(flags: Record<string, string>, tools: string[], handle: Partial<PrimitivesHandle> | undefined = {}) {
	flags = deployFlags(flags);
	const handlers = new Map<string, Handler[]>();
	const defaults: Record<string, unknown> = {};
	const entries: { type: string; data: any }[] = [];
	const listeners = new Map<string, ((data: unknown) => void)[]>();
	let provider: any;
	const pi = {
		events: {
			on: (channel: string, fn: (data: unknown) => void) => {
				listeners.set(channel, [...(listeners.get(channel) ?? []), fn]);
				return () => {};
			},
		},
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			defaults[name] = o.default;
		},
		getFlag: (name: string) => (name in flags ? flags[name] : defaults[name]),
		getActiveTools: () => tools,
		registerProvider: (p: any) => {
			provider = p;
		},
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
	} as unknown as ExtensionAPI;
	const warnings: string[] = [];
	const ctx = {
		hasUI: true,
		model: { provider: "finetuned" },
		ui: { notify: (s: string) => warnings.push(s) },
		sessionManager: {
			getBranch: () => [
				{ type: "custom", customType: "robot_task", data: { robot: "toy", task: "PickCube-v1", seed: "3" } },
			],
		},
	};
	finetuned(pi);
	const published: PrimitivesHandle | undefined = handle
		? {
				robot: "toy",
				manifest: TOY,
				vars: () => ({}),
				has: () => true,
				tools: () => TOOLS,
				prepare: () => async () => ({ content: [], details: {} }),
				...handle,
			}
		: undefined;
	const emit = async (name: string) => {
		if (name === "session_start" && published) for (const fn of listeners.get(PRIMITIVES_EVENT) ?? []) fn(published);
		for (const fn of handlers.get(name) ?? []) await fn({ type: name }, ctx);
	};
	const turn = (messages: unknown[], signal?: AbortSignal) =>
		provider
			.streamSimple(
				provider.getModels().find((m: any) => m.id === "local"),
				{ messages },
				{ signal },
			)
			.result();
	return { emit, turn, entries, warnings };
}
const calls = (m: any) => m.content.filter((c: any) => c.type === "toolCall");
/** A tool result with a JSON text, two images and `details`. */
const observation = (m: any, text = '{"state": {"tcp_pos": [0.4, 0, 0.3]}}', details: Record<string, unknown> = {}) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [
			{ type: "text", text },
			{ type: "image", data: AGENT, mimeType: "image/png" },
			{ type: "image", data: WRIST, mimeType: "image/png" },
		],
		details,
		isError: false,
	}));
/** Drive the policy until it finishes or `max` turns; the calls it made (name, arguments). */
async function drive(p: ReturnType<typeof fakePi>, max = 10, text?: string) {
	const t0 = await p.turn([]);
	const history: unknown[] = [t0, ...observation(t0, text)];
	const made: [string, any][] = [[calls(t0)[0].name, calls(t0)[0].arguments]];
	let m = t0;
	for (let i = 0; i < max && calls(m)[0]?.name !== "finish"; i++) {
		m = await p.turn(history);
		history.push(m, ...observation(m, text));
		made.push([calls(m)[0].name, calls(m)[0].arguments]);
	}
	return { made, last: m };
}

test("the surface flag defaults to units: the opening is the runner's RELEASE, and a wrong value is refused", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1", "ft-model": "x" }, ["act", "finish"]);
	await p.emit("session_start");
	assert.deepEqual(calls(await p.turn([]))[0], {
		type: "toolCall",
		id: "ft_1",
		name: "act",
		arguments: { unit: "RELEASE" },
	});
	const bad = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1", "ft-model": "x", "ft-surface": "voice" }, [
		"act",
		"finish",
	]);
	await bad.emit("session_start");
	assert.match(bad.warnings.join("\n"), /--ft-surface voice: not one of units, tools, code/);
	assert.deepEqual(calls(await bad.turn([])), []);
	// tools needs the observe tool active (the fallback must be a read that can run); code needs run_code.
	const noObserve = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1", "ft-model": "x", "ft-surface": "tools" }, [
		"move_to",
		"finish",
	]);
	await noObserve.emit("session_start");
	assert.match(noObserve.warnings.join("\n"), /observe tool view_env_state is not active/);
	const noCode = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1", "ft-model": "x", "ft-surface": "code" }, [
		"act",
		"finish",
	]);
	await noCode.emit("session_start");
	assert.match(noCode.warnings.join("\n"), /run_code is not active/);
});

test("student x tools: one stateless request per step lists the tools; the call is checked; garbage falls back to the observe tool; finish ends", async () => {
	const ep = await endpoint([
		'{"tool": "move_to", "arguments": {"xyz": [0.1, 0.2, 0.3], "gripper": -1}}',
		"I would rather MV_DOWN",
		'{"tool": "pi0_pick", "arguments": {"prompt": "the cube"}}',
		'{"tool": "move_to", "arguments": {"xyz": "nope"}}',
		'{"tool": "finish", "arguments": {"status": "success", "summary": "placed"}}',
	]);
	const p = fakePi({ "ft-endpoint": ep.url, "ft-model": "student", "ft-surface": "tools" }, [
		"view_env_state",
		"move_to",
		"finish",
	]);
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.deepEqual(p.warnings, []);
		const { made, last } = await drive(p);
		assert.deepEqual(made, [
			["view_env_state", {}],
			["move_to", { xyz: [0.1, 0.2, 0.3], gripper: -1 }],
			// Unparsable, an unoffered tool, and a schema failure: each falls back to the read, never to a move.
			["view_env_state", {}],
			["view_env_state", {}],
			["view_env_state", {}],
			["finish", { status: "success", summary: "placed" }],
		]);
		assert.equal(last.usage.totalTokens, 0);
		assert.equal(ep.bodies.length, 5);
		const first = ep.bodies[0];
		assert.equal(first.messages.length, 1, "stateless: one user message");
		assert.equal(first.max_tokens, SURFACE_MAX_TOKENS.tools);
		assert.equal(first.temperature, 0);
		assert.equal(first.model, "student");
		const parts = first.messages[0].content;
		assert.deepEqual(
			parts.map((c: any) => c.type),
			["image_url", "image_url", "text"],
			"both images as recorded, then the prompt",
		);
		assert.equal(
			Buffer.from(parts[0].image_url.url.split(",")[1], "base64").toString("base64"),
			AGENT,
			"untransformed",
		);
		const prompt: string = parts[2].text;
		assert.match(prompt, /^You are controlling a robot through its tools\./);
		assert.match(prompt, /\nTask: task PickCube-v1, seed 3\n/);
		assert.match(prompt, /\nRecent calls, newest first: none\n/);
		assert.match(prompt, /\nLast result: \{"state": \{"tcp_pos": \[0\.4, 0, 0\.3\]\}\}\n/);
		assert.match(prompt, /\n- move_to: Servo to xyz\.\n {4}xyz: \[number, number, number\], required: target/);
		assert.match(prompt, /\n- finish: End the episode/);
		assert.match(prompt, /\{"tool": "<tool name>", "arguments": \{\.\.\.\}\}/);
		// The history lists the executed calls newest first (the fallbacks included).
		assert.match(
			ep.bodies[2].messages[0].content[2].text,
			/Recent calls, newest first: view_env_state\(\{\}\), move_to\(\{"xyz":\[0\.1,0\.2,0\.3\],"gripper":-1\}\)\n/,
		);
		const steps = p.entries.filter((e) => e.type === STEP_ENTRY).map((e) => e.data);
		assert.deepEqual(
			steps.map((s) => [s.surface, s.call.name, s.fallback]),
			[
				["tools", "move_to", false],
				["tools", "view_env_state", true],
				["tools", "view_env_state", true],
				["tools", "view_env_state", true],
				["tools", "finish", false],
			],
		);
		assert.match(steps[2].fallback_reason, /pi0_pick is not one of the offered tools/);
		assert.deepEqual(
			steps[0].media.map((x: any) => [x.camera, x.shape]),
			[
				["view0", [64, 64, 3]],
				["view1", [32, 32, 3]],
			],
		);
		assert.equal(steps[0].prompt, prompt);
		assert.deepEqual(steps[4].done, { status: "success", summary: "placed" });
	} finally {
		await ep.close();
	}
});

test("student x tools: env success and max steps end the episode; a failed tool result is an observation", async () => {
	const ep = await endpoint([
		'{"tool": "view_env_state", "arguments": {}}',
		'{"tool": "view_env_state", "arguments": {}}',
	]);
	const p = fakePi({ "ft-endpoint": ep.url, "ft-model": "s", "ft-surface": "tools", "ft-max-steps": "1" }, [
		"view_env_state",
		"finish",
	]);
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		const t0 = await p.turn([]);
		// A failed call: the student is told and goes on.
		const failed = observation(t0, "move_to refused: outside the workspace").map((r: any) => ({
			...r,
			isError: true,
		}));
		const t1 = await p.turn([t0, ...failed]);
		assert.equal(calls(t1)[0].name, "view_env_state");
		assert.match(ep.bodies[0].messages[0].content[2].text, /Last result: move_to refused: outside the workspace\n/);
		const t2 = await p.turn([t0, ...failed, t1, ...observation(t1)]);
		assert.equal(calls(t2)[0].arguments.status, "failure");
		assert.match(calls(t2)[0].arguments.summary, /max_steps \(1\)/);
		const q = fakePi({ "ft-endpoint": ep.url, "ft-model": "s", "ft-surface": "tools" }, ["view_env_state", "finish"]);
		await q.emit("session_start");
		await q.emit("before_agent_start");
		const s0 = await q.turn([]);
		const done = await q.turn([s0, ...observation(s0, "{}", { success: true })]);
		assert.equal(calls(done)[0].arguments.status, "success");
		// One request in all: max steps and env success end the episode without asking the model.
		assert.equal(ep.bodies.length, 1);
	} finally {
		await ep.close();
	}
});

test("student x code: the prompt carries code mode's rendering of the tier; one fenced program runs through run_code; garbage runs the observe-only program; DONE finishes", async () => {
	const ep = await endpoint([
		"```python\nr = move_to([0.1, 0.2, 0.3])\nRESULT = r\n```",
		"Let me think. move_to([0, 0, 0])",
		"```python\na\n```\n```python\nb\n```",
		"DONE",
	]);
	const p = fakePi({ "ft-endpoint": ep.url, "ft-model": "student", "ft-surface": "code", "code-api": "high" }, [
		"run_code",
		"finish",
	]);
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.deepEqual(p.warnings, []);
		const report = 'run_code: {"status": "ran", "stdout": "", "result": null}';
		const { made } = await drive(p, 10, report);
		assert.deepEqual(made, [
			["run_code", { code: NOOP_PROGRAM }],
			["run_code", { code: "r = move_to([0.1, 0.2, 0.3])\nRESULT = r\n" }],
			["run_code", { code: NOOP_PROGRAM }],
			["run_code", { code: NOOP_PROGRAM }],
			["finish", { status: "success", summary: "the policy answered DONE after 3 steps" }],
		]);
		const first = ep.bodies[0];
		assert.equal(first.max_tokens, SURFACE_MAX_TOKENS.code);
		const prompt: string = first.messages[0].content[2].text;
		assert.match(prompt, /^You control a robot arm by writing Python programs\./);
		assert.match(prompt, /\nTask: task PickCube-v1, seed 3\n/);
		assert.match(prompt, /\nLast report: run_code: \{"status": "ran", "stdout": "", "result": null\}\n/);
		assert.match(prompt, /PRIMITIVES \(high tier\)/);
		// code mode's own renderer over the toy manifest's high tier: move_to and get_state, not the low-tier move_delta.
		assert.match(prompt, /\ndef move_to\(xyz: vec3, gripper: number = None\):\n {4}Servo to xyz\.\n/);
		assert.match(prompt, /\ndef get_state\(\):\n {4}Proprioception\./);
		assert.doesNotMatch(prompt, /move_delta/);
		const steps = p.entries.filter((e) => e.type === STEP_ENTRY).map((e) => e.data);
		assert.deepEqual(
			steps.map((s) => [s.surface, s.fallback, s.code === NOOP_PROGRAM]),
			[
				["code", false, false],
				["code", true, true],
				["code", true, true],
				["code", false, false],
			],
		);
		assert.match(steps[1].fallback_reason, /found 0/);
		assert.match(steps[2].fallback_reason, /found 2/);
		assert.equal(steps[3].code, "");
	} finally {
		await ep.close();
	}
});

// ---------------------------------------------------------------------------
// replay x A1 (units) and x A3 (program): the plan kinds, read by content, played by flash/replay

test("readKind judges a plan by its content: a units JSON, an act JSONL, a .py or a JSON program; a tool plan is not its business", () => {
	const dir = mkdtempSync(join(tmpdir(), "kinds-"));
	const units = join(dir, "cell_units.json");
	writeFileSync(
		units,
		JSON.stringify({
			kind: "units",
			units: [{ unit: "MV_UP", n: 2 }, "GRASP", { unit: "STILL", arm: "left", other: "MV_FWD" }],
		}),
	);
	assert.deepEqual(readKind(units, "cell"), {
		name: "cell",
		kind: "units",
		units: [{ unit: "MV_UP", n: 2 }, { unit: "GRASP" }, { unit: "STILL", arm: "left", other: "MV_FWD" }],
	});
	const acts = join(dir, "cell_recipe.jsonl");
	writeFileSync(acts, '{"action":"act","unit":"MV_DOWN"}\n{"action":"act","unit":"RELEASE","n":1}\n');
	assert.deepEqual(readKind(acts, "cell")?.kind, "units");
	const py = join(dir, "cell_program.py");
	writeFileSync(py, "# tier: high\nmove_to([0.1, 0.2, 0.3])\n");
	const program = readKind(py, "cell");
	assert.equal(program?.kind, "program");
	assert.equal(program?.kind === "program" ? program.code : "", "# tier: high\nmove_to([0.1, 0.2, 0.3])\n");
	const asJson = join(dir, "p.json");
	writeFileSync(asJson, JSON.stringify({ code: "print(1)\n" }));
	assert.equal(readKind(asJson, "p")?.kind, "program");
	// A tool-call plan or recipe: not a kind of ours.
	const plan = join(dir, "cell_plan.json");
	writeFileSync(plan, JSON.stringify({ plan: [{ action: "move_to", arguments: { xyz: [0, 0, 0] } }] }));
	assert.equal(readKind(plan, "cell"), undefined);
	const recipe = join(dir, "r.jsonl");
	writeFileSync(recipe, '{"action":"move_to","xyz":[0,0,0]}\n{"action":"act","unit":"MV_UP"}\n');
	assert.equal(readKind(recipe, "r"), undefined);
	// Malformed unit entries are refused, not replayed.
	writeFileSync(units, JSON.stringify({ units: [{ unit: "MV_UP", speed: 3 }] }));
	assert.throws(() => readKind(units, "cell"), /unknown unit field speed/);
	assert.deepEqual(kindFiles("cell"), ["cell_units.json", "cell_program.py", "cell_program.json"]);
});

/** A stub pi for flash (as test/flash.test.ts) with `turn` asking the replay provider for a model turn. */
function flashPi(flagValues: Record<string, unknown> = {}) {
	flagValues = deployFlags(flagValues);
	const handlers = new Map<string, Handler[]>();
	const flags: Record<string, unknown> = {};
	const providers: any[] = [];
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = name in flagValues ? flagValues[name] : o.default;
		},
		getFlag: (name: string) => (name in flags ? flags[name] : flagValues[name]),
		registerProvider: (p: any) => providers.push(p),
		getActiveTools: () => ["act", "run_code", "finish"],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = { hasUI: false, cwd: "/", ui: { notify: () => {} }, sessionManager: { getBranch: () => [] } };
	const emit = async (name: string) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name }, ctx);
	};
	const turn = (messages: unknown[] = []) => {
		const provider = providers.find((p) => p.id === "flash");
		return provider.streamSimple(provider.getModels()[0], { messages }, {}).result();
	};
	return { pi, emit, turn };
}
type Exec = (name: string, args: any) => { text: string; details?: Record<string, unknown>; isError?: boolean };
/** Answer every turn's calls with `exec` until the replay calls finish. */
async function driveFlash(p: ReturnType<typeof flashPi>, exec: Exec) {
	await p.emit("session_start");
	const history: unknown[] = [];
	const sent: [string, any][] = [];
	const notes: string[] = [];
	for (let i = 0; i < 50; i++) {
		const m = await p.turn(history);
		notes.push(
			m.content
				.filter((c: any) => c.type === "text")
				.map((c: any) => c.text)
				.join("\n"),
		);
		const made = calls(m);
		const finish = made.find((c: any) => c.name === "finish");
		if (finish) return { sent, notes: notes.join("\n"), finish: finish.arguments };
		assert.ok(made.length, `turn ${i} called nothing`);
		sent.push(...made.map((c: any): [string, any] => [c.name, c.arguments]));
		history.push(m);
		for (const c of made) {
			const r = exec(c.name, c.arguments);
			history.push({
				role: "toolResult",
				toolCallId: c.id,
				toolName: c.name,
				content: [
					{ type: "text", text: r.text },
					{ type: "image", data: AGENT, mimeType: "image/png" },
				],
				details: r.details ?? {},
				isError: r.isError ?? false,
			});
		}
	}
	throw new Error("the replay never finished");
}
/** The maniskill-style hook: success in the observation JSON ends and solves. */
const hookOver = (program: KindProgram | FlashProgram): FlashHook => ({
	load: () => program,
	start: async () => {
		throw new Error("start must not run for a units or program plan");
	},
	over: (latest) => latest.json.success === true,
	solved: (latest) => latest.json.success === true,
});

test("replay x A1: a unit sequence is sent as act calls in order, untouched, until the robot reports success", async () => {
	const units = [
		{ unit: "MV_DOWN", n: 2 },
		{ unit: "GRASP" },
		{ unit: "MV_UP" },
		{ unit: "RELEASE" },
		{ unit: "MV_BACK" },
	];
	const p = flashPi();
	flash(p.pi, hookOver({ name: "cell", kind: "units", units }));
	let n = 0;
	// The act result as units mode returns it: a header line, then the robot's observation JSON.
	const out = await driveFlash(p, () => ({
		text: `units: x x1\nRecent units: none\n${JSON.stringify({ success: ++n === 4 })}`,
		details: { success: n === 4 },
	}));
	assert.deepEqual(out.sent, [
		["act", { unit: "MV_DOWN", n: 2 }],
		["act", { unit: "GRASP" }],
		["act", { unit: "MV_UP" }],
		["act", { unit: "RELEASE" }],
	]);
	assert.equal(out.finish.status, "success");
	assert.match(
		out.finish.summary,
		/^replayed the cell unit sequence: 4 of 5 units sent \(pure replay, nothing re-anchored\)/,
	);
	assert.match(out.notes, /pure replay: 5 units as recorded \(nothing is re-anchored\)/);
	// A failed act ends the replay with an error summary; an unsolved sequence is a failure.
	const q = flashPi();
	flash(q.pi, hookOver({ name: "cell", kind: "units", units: units.slice(0, 2) }));
	const failed = await driveFlash(q, (_name, args) =>
		args.unit === "GRASP" ? { text: "act: gripper fault", isError: true } : { text: "{}" },
	);
	assert.equal(failed.finish.status, "failure");
	assert.match(failed.finish.summary, /^flash error: act failed: act: gripper fault/);
	const r = flashPi();
	flash(r.pi, hookOver({ name: "cell", kind: "units", units: units.slice(0, 2) }));
	const unsolved = await driveFlash(r, () => ({ text: "{}" }));
	assert.equal(unsolved.finish.status, "failure");
	assert.match(unsolved.finish.summary, /2 of 2 units sent/);
});

test("replay x A3: a program runs once through run_code, then finish with the robot's success", async () => {
	const code = "# tier: high\nr = move_to([0.1, 0.2, 0.3])\nRESULT = r\n";
	const p = flashPi();
	flash(p.pi, hookOver({ name: "cell", kind: "program", code, sha256: "abc123def4567" }));
	const out = await driveFlash(p, (name, args) => {
		assert.equal(name, "run_code");
		assert.equal(args.code, code);
		// run_code's result: the report line, then the robot's observation JSON (the success flag).
		return {
			text: 'run_code: {"status": "ran", "n_calls": 1}\n{"success": true, "state": {}}',
			details: { status: "ran", success: true },
		};
	});
	assert.deepEqual(out.sent, [["run_code", { code }]]);
	assert.equal(out.finish.status, "success");
	assert.match(out.finish.summary, /^ran the cell program once through run_code \(ran\)/);
	assert.match(out.notes, /sha256 abc123def456\)/);
	const q = flashPi();
	flash(q.pi, hookOver({ name: "cell", kind: "program", code, sha256: "0" }));
	const failed = await driveFlash(q, () => ({
		text: 'run_code: {"status": "error"}\n{"success": false}',
		details: { status: "error" },
	}));
	assert.equal(failed.finish.status, "failure");
	assert.match(failed.finish.summary, /through run_code \(error\)/);
});

test("--flash-plan names any plan kind directly; a tool plan needs the hook's loadFile; the A2 path is untouched", async () => {
	const dir = mkdtempSync(join(tmpdir(), "flash-plan-"));
	const file = join(dir, "x_units.json");
	writeFileSync(file, JSON.stringify({ units: ["MV_UP"] }));
	const p = flashPi({ "flash-plan": file });
	const loads: string[] = [];
	flash(p.pi, {
		...hookOver({ name: "never", plan: [] }),
		load: () => {
			loads.push("load");
			return { name: "never", plan: [] };
		},
	});
	const out = await driveFlash(p, () => ({ text: '{"success": true}' }));
	assert.deepEqual(out.sent, [["act", { unit: "MV_UP" }]]);
	assert.deepEqual(loads, [], "the robot's own search is skipped");
	// A tool plan named by file: this hook has no loadFile, so the replay is invalid.
	const plan = join(dir, "cell_plan.json");
	writeFileSync(plan, JSON.stringify({ plan: [{ action: "move_to", arguments: {} }] }));
	const invalid: string[] = [];
	const q = flashPi({ "flash-plan": plan });
	flash(q.pi, hookOver({ name: "never", plan: [] }), { invalid: (why) => invalid.push(why) });
	const refused = await driveFlash(q, () => ({ text: "{}" }));
	assert.match(refused.finish.summary, /neither a units sequence nor a program/);
	assert.equal(invalid.length, 1);
	// With loadFile the hook's own start and rewrite run, as for a plan found in its directories.
	const r = flashPi({ "flash-plan": plan });
	flash(r.pi, {
		load: () => {
			throw new Error("not here");
		},
		loadFile: (path) => ({ name: "cell", plan: JSON.parse(readFileSync(path, "utf8")).plan }),
		start: async () => ({
			localized: 0,
			rewrite: (e) => ({ name: e.action, arguments: { ...e.arguments, xyz: [1, 2, 3] } }),
		}),
		over: (l) => l.json.success === true,
		solved: (l) => l.json.success === true,
	});
	const tools = await driveFlash(r, () => ({ text: '{"success": true}' }));
	assert.deepEqual(tools.sent, [["move_to", { xyz: [1, 2, 3] }]]);
	assert.match(tools.finish.summary, /^replayed the cell program: 1 actions, 0 anchors re-localized/);
});

test("replayUnits stops at the robot's success and reports what it sent", async () => {
	const sent: string[] = [];
	let latest = { json: {} as Record<string, unknown>, images: [] as string[] };
	const robot = {
		act: async () => [],
		move: async (c: { name: string; arguments: Record<string, unknown> }) => {
			sent.push(String(c.arguments.unit));
			latest = { json: { success: sent.length === 2 }, images: [] };
			return latest;
		},
		latest: () => latest,
		note: () => {},
	};
	const out = await replayUnits(
		{ name: "u", kind: "units", units: ["A", "B", "C"].map((unit) => ({ unit })) },
		robot,
		(l) => l.json.success === true,
	);
	assert.deepEqual([out.sent, sent], [2, ["A", "B"]]);
});

// ---------------------------------------------------------------------------
// flash-generate --kind units | program, and the SFT data scripts, from one fake teacher session

/** A session on the toy robot: an observation, a move, a program, units, finish; the last motion solves. */
function session(o: { claimed?: string; env_error?: boolean } = {}): SessionEntry[] {
	const steps: [string, Record<string, unknown>, Record<string, unknown>, boolean][] = [
		["view_env_state", {}, {}, false],
		["move_to", { xyz: [0.1, 0.2, 0.3], gripper: -1 }, { state: { tcp_pos: [0.1, 0.2, 0.3] } }, false],
		["run_code", { code: "r = get_state()\nRESULT = r\n" }, { status: "ran", run: {} }, false],
		["run_code", { code: "boom(" }, { status: "error" }, false],
		["act", { unit: "MV_DOWN", n: 2 }, { ran: 2 }, false],
		["act", { unit: "GRASP" }, { ran: 1 }, false],
		["act", { unit: "MV_UP" }, { ran: 0 }, false],
		["run_code", { code: "move_to([0.1, 0.2, 0.4])\n" }, { status: "ran", success: true, terminated: true }, false],
		["finish", { status: "success", summary: "done" }, {}, false],
	];
	const entries: SessionEntry[] = [
		{
			type: "custom",
			customType: "robot_task",
			data: { robot: "toy", task: "PickCube-v1", seed: "3" },
		} as unknown as SessionEntry,
		{
			type: "custom",
			customType: "code_api",
			data: { tier: null, available: ["move_to", "get_state", "move_delta"], manifest_digest: "", digest: "" },
		} as unknown as SessionEntry,
	];
	steps.forEach(([name, args, details, isError], i) => {
		const id = `c${i}`;
		entries.push({
			type: "message",
			message: { role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }] },
		} as unknown as SessionEntry);
		entries.push({
			type: "message",
			message: {
				role: "toolResult",
				toolCallId: id,
				toolName: name,
				content: [
					{
						type: "text",
						text: name === "run_code" ? `run_code: ${JSON.stringify(details)}` : JSON.stringify(details),
					},
					...(name === "finish"
						? []
						: [
								{ type: "image", data: AGENT, mimeType: "image/png" },
								{ type: "image", data: WRIST, mimeType: "image/png" },
							]),
				],
				details,
				isError,
			},
		} as unknown as SessionEntry);
	});
	entries.push({
		type: "custom",
		customType: "robot_result",
		data: {
			robot: "toy",
			claimed: o.claimed ?? "success",
			env_error: o.env_error ?? false,
			planner_error: null,
			code_api: "high",
		},
	} as unknown as SessionEntry);
	return entries;
}

test("flash-generate writes a unit sequence (the successful act calls) and a program (the successful run_code programs) from a solved session", () => {
	const entries = session();
	assert.deepEqual(sessionUnits(entries), [{ unit: "MV_DOWN", n: 2 }, { unit: "GRASP" }]);
	assert.deepEqual(sessionPrograms(entries), ["r = get_state()\nRESULT = r\n", "move_to([0.1, 0.2, 0.4])\n"]);
	const dir = mkdtempSync(join(tmpdir(), "gen-"));
	const file = join(dir, "s.jsonl");
	writeFileSync(file, entries.map((e) => `${JSON.stringify(e)}\n`).join(""));
	const u = generateKind({ kind: "units", session: file, name: "cell", destination: join(dir, "flash") });
	assert.deepEqual([u.kind, u.calls], ["units", 2]);
	assert.deepEqual(readKind(u.path, "cell"), {
		name: "cell",
		kind: "units",
		units: [{ unit: "MV_DOWN", n: 2 }, { unit: "GRASP" }],
		source: file,
	});
	const g = generateKind({ kind: "program", session: file, name: "cell", destination: join(dir, "flash") });
	assert.deepEqual([g.kind, g.calls, g.path.endsWith("cell_program.py")], ["program", 2, true]);
	const program = readKind(g.path, "cell");
	assert.equal(
		program?.kind === "program" ? program.code : "",
		`# source: ${file}\n# kind: program\n# programs: 2\n\n# --- program 1 ---\nr = get_state()\nRESULT = r\n\n# --- program 2 ---\nmove_to([0.1, 0.2, 0.4])\n`,
	);
	// An unsolved session is refused.
	const unsolved = entries.filter(
		(e) =>
			!(e.type === "message" && (e.message as any).toolCallId === "c7") &&
			!(e.type === "message" && (e.message as any).content?.[0]?.id === "c7"),
	);
	assert.throws(() => sessionUnits(unsolved), /did not solve/);
});

test("sessions_to_sft renders the provider's prompts for the teacher's tool calls and programs; --dry-run counts", () => {
	const entries = session();
	const stored: string[] = [];
	const image = (step: number, i: number) => {
		stored.push(`${step}_${i}`);
		return `images/s/${step}_${i}.png`;
	};
	const tools = toolSamples(entries, [OBSERVE, MOVE_TO, FINISH_TOOL], "task PickCube-v1, seed 3", image);
	// view_env_state (no observation before it: skipped), move_to, finish (after the last observation); run_code and act are not offered tools.
	assert.deepEqual(
		tools.map((s) => s.output),
		[
			'{"tool":"move_to","arguments":{"xyz":[0.1,0.2,0.3],"gripper":-1}}',
			'{"tool":"finish","arguments":{"status":"success","summary":"done"}}',
		],
	);
	assert.match(tools[0].instruction, /^<image><image>You are controlling a robot through its tools\./);
	assert.match(tools[0].instruction, /\nTask: task PickCube-v1, seed 3\n/);
	assert.match(tools[0].instruction, /\nLast result: \{\}\n/);
	assert.match(
		tools[0].instruction,
		/\n- move_to: Servo to xyz\.\n {4}xyz: \[number, number, number\], required: target \[x, y, z\] in metres\n/,
	);
	assert.match(
		tools[1].instruction,
		/\nRecent calls, newest first: move_to\(\{"xyz":\[0\.1,0\.2,0\.3\],"gripper":-1\}\)\n/,
	);
	assert.deepEqual(tools[0].images, ["images/s/0_0.png", "images/s/0_1.png"]);
	const code = codeSamples(
		entries,
		{ tier: "high", api: "def move_to(xyz: vec3, gripper: number = None):\n    Servo to xyz." },
		"task PickCube-v1, seed 3",
		image,
	);
	// The errored program is left out; each kept one is the fenced program.
	assert.deepEqual(
		code.map((s) => s.output),
		["```python\nr = get_state()\nRESULT = r\n```", "```python\nmove_to([0.1, 0.2, 0.4])\n```"],
	);
	assert.match(code[0].instruction, /^<image><image>You control a robot arm by writing Python programs\./);
	assert.match(
		code[0].instruction,
		/\nPRIMITIVES \(high tier\)[^\n]*\ndef move_to\(xyz: vec3, gripper: number = None\):\n/,
	);
	// The observation before the second program is the act result (this session mixes surfaces): shown as it was.
	assert.match(code[1].instruction, /\nLast report: \{"ran":0\}\n/);
	// The CLI path over a session file, dry and for real.
	const dir = mkdtempSync(join(tmpdir(), "sft-"));
	writeFileSync(join(dir, "good.jsonl"), entries.map((e) => `${JSON.stringify(e)}\n`).join(""));
	writeFileSync(
		join(dir, "bad.jsonl"),
		session({ claimed: "failure" })
			.map((e) => `${JSON.stringify(e)}\n`)
			.join(""),
	);
	const dry = convert({ surface: "code", sessions: [dir], out: join(dir, "out"), "dry-run": true });
	assert.deepEqual(
		[dry.out, dry.samples, dry.sessions.map((s) => [s.samples, s.skipped ?? null])],
		[
			null,
			2,
			[
				[0, "not a successful run"],
				[2, null],
			],
		],
	);
	const both = convert({
		surface: "code",
		sessions: [dir],
		out: join(dir, "out"),
		"dry-run": true,
		"include-failures": true,
	});
	assert.equal(both.samples, 4);
	const real = convert({
		surface: "tools",
		sessions: [join(dir, "good.jsonl")],
		out: join(dir, "out"),
		tools: "move_to",
	});
	assert.equal(real.samples, 2);
	const written = JSON.parse(readFileSync(join(dir, "out", "sft.json"), "utf8"));
	assert.equal(written.length, 2);
	// The manifest's move_to and finish only (--tools), rendered from the manifest with the same renderer.
	assert.match(
		written[0].instruction,
		/\nTools:\n- move_to: Servo to xyz\.\n {4}xyz: \[number, number, number\], required: target \[x, y, z\] in metres\n {4}gripper: number: -1 opens, \+1 closes\n- finish: End the episode/,
	);
	assert.doesNotMatch(written[0].instruction, /segment/);
	assert.equal(readFileSync(join(dir, "out", written[0].images[0])).toString("base64"), AGENT);
	assert.deepEqual(parseVars("arms=left,right;max_move=0.2;name=x;empty="), {
		arms: ["left", "right"],
		max_move: 0.2,
		name: "x",
		empty: [],
	});
});
