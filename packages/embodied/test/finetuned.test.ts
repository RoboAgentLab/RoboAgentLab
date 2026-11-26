import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { isContextOverflow, isRetryableAssistantError, normalizeContext } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { encodePng } from "../src/infra/png.ts";
import finetuned, {
	activeSubgoal,
	allowedTokens,
	buildRequest,
	DUAL_PROMPTS,
	formatPrompt,
	isV5,
	PRESETS,
	PROMPTS,
	parsePair,
	parseToken,
	prepareImages,
	RELEASED,
	recentText,
	STEP_ENTRY,
	V5_ACTIONS,
} from "../src/modes/finetuned/index.ts";
import { convertRun, findRuns, trainedTokens } from "../src/modes/finetuned/prepare.ts";
import { decodePng, fingerprint, parseView, prepareView } from "../src/modes/finetuned/views.ts";
import { RT_UNITS, UNITS, UNITS_EVENT, type UnitsHandle } from "../src/modes/units/index.ts";
import { LARGEST } from "../src/planner/call-ids.ts";
import { RELAY_ERRORS } from "../src/planner/fallback.ts";
import { deployFlags } from "./helpers/deployment.ts";
import { within } from "./helpers/pi-session.ts";

/**
 * The reference: Show-Harness @137d571's own MvTokenController + VLMClient.complete_action_token
 * (with _post_chat captured) and core/record/images.prepare_view under opencv 4.11 / numpy 1.26, fed
 * the two synthetic frames below; image parts are replaced by the pixel fingerprint of the PNG sent.
 */
const TASK = "pick up the red cube and place it on the plate";
const REF_PROMPT = `You are controlling a robot arm with two cameras:
- Agentview: overhead view of the robot and workspace
- Wristview: close-up view from the gripper

Task: ${TASK}
Recent moves, newest first: MV_LEFT, MV_DOWN

Output exactly one action token:
MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, GRASP, RELEASE, DONE

Check both camera views and choose the next action:
- Use AgentView to locate the target when it is not in wrist view
- Use wrist view to fine-align when target is visible up close
- GRASP when gripper fingers are aligned around the object
- RELEASE when object is above the destination
- DONE when the task is complete: the object is at its destination and the gripper is clear
- Avoid repeating a direction that conflicts with the most recent move

Return the single token only, no punctuation, no explanation:`;
const REF = {
	raw: {
		agent: { sha1: "b18f71cedaac", shape: [512, 512, 3] },
		wrist: { sha1: "ff8788f8ee4b", shape: [480, 640, 3] },
	},
	payload: {
		model: "qwen3_5_2b_showharness_sim",
		messages: [
			{
				role: "user",
				content: [
					{ type: "image_url", image_url: { url: "<png>", sha1: "6c883980c15b", shape: [256, 256, 3] } },
					{ type: "image_url", image_url: { url: "<png>", sha1: "1e53f10798cd", shape: [256, 256, 3] } },
					{ type: "text", text: REF_PROMPT },
				],
			},
		],
		temperature: 0,
		max_tokens: 24,
		chat_template_kwargs: { enable_thinking: false, thinking: false },
	},
	/** vlm_client._parse_single_token on these replies (null: it raises, and the runner falls back). */
	parsed: {
		MV_LEFT: "MV_LEFT",
		" GRASP\n": "GRASP",
		"<think>hmm MV_UP</think>DONE": "DONE",
		'{"token": "RELEASE"}': "RELEASE",
		"I will MV_FWD.": "MV_FWD",
		"Therefore the answer is MV_BACK because": "MV_BACK",
		"MV_DOWN is best": "MV_DOWN",
		"move left": null,
		MV_LEFTX: null,
		"MV_UP then MV_DOWN": "MV_DOWN",
		mv_left: null,
		// upstream returns "grasp" as written, which no controller executes; we return the canonical unit.
		"final: grasp": "GRASP",
		"The gripper should go MV_RIGHT.": "MV_RIGHT",
	} as Record<string, string | null>,
};

/** The synthetic frame of the reference script: (x*3+y+seed, x*y+seed, x^y) mod 256. */
function pattern(h: number, w: number, seed: number) {
	const rgb = Buffer.alloc(h * w * 3);
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++) {
			const o = (y * w + x) * 3;
			rgb[o] = (x * 3 + y + seed) % 256;
			rgb[o + 1] = (x * y + seed) % 256;
			rgb[o + 2] = (x ^ y) & 255;
		}
	return { width: w, height: h, rgb };
}
const agentRaw = pattern(512, 512, 7);
const wristRaw = pattern(480, 640, 11);
const SPECS = [parseView("square=256"), parseView("rot=90,flip=vertical,crop=1.3333,square=256")];
const png = (img: { width: number; height: number; rgb: Buffer }) => encodePng(img.rgb, img.width, img.height);

/** A request body with its images replaced by the fingerprints of their pixels, as REF stores them. */
function fingerprinted(body: any) {
	const b = structuredClone(body);
	for (const part of b.messages[0].content)
		if (part.type === "image_url") {
			const url: string = part.image_url.url;
			assert.ok(url.startsWith("data:image/png;base64,"));
			part.image_url = { url: "<png>", ...fingerprint(decodePng(Buffer.from(url.split(",")[1], "base64"))) };
		}
	return b;
}

test("the request matches Show-Harness's own rendering: prompt bytes, image order, pixels, sampling", () => {
	assert.deepEqual(fingerprint(agentRaw), REF.raw.agent);
	assert.deepEqual(fingerprint(wristRaw), REF.raw.wrist);
	const tpl = PROMPTS.v3;
	const prompt = formatPrompt(tpl, {
		task: TASK,
		recent_moves: recentText(["MV_LEFT", "MV_DOWN"]),
		gripper_state: "open",
	});
	assert.equal(prompt, REF_PROMPT);
	const sent = prepareImages(
		[png(agentRaw), png(wristRaw)].map((p) => ({ data: p.toString("base64") })),
		[0, 1],
		SPECS,
	);
	const body = buildRequest(
		"qwen3_5_2b_showharness_sim",
		prompt,
		sent.map((s) => s.png),
	);
	assert.deepEqual(fingerprinted(body), REF.payload);
	assert.deepEqual(allowedTokens(tpl), [
		"MV_FWD",
		"MV_BACK",
		"MV_LEFT",
		"MV_RIGHT",
		"MV_UP",
		"MV_DOWN",
		"GRASP",
		"RELEASE",
		"DONE",
	]);
	assert.deepEqual(fingerprint(prepareView(wristRaw, SPECS[1])), fingerprint(sent[1].view));
	assert.equal(formatPrompt("a {{b}} {c}", { c: "d" }), "a {b} d");
	assert.throws(() => formatPrompt("{missing}", {}), /has no value/);
	assert.equal(recentText([]), "none");
});

test("replies parse like vlm_client._parse_single_token", () => {
	const allowed = allowedTokens(PROMPTS.v3);
	for (const [raw, want] of Object.entries(REF.parsed)) {
		if (want === null) assert.throws(() => parseToken(raw, allowed), /invalid token/, raw);
		else assert.equal(parseToken(raw, allowed), want, raw);
	}
});

// ---------------------------------------------------------------------------
// the provider, driven like pi's agent loop, against a local OpenAI-compatible endpoint

type Handler = (event: any, ctx: any) => unknown;

/** A reply the endpoint holds back: `arrived` once its request is in; it is sent when the test calls `release`. */
function hold() {
	const h = { reached: () => {}, release: () => {}, arrived: Promise.resolve(), released: Promise.resolve() };
	h.arrived = new Promise<void>((resolve) => {
		h.reached = () => resolve();
	});
	h.released = new Promise<void>((resolve) => {
		h.release = () => resolve();
	});
	return h;
}
type Hold = ReturnType<typeof hold>;

async function endpoint(replies: (string | { status: number; body: string } | { held: Hold; reply: string })[]) {
	const bodies: any[] = [];
	const server = createServer(async (req: IncomingMessage, res) => {
		let data = "";
		for await (const chunk of req) data += chunk;
		bodies.push({ path: req.url, auth: req.headers.authorization, body: JSON.parse(data) });
		let r = replies.shift() ?? "DONE";
		if (typeof r !== "string" && "held" in r) {
			r.held.reached();
			await r.held.released;
			r = r.reply;
		}
		if (typeof r !== "string") {
			res.writeHead(r.status, { "content-type": "application/json" });
			res.end(r.body);
			return;
		}
		res.writeHead(200, { "content-type": "application/json" });
		res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: r } }] }));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
	return {
		url,
		bodies,
		// A request still held back must not keep the server open.
		close: () =>
			new Promise<void>((resolve) => {
				server.closeAllConnections();
				server.close(() => resolve());
			}),
	};
}

/** The stub's first session id. pi sends a session's id with every request of that session's own loop. */
const SESSION = "session-1";
/** The model a step is asked for unless a test names another. */
const ADAPTER = "qwen3_5_2b_showharness_sim";
/** A user message to ask with, as the dashboard's model check asks. */
const CHECK = { role: "user", content: "Reply with the single word OK.", timestamp: 0 };

function fakePi(
	flags: Record<string, string>,
	robot = "maniskill",
	tools = ["act", "finish"],
	units?: Partial<UnitsHandle>,
) {
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
	/** The id of the session the stub is in: a test moves it on for a new session. */
	const session = { id: SESSION };
	/** The session's file as pi reads it (every entry, every branch): a test puts an earlier episode's in it. */
	const file: unknown[] = [];
	const ctx = {
		hasUI: true,
		model: { provider: "finetuned" },
		ui: { notify: (s: string) => warnings.push(s) },
		sessionManager: {
			getBranch: () => [{ type: "custom", customType: "robot_task", data: { robot } }],
			getEntries: () => file,
			getSessionId: () => session.id,
		},
	};
	finetuned(pi);
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		// The robot's units layer publishes its handle at session start, before the provider's handlers.
		if (name === "session_start" && units) for (const fn of listeners.get(UNITS_EVENT) ?? []) fn(units);
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	};
	const model = (id: string) => provider.getModels().find((m: any) => m.id === id);
	/** One step as pi's agent loop asks for it: with the session's id. It is awaited ten seconds at most. */
	const turn = (messages: unknown[], id = ADAPTER, signal?: AbortSignal) =>
		within<any>(
			10_000,
			provider.streamSimple(model(id), { messages }, { signal, sessionId: session.id }).result(),
			"the session's own turn",
		);
	/**
	 * A request that is not the loop's, as ctx.modelRegistry.streamSimple hands one over (the dashboard's model check,
	 * a question to a VLM): one user message unless a transcript or a whole context is given, and no session id
	 * unless `options` holds one.
	 */
	const ask = (options: Record<string, unknown> = {}, context?: unknown[] | object, id = ADAPTER) =>
		provider.streamSimple(
			model(id),
			Array.isArray(context) ? { messages: context } : (context ?? { messages: [CHECK] }),
			{ maxTokens: 1, ...options },
		);
	/** The answer to such a request, awaited two seconds at most (one taken for a turn may wait on the endpoint). */
	const side = (options: Record<string, unknown> = {}, context?: unknown[] | object, id = ADAPTER) =>
		within<any>(2000, ask(options, context, id).result(), "a request that is not the session's turn");
	/** The answer to one from a caller that passes no options at all. */
	const bare = (id = ADAPTER) =>
		within<any>(2000, provider.streamSimple(model(id), { messages: [] }).result(), "a request without options");
	return { emit, turn, ask, side, bare, session, file, ctx, entries, warnings, model };
}

const calls = (m: any) => m.content.filter((c: any) => c.type === "toolCall");
/** The units `act` result: its header (with the TASK line), then the robot's state and images. */
const observation = (m: any, details: Record<string, unknown> = {}) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [
			{ type: "text", text: `units: ${c.arguments.unit} x1\nRecent units: none\nTASK: ${TASK}` },
			{ type: "text", text: "{}" },
			{ type: "image", data: png(agentRaw).toString("base64"), mimeType: "image/png" },
			{ type: "image", data: png(wristRaw).toString("base64"), mimeType: "image/png" },
		],
		details,
		isError: false,
	}));

test("one stateless request per step; the token becomes a units act; DONE finishes", async () => {
	const ep = await endpoint(["MV_DOWN", "MV_LEFT", "no idea", "GRASP", "DONE"]);
	const p = fakePi({ "ft-endpoint": ep.url, "ft-wrist": "rot=90,flip=vertical,crop=1.3333,square=256" });
	try {
		assert.deepEqual(
			[...RELEASED, "local"].map((id) => Boolean(p.model(id))),
			[...RELEASED, "local"].map(() => true),
		);
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.deepEqual(p.warnings, []);
		// The opening RELEASE asks no model.
		const t0 = await p.turn([]);
		assert.deepEqual(calls(t0)[0].arguments, { unit: "RELEASE" });
		assert.equal(t0.usage.totalTokens, 0);
		assert.equal(ep.bodies.length, 0);
		const history: unknown[] = [t0, ...observation(t0)];
		const units: string[] = [];
		let m = t0;
		for (let i = 0; i < 6 && calls(m)[0]?.name === "act"; i++) {
			m = await p.turn(history);
			history.push(m, ...observation(m));
			units.push(
				calls(m)
					.map((c: any) => (c.name === "act" ? c.arguments.unit : c.name))
					.join(),
			);
		}
		// "no idea" parses to nothing: the runner's MV_DOWN fallback.
		assert.deepEqual(units, ["MV_DOWN", "MV_LEFT", "MV_DOWN", "GRASP", "finish"]);
		assert.deepEqual(calls(m)[0].arguments.status, "success");
		assert.equal(ep.bodies.length, 5);
		assert.equal(ep.bodies[0].path, "/v1/chat/completions");
		assert.equal(ep.bodies[0].auth, "Bearer EMPTY");
		// Step 2's request is exactly the reference: after MV_DOWN then MV_LEFT, newest first.
		const second = fingerprinted(ep.bodies[2].body);
		assert.deepEqual(second, REF.payload);
		// GRASP / RELEASE never enter the move history; each request is one user message.
		const last = ep.bodies[4].body;
		assert.equal(last.messages.length, 1);
		assert.match(last.messages[0].content[2].text, /Recent moves, newest first: MV_DOWN, MV_LEFT, MV_DOWN\n/);
		const steps = p.entries.filter((e) => e.type === STEP_ENTRY).map((e) => e.data);
		assert.deepEqual(
			steps.map((s) => [s.token, s.fallback]),
			[
				["MV_DOWN", false],
				["MV_LEFT", false],
				["MV_DOWN", true],
				["GRASP", false],
				["DONE", false],
			],
		);
		assert.equal(steps[2].raw, "no idea");
		assert.deepEqual(
			steps[2].media.map((x: any) => [x.camera, x.sha1]),
			[
				["agentview", "6c883980c15b"],
				["wrist", "1e53f10798cd"],
			],
		);
		// pi stops after finish; asked again, the policy is spent.
		assert.equal(calls(await p.turn(history)).length, 0);
	} finally {
		await ep.close();
	}
});

test("the execution swap changes only the executed unit; env success, max steps and errors end the episode", async () => {
	const ep = await endpoint(["MV_FWD", "MV_FWD"]);
	const p = fakePi({ "ft-endpoint": ep.url, "ft-swap": "MV_FWD,MV_BACK", "ft-max-steps": "1" }, "piper");
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		const t0 = await p.turn([]);
		const h: unknown[] = [t0, ...observation(t0)];
		const t1 = await p.turn(h);
		assert.deepEqual(calls(t1)[0].arguments, { unit: "MV_BACK" });
		h.push(t1, ...observation(t1));
		const t2 = await p.turn(h);
		assert.deepEqual(calls(t2)[0].arguments.status, "failure");
		assert.match(calls(t2)[0].arguments.summary, /max_steps \(1\)/);
		const step = p.entries.find((e) => e.type === STEP_ENTRY)?.data;
		assert.deepEqual([step.token, step.executed], ["MV_FWD", "MV_BACK"]);
	} finally {
		await ep.close();
	}

	const q = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" }, "maniskill");
	await q.emit("session_start");
	const t0 = await q.turn([]);
	const done = await q.turn([t0, ...observation(t0, { terminated: true, success: true })]);
	assert.deepEqual(calls(done)[0].arguments.status, "success");
	assert.match(calls(done)[0].arguments.summary, /environment reports success/);

	// A non-retryable HTTP error is a model error, not a fallback unit.
	const bad = await endpoint([{ status: 400, body: '{"error":"no such adapter"}' }]);
	const r = fakePi({ "ft-endpoint": bad.url }, "robolab", ["finish"]);
	try {
		await r.emit("session_start");
		await r.emit("before_agent_start");
		assert.match(r.warnings.join("\n"), /--units/);
		const s0 = await r.turn([]);
		const err = await r.turn([s0, ...observation(s0)]);
		assert.equal(err.stopReason, "error");
		assert.match(err.errorMessage, /HTTP 400/);
		// An aborted turn ends the policy.
		const ac = new AbortController();
		ac.abort();
		assert.equal((await r.turn([], "qwen3_5_2b_showharness_sim", ac.signal)).stopReason, "aborted");
		// finetuned/local needs an adapter name.
		await r.emit("session_start");
		const l0 = await r.turn([], "local");
		assert.match((await r.turn([l0, ...observation(l0)], "local")).errorMessage, /--ft-model/);
	} finally {
		await bad.close();
	}
});

test("prepare turns a GUMI run into a Show-Harness rollout with the provider's camera transform", () => {
	const run = mkdtempSync(join(tmpdir(), "gumi-"));
	mkdirSync(join(run, "images", "agentview"), { recursive: true });
	mkdirSync(join(run, "images", "wrist"), { recursive: true });
	writeFileSync(join(run, "images", "agentview", "0000.png"), png(agentRaw));
	writeFileSync(join(run, "images", "wrist", "0000.png"), png(wristRaw));
	const rows = [
		{ step: 0, token: "MV_DOWN", gripper_closed: false, agentview: "images/agentview/0000.png", src: "agent" },
		{
			step: 1,
			token: "GRASP",
			gripper_closed: true,
			agentview: "images/agentview/0000.png",
			wrist: "images/wrist/0000.png",
		},
	];
	writeFileSync(join(run, "actions.jsonl"), `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`);
	const out = mkdtempSync(join(tmpdir(), "rollouts-"));
	const views = { agentview: SPECS[0], wrist: SPECS[1] };
	const r = convertRun(run, join(out, "t"), 0, views, TASK, { robot: "libero" });
	assert.equal(r.rows, 1);
	assert.match(r.warnings.join(), /dropped steps 0/);
	const line = JSON.parse(readFileSync(join(r.out, "actions.jsonl"), "utf8").trim());
	assert.deepEqual(
		[line.token, line.gripper_closed, line.agentview, line.wrist],
		["GRASP", true, "agentview/0000.png", "wrist/0000.png"],
	);
	// Byte-identical to what the provider sends for the same frames (the REF fingerprints).
	assert.equal(fingerprint(decodePng(readFileSync(join(r.out, line.agentview)))).sha1, "6c883980c15b");
	assert.equal(fingerprint(decodePng(readFileSync(join(r.out, line.wrist)))).sha1, "1e53f10798cd");
	const meta = JSON.parse(readFileSync(join(r.out, "metadata.json"), "utf8"));
	assert.equal(meta.task_text, TASK);
	assert.deepEqual(findRuns(run), [run]);
});

test("prepare keeps RT_* turns for the v5 vocabulary (15 units) and drops them for v3", () => {
	assert.equal(trainedTokens("v5-libero").length, 15);
	assert.ok(RT_UNITS.every((u) => trainedTokens("v5-libero").includes(u)));
	assert.ok(!RT_UNITS.some((u) => trainedTokens("v3").includes(u)));
	assert.throws(() => trainedTokens("v9"), /unknown prompt version v9/);
	const run = mkdtempSync(join(tmpdir(), "gumi-"));
	mkdirSync(join(run, "images", "agentview"), { recursive: true });
	mkdirSync(join(run, "images", "wrist"), { recursive: true });
	writeFileSync(join(run, "images", "agentview", "0000.png"), png(agentRaw));
	writeFileSync(join(run, "images", "wrist", "0000.png"), png(wristRaw));
	const frame = { agentview: "images/agentview/0000.png", wrist: "images/wrist/0000.png" };
	const tokens = ["MV_DOWN", "RT_PITCH_FWD", "ROTATE_CW", "RT_YAW_CCW", "GRASP"];
	const rows = tokens.map((token, step) => ({ step, token, kind: "move", gripper_closed: false, ...frame }));
	writeFileSync(join(run, "actions.jsonl"), `${rows.map((r) => JSON.stringify(r)).join("\n")}\n`);
	const views = { agentview: SPECS[0], wrist: SPECS[1] };
	const convert = (version?: string) => {
		const out = mkdtempSync(join(tmpdir(), "rollouts-"));
		const r = convertRun(run, join(out, "t"), 0, views, TASK, { robot: "libero" }, version);
		const lines = readFileSync(join(r.out, "actions.jsonl"), "utf8")
			.trim()
			.split("\n")
			.map((l) => JSON.parse(l));
		const meta = JSON.parse(readFileSync(join(r.out, "metadata.json"), "utf8"));
		return { r, lines, meta };
	};
	const v3 = convert();
	// ROTATE_* stays for the converter to skip, as before; the images are renumbered densely.
	assert.deepEqual(
		v3.lines.map((l) => l.token),
		["MV_DOWN", "ROTATE_CW", "GRASP"],
	);
	assert.deepEqual(
		v3.lines.map((l) => l.agentview),
		["agentview/0000.png", "agentview/0001.png", "agentview/0002.png"],
	);
	assert.match(v3.r.warnings.join(), /dropped RT_\* steps 1, 3 \(v3 has no turns/);
	assert.equal(v3.meta.prompt_version, "v3");
	const v5 = convert("v5-libero");
	assert.deepEqual(
		v5.lines.map((l) => l.token),
		tokens,
	);
	assert.deepEqual(v5.r.warnings, []);
	assert.equal(v5.meta.prompt_version, "v5-libero");
});

const V5_TEMPLATE = `Task: {task}
Recent moves, newest first: {recent_moves}
Output one of: MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, RT_ROLL_LEFT, RT_ROLL_RIGHT, RT_PITCH_FWD, RT_PITCH_BACK, RT_YAW_CW, RT_YAW_CCW, GRASP, RELEASE, DONE`;

/** A v5 run on a robot whose units layer reports `eef` (m) at each decision. */
function v5Run(ep: { url: string }, eef: number[][], vocabulary: readonly string[] = UNITS, flags = {}) {
	const file = join(mkdtempSync(join(tmpdir(), "v5-")), "prompt_v5.txt");
	writeFileSync(file, `${V5_TEMPLATE}\n`);
	return fakePi(
		{ "ft-endpoint": ep.url, "ft-prompt": "v5-libero", "ft-prompt-file": file, "units-plugins": "", ...flags },
		"libero",
		["act", "finish"],
		{ vocabulary, state: async () => ({ eef_xyz: eef.shift() ?? [0, 0, 0] }) },
	);
}

test("v5: 15 units, turns in the history, the no-progress guard, DONE and out-of-vocabulary asked again", async () => {
	assert.deepEqual(
		allowedTokens(V5_TEMPLATE),
		V5_ACTIONS.map((t) => t),
	);
	const ep = await endpoint(["MV_DOWN", "MV_DOWN", "RT_YAW_CW", "DONE", "GRASP", "MV_STRETCH", "MV_UP"]);
	// MV_DOWN moves 1 mm, then 1 mm again: withheld from the third decision.
	const p = v5Run(ep, [
		[0.5, 0, 0.2],
		[0.5, 0, 0.199],
		[0.5, 0, 0.198],
		[0.5, 0, 0.198],
		[0.5, 0, 0.198],
		[0.5, 0, 0.198],
	]);
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.deepEqual(p.warnings, []);
		const t0 = await p.turn([]);
		const history: unknown[] = [t0, ...observation(t0)];
		const units: string[] = [];
		for (let i = 0; i < 5; i++) {
			const m = await p.turn(history);
			history.push(m, ...observation(m, i === 4 ? { terminated: true } : {}));
			units.push(calls(m)[0].arguments.unit);
		}
		assert.deepEqual(units, ["MV_DOWN", "MV_DOWN", "RT_YAW_CW", "GRASP", "MV_UP"]);
		const done = await p.turn(history);
		assert.equal(calls(done)[0].arguments.status, "success");
		assert.match(calls(done)[0].arguments.summary, /environment reports success/);

		const b = ep.bodies.map((x) => x.body);
		assert.equal(b.length, 7);
		// Unconstrained unless something is withheld.
		assert.equal(b[0].structured_outputs, undefined);
		assert.deepEqual(
			b[2].structured_outputs.choice,
			V5_ACTIONS.filter((t) => t !== "MV_DOWN"),
		);
		// DONE counted, then asked again without it; the history holds the turn.
		assert.deepEqual(
			b[4].structured_outputs.choice,
			V5_ACTIONS.filter((t) => t !== "DONE"),
		);
		assert.match(b[4].messages[0].content[2].text, /Recent moves, newest first: RT_YAW_CW, MV_DOWN, MV_DOWN\n/);
		// Out of vocabulary: asked again, constrained to the whole vocabulary.
		assert.equal(b[5].structured_outputs, undefined);
		assert.deepEqual(b[6].structured_outputs.choice, [...V5_ACTIONS]);
		const steps = p.entries.filter((e) => e.type === STEP_ENTRY).map((e) => e.data);
		assert.deepEqual(
			steps.map((s) => [s.token, s.withheld ?? null, s.moved_mm ?? null]),
			[
				["MV_DOWN", null, null],
				["MV_DOWN", null, 1],
				["RT_YAW_CW", ["MV_DOWN"], 1],
				["GRASP", null, null],
				["MV_UP", null, null],
			],
		);
		assert.equal(steps[3].done_ignored, 1);
		assert.equal(steps[4].asks[0].raw, "MV_STRETCH");
	} finally {
		await ep.close();
	}
});

test("v5: the budget, an endpoint that ignores the constraint, and a robot that cannot turn", async () => {
	// The step budget ends a v5 episode; the reason names env success, not DONE.
	const b = await endpoint(["MV_FWD"]);
	const budget = v5Run(b, [], UNITS, { "ft-max-steps": "1" });
	try {
		await budget.emit("session_start");
		const t0 = await budget.turn([]);
		const t1 = await budget.turn([t0, ...observation(t0)]);
		assert.equal(calls(t1)[0].arguments.unit, "MV_FWD");
		const t2 = await budget.turn([t0, ...observation(t0), t1, ...observation(t1)]);
		assert.match(calls(t2)[0].arguments.summary, /max_steps \(1\) reached without env success/);
	} finally {
		await b.close();
	}

	// DONE asked again, but the endpoint answers DONE anyway: a model error, never a silent finish.
	const ignoring = await endpoint(["DONE", "DONE"]);
	const q = v5Run(ignoring, []);
	try {
		await q.emit("session_start");
		const s0 = await q.turn([]);
		const err = await q.turn([s0, ...observation(s0)]);
		assert.equal(err.stopReason, "error");
		assert.match(err.errorMessage, /ignored structured_outputs/);
	} finally {
		await ignoring.close();
	}

	// Without RT_* in the robot's act: a warning up front, and an emitted turn ends the episode with the reason.
	const turnless = await endpoint(["RT_ROLL_LEFT"]);
	const r = v5Run(
		turnless,
		[],
		UNITS.filter((u) => !u.startsWith("RT_")),
	);
	try {
		await r.emit("session_start");
		await r.emit("before_agent_start");
		assert.match(r.warnings.join("\n"), /does not offer RT_ROLL_LEFT.*--units-rt=true/);
		const s0 = await r.turn([]);
		const end = await r.turn([s0, ...observation(s0)]);
		assert.equal(calls(end)[0].name, "finish");
		assert.equal(calls(end)[0].arguments.status, "failure");
		assert.match(calls(end)[0].arguments.summary, /RT_ROLL_LEFT, which this robot's act does not offer/);
	} finally {
		await turnless.close();
	}

	// v5 reads the vendored prompt (no warning); plugins that regrind a token are flagged.
	const none = fakePi({ "ft-prompt": "v5-libero", "units-plugins": "variable_step,mem_text" }, "libero");
	await none.emit("session_start");
	await none.emit("before_agent_start");
	assert.doesNotMatch(none.warnings.join("\n"), /not vendored/);
	assert.match(none.warnings.join("\n"), /units plugins variable_step change what a token does/);
});

test("fine-tuned mode refuses a robot whose images are not an agentview and a wrist view, before any step", async () => {
	const base = { vocabulary: ["MV_FWD", "GRASP", "RELEASE", "DONE"], arms: [] as string[] };
	// No wrist camera (ManiSkill widowxai, the default dual Franka, a UR5e with fixed cameras).
	const none = fakePi({}, "maniskill", ["act", "finish"], { ...base, wrist: () => false });
	await none.emit("session_start");
	assert.match(
		none.warnings.join("\n"),
		/finetuned: the fine-tuned adapters read an agentview and a wrist image, and this robot configuration has no wrist camera/,
	);
	const t = await none.turn([]);
	assert.deepEqual(calls(t), [], "no act: not even the opening RELEASE");
	assert.match(t.content[0].text, /refused this robot/);
	// --ft-cameras must name a third-person view and a wrist view of the robot's observation.
	const nav = { ...base, wrist: () => true, views: () => ({ views: 3, wrist: 2 }) };
	const wrong = fakePi({}, "robocasa", ["act", "finish"], nav);
	await wrong.emit("session_start");
	assert.match(
		wrong.warnings.join("\n"),
		/--ft-cameras 0,1: image 1 must be a wrist view and image 0 a third-person view \(the robot's wrist views are image\(s\) 2\)/,
	);
	assert.deepEqual(calls(await wrong.turn([])), []);
	const short = fakePi({}, "maniskill", ["act", "finish"], { ...base, views: () => ({ views: 1 }) });
	await short.emit("session_start");
	assert.match(
		short.warnings.join("\n"),
		/--ft-cameras 0,1 names image 1, but the robot's observation carries 1 image/,
	);
	// The right pair runs.
	const right = fakePi({ "ft-cameras": "0,2" }, "robocasa", ["act", "finish"], nav);
	await right.emit("session_start");
	assert.doesNotMatch(right.warnings.join("\n"), /refused|ft-cameras/);
	assert.deepEqual(calls(await right.turn([]))[0].arguments, { unit: "RELEASE" });
	const libero = fakePi({}, "libero", ["act", "finish"], {
		...base,
		wrist: () => true,
		views: () => ({ views: 2, wrist: 1 }),
	});
	await libero.emit("session_start");
	assert.deepEqual(calls(await libero.turn([]))[0].arguments, { unit: "RELEASE" });
});

/** A dual-arm units `act` result: the header, then the front, left wrist and right wrist images. */
const dualObservation = (m: any) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [
			{ type: "text", text: `units: ${c.arguments.unit} x1\nTASK: fold the towel` },
			{ type: "image", data: png(agentRaw).toString("base64"), mimeType: "image/png" },
			{ type: "image", data: png(wristRaw).toString("base64"), mimeType: "image/png" },
			{ type: "image", data: png(wristRaw).toString("base64"), mimeType: "image/png" },
		],
		details: {},
		isError: false,
	}));
const DUAL_UNITS: Partial<UnitsHandle> = {
	arms: ["left", "right"],
	vocabulary: [...UNITS],
	wrist: () => true,
	views: () => ({ views: 3, wrist: [1, 2] }),
};

test("dual prompts: the v4 templates verbatim, one token per arm, and the pair parser", () => {
	assert.match(DUAL_PROMPTS["v4-dual-once"].template, /^You are controlling a dual-arm robot with three cameras:/);
	assert.match(DUAL_PROMPTS["v4-dual-once"].template, /Return the two tokens only, no punctuation, no explanation:$/);
	assert.match(DUAL_PROMPTS["v4-dual-twice"].template, /Control the \{arm\} arm now\./);
	assert.match(DUAL_PROMPTS["v4-dual-chain"].followup ?? "", /Now choose the next action for the RIGHT arm/);
	const allowed = ["MV_FWD", "MV_UP", "STILL", "DONE"];
	assert.deepEqual(parsePair("MV_UP STILL", allowed), ["MV_UP", "STILL"]);
	assert.deepEqual(
		parsePair("<think>x</think>STILL, MV_FWD.", allowed),
		["STILL", "MV_FWD"],
		"reading order, not vocabulary order",
	);
	assert.throws(() => parsePair("MV_UP", allowed), /needs TWO tokens/);
});

test("dual fine-tuned (once): both arms open, pairs run as one paired act, a lone DONE holds, both DONE finish", async () => {
	const ep = await endpoint(["MV_UP MV_FWD", "STILL GRASP", "DONE MV_UP", "garbage", "DONE DONE"]);
	const p = fakePi(
		{ "ft-endpoint": ep.url, "ft-prompt": "v4-dual-once" },
		"piper_dual",
		["act", "finish"],
		DUAL_UNITS,
	);
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		const t0 = await p.turn([]);
		assert.deepEqual(calls(t0)[0].arguments, { unit: "RELEASE", arm: "left", other: "RELEASE" });
		const history: unknown[] = [t0, ...dualObservation(t0)];
		const acts: unknown[] = [];
		let m = t0;
		for (let i = 0; i < 6 && calls(m)[0]?.name === "act"; i++) {
			m = await p.turn(history);
			history.push(m, ...dualObservation(m));
			acts.push(calls(m)[0].name === "act" ? calls(m)[0].arguments : calls(m)[0].name);
		}
		assert.deepEqual(acts, [
			{ unit: "MV_UP", arm: "left", other: "MV_FWD" },
			{ unit: "GRASP", arm: "right" },
			{ unit: "MV_UP", arm: "right" },
			// Unparsable: both arms hold; STOP on one arm returns the next observation.
			{ unit: "STOP", arm: "left" },
			"finish",
		]);
		assert.equal(calls(m)[0].arguments.status, "success");
		const body = ep.bodies[3].body;
		assert.equal(body.max_tokens, 48, "two tokens: the budget doubles");
		assert.equal(body.messages[0].content.filter((c: any) => c.type === "image_url").length, 3);
		const text = body.messages[0].content[3].text;
		// Per-arm history, newest first, STILL included (the lone DONE is recorded as the STILL it ran).
		assert.match(text, /Left arm recent moves, newest first: STILL, STILL, MV_UP\n/);
		assert.match(text, /Right arm recent moves, newest first: MV_UP, MV_FWD\n/);
		const steps = p.entries.filter((e) => e.type === STEP_ENTRY).map((e) => e.data);
		assert.deepEqual(steps[2].executed, { left: "STILL", right: "MV_UP" });
		assert.equal(steps[3].fallback, true);
		assert.deepEqual(
			steps[0].media.map((x: any) => x.camera),
			["agentview", "wrist_left", "wrist_right"],
		);
	} finally {
		await ep.close();
	}
});

test("dual fine-tuned twice and chain: the request shapes of the scheme", async () => {
	const ep = await endpoint(["MV_UP", "MV_FWD", "MV_DOWN", "GRASP"]);
	try {
		const t = fakePi(
			{ "ft-endpoint": ep.url, "ft-prompt": "v4-dual-twice" },
			"piper_dual",
			["act", "finish"],
			DUAL_UNITS,
		);
		await t.emit("session_start");
		const t0 = await t.turn([]);
		const t1 = await t.turn([t0, ...dualObservation(t0)]);
		assert.deepEqual(calls(t1)[0].arguments, { unit: "MV_UP", arm: "left", other: "MV_FWD" });
		const [left, right] = ep.bodies.map((b) => b.body.messages[0].content.at(-1).text);
		assert.match(left, /Control the LEFT arm now\./);
		assert.match(right, /Control the RIGHT arm now\./);
		assert.equal(
			right,
			left
				.replaceAll("LEFT arm", "RIGHT arm")
				.replaceAll("LEFT wrist", "RIGHT wrist")
				.replaceAll("LEFT gripper fingers", "RIGHT gripper fingers")
				.replaceAll("LEFT gripper holds", "RIGHT gripper holds"),
			"the right call is not told the left token",
		);

		const c = fakePi(
			{ "ft-endpoint": ep.url, "ft-prompt": "v4-dual-chain" },
			"piper_dual",
			["act", "finish"],
			DUAL_UNITS,
		);
		await c.emit("session_start");
		const c0 = await c.turn([]);
		const c1 = await c.turn([c0, ...dualObservation(c0)]);
		assert.deepEqual(calls(c1)[0].arguments, { unit: "MV_DOWN", arm: "left", other: "GRASP" });
		const follow = ep.bodies[3].body.messages;
		assert.equal(follow.length, 3);
		assert.deepEqual(follow[1], { role: "assistant", content: "MV_DOWN" });
		assert.match(follow[2].content, /Now choose the next action for the RIGHT arm/);
	} finally {
		await ep.close();
	}
	// A dual prompt on one arm, and a single-arm prompt on two, are refused before any step.
	const one = fakePi({ "ft-prompt": "v4-dual-once" }, "piper", ["act", "finish"], {
		arms: [],
		vocabulary: [...UNITS],
	});
	await one.emit("session_start");
	assert.match(one.warnings.join("\n"), /drives two arms .* this robot has 1/);
	const two = fakePi({ "ft-prompt": "v4-piper" }, "piper_dual", ["act", "finish"], DUAL_UNITS);
	await two.emit("session_start");
	assert.match(two.warnings.join("\n"), /this robot has two arms/);
	// A dual prompt names the arms by side (tokens, images, act's `arm`): two arms called otherwise are refused too.
	const named = fakePi({ "ft-prompt": "v4-dual-once" }, "robosuite", ["act", "finish"], {
		...DUAL_UNITS,
		arms: ["robot0", "robot1"],
		// robosuite's two-arm observation: the names are the reason given, not the cameras.
		views: () => ({ views: 2, wrist: 1 }),
	});
	await named.emit("session_start");
	assert.match(
		named.warnings.join("\n"),
		/--ft-prompt v4-dual-once names its two arms left and right, and this robot's are robot0, robot1/,
	);
});

test("fine-tuned prompts on two arms: a dual prompt is advised only to a pair the dual prompts name, and a dual prompt's count and names are refused before the camera", async () => {
	const start = async (flags: Record<string, string>, robot: string, units: Partial<UnitsHandle>) => {
		const p = fakePi(flags, robot, ["act", "finish"], units);
		await p.emit("session_start");
		return p.warnings.join("\n");
	};
	// A pair the dual prompts name, with its wrist cameras: the advice, and every advised prompt starts there.
	assert.match(
		await start({ "ft-prompt": "v3" }, "piper_dual", DUAL_UNITS),
		/cannot drive it; pass a dual prompt \(v4-dual-once, v4-dual-twice, v4-dual-chain\)$/,
	);
	for (const p of Object.keys(DUAL_PROMPTS))
		assert.equal(await start({ "ft-prompt": p }, "piper_dual", DUAL_UNITS), "");
	// A pair of other names (robosuite's two-arm tasks): no dual prompt starts there, so none is advised.
	const other = await start({ "ft-prompt": "v3" }, "robosuite", { ...DUAL_UNITS, arms: ["robot0", "robot1"] });
	assert.match(
		other,
		/the single-arm prompt v3 cannot drive it, and the dual prompts name their arms left and right \(this robot's are robot0, robot1\); run it with a planner model instead$/,
	);
	assert.doesNotMatch(other, /pass a dual prompt/);
	// Two arms and no wrist camera (ManiSkill's panda_pair): no prompt runs there, and the camera is the reason given.
	const blind = await start({ "ft-prompt": "v3" }, "maniskill", {
		...DUAL_UNITS,
		wrist: () => false,
		views: () => ({ views: 1 }),
	});
	assert.match(blind, /this robot configuration has no wrist camera; run it with a planner model instead$/);
	assert.doesNotMatch(blind, /pass a dual prompt/);
	// With a dual prompt too, on the pair it names.
	assert.match(
		await start({ "ft-prompt": "v4-dual-once" }, "maniskill", {
			...DUAL_UNITS,
			wrist: () => false,
			views: () => ({ views: 1 }),
		}),
		/this robot configuration has no wrist camera; run it with a planner model instead$/,
	);
	// A dual prompt's own refusals come before the camera: the arm count, then the names.
	assert.match(
		await start({ "ft-prompt": "v4-dual-once" }, "maniskill", {
			arms: [],
			vocabulary: [...UNITS],
			wrist: () => false,
		}),
		/drives two arms \(one token per arm\), and this robot has 1$/,
	);
	assert.match(
		await start({ "ft-prompt": "v4-dual-once" }, "robosuite", {
			...DUAL_UNITS,
			arms: ["robot0", "robot1"],
			wrist: () => false,
		}),
		/names its two arms left and right, and this robot's are robot0, robot1; run it with a planner model instead$/,
	);
	// The order of the pair counts, as for GUMI: right, left is not the pair the dual prompts name.
	const swapped = { ...DUAL_UNITS, arms: ["right", "left"] };
	assert.match(
		await start({ "ft-prompt": "v4-dual-once" }, "piper_dual", swapped),
		/--ft-prompt v4-dual-once names its two arms left and right, and this robot's are right, left; run it with a planner model instead$/,
	);
	assert.match(
		await start({ "ft-prompt": "v3" }, "piper_dual", swapped),
		/cannot drive it, and the dual prompts name their arms left and right \(this robot's are right, left\); run it with a planner model instead$/,
	);
});

const PLAN = [
	{
		motion: "GRASP",
		target: "banana",
		affordance: "left end",
		description: "approach then close",
		completion: "banana between the fingers",
	},
	{
		motion: "LIFT",
		target: "banana",
		affordance: "left end",
		description: "lift",
		completion: "banana above the table",
	},
	{ motion: "MOVE", target: "plate", affordance: "center", description: "carry", completion: "banana over the plate" },
	{ motion: "RELEASE", target: "plate", affordance: "center", description: "open", completion: "banana on the plate" },
	{
		motion: "RETREAT",
		target: "gripper",
		affordance: "gripper",
		description: "lift away",
		completion: "gripper clear",
	},
];

test("subgoal conditioning follows rollouts_to_alpaca.py's phase anchors and even split", () => {
	assert.equal(activeSubgoal(PLAN, 0, 5, 20)?.index, 0, "approach + grasp: the GRASP subgoal");
	assert.equal(activeSubgoal(PLAN, 1, 0, 20)?.index, 1, "transport starts at LIFT");
	assert.equal(activeSubgoal(PLAN, 1, 12, 20)?.index, 2, "and moves on to MOVE over the phase");
	assert.equal(activeSubgoal(PLAN, 1, 99, 20)?.index, 2, "never past the phase");
	assert.equal(activeSubgoal(PLAN, 2, 0, 20)?.index, 3);
	assert.equal(activeSubgoal(PLAN, 2, 15, 20)?.index, 4);
	assert.equal(activeSubgoal([], 0, 0, 20), undefined);
});

test("v3-subgoal and v3-affordance prompts carry the task's config; the policy's GRASP moves the phase", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ft-cond-"));
	writeFileSync(join(dir, "task_config.json"), JSON.stringify({ task: TASK, subgoals: PLAN }));
	writeFileSync(
		join(dir, "affordance_config.json"),
		JSON.stringify({ task: TASK, target: "banana", affordance: "left end" }),
	);
	const ep = await endpoint(["GRASP", "MV_UP", "MV_LEFT"]);
	try {
		const p = fakePi({
			"ft-endpoint": ep.url,
			"ft-prompt": "v3-subgoal",
			"ft-task-config": join(dir, "task_config.json"),
		});
		await p.emit("session_start");
		const t0 = await p.turn([]);
		const h: unknown[] = [t0, ...observation(t0)];
		for (let i = 0; i < 2; i++) {
			const m = await p.turn(h);
			h.push(m, ...observation(m));
		}
		const texts = ep.bodies.map((b) => b.body.messages[0].content.at(-1).text as string);
		assert.match(
			texts[0],
			/Stage: GRASP\nTarget: banana\nAffordance: left end\nStage goal: approach then close\nDone when: banana between the fingers\n/,
		);
		assert.match(texts[0], /between the black gripper fingers/);
		assert.match(texts[1], /Stage: LIFT\n/, "after the policy's GRASP: transport");
		const steps = p.entries.filter((e) => e.type === STEP_ENTRY).map((e) => e.data.conditioning);
		assert.deepEqual(steps[1], { phase: 1, phase_step: 0, subgoal: 1 });

		const q = fakePi({
			"ft-endpoint": ep.url,
			"ft-prompt": "v3-affordance",
			"ft-affordance-config": join(dir, "affordance_config.json"),
		});
		await q.emit("session_start");
		const q0 = await q.turn([]);
		await q.turn([q0, ...observation(q0)]);
		assert.match(ep.bodies.at(-1)?.body.messages[0].content.at(-1).text, /Grasp first: banana, at left end\n/);
		// Without its config the conditioned prompt fails rather than rendering blanks.
		const r = fakePi({ "ft-endpoint": ep.url, "ft-prompt": "v3-subgoal" });
		await r.emit("session_start");
		const r0 = await r.turn([]);
		const failed = await r.turn([r0, ...observation(r0)]);
		assert.equal(failed.stopReason, "error");
		assert.match(failed.errorMessage, /needs --ft-task-config/);
	} finally {
		await ep.close();
	}
});

test("v5-libero runs the vendored prompt_v5.txt verbatim with the 15-unit vocabulary", () => {
	const tpl = readFileSync(new URL("../src/modes/finetuned/templates/v5_libero_mvtoken.txt", import.meta.url), "utf8");
	const vendored = readFileSync(
		new URL("../../../services/pi_embodied_services/finetuned/showharness/prompts/v5/prompt_v5.txt", import.meta.url),
		"utf8",
	);
	assert.equal(tpl, vendored);
	assert.deepEqual(allowedTokens(tpl.trim()), [...V5_ACTIONS]);
	assert.equal(trainedTokens("v5-libero").length, 15);
	assert.match(formatPrompt(tpl.trim(), { task: "t", recent_moves: "none" }), /^You are controlling a robot arm/);
	// The file is not Show-Harness's: its NOTICE names the source dataset, the declared license and this exact hash (U4).
	const notice = readFileSync(
		new URL("../../../services/pi_embodied_services/finetuned/showharness/prompts/v5/NOTICE", import.meta.url),
		"utf8",
	);
	assert.match(notice, /huggingface\.co\/datasets\/aaroncaozj\/libero_show-harness_tokenized/);
	assert.match(notice, /Declared license: MIT/);
	assert.ok(
		notice.includes(createHash("sha256").update(vendored).digest("hex")),
		"the NOTICE names the vendored file's sha256",
	);
});

test("--ft-prompt names a prompt version or a template file; anything else is refused before the first step (U3)", async () => {
	// v5-lite (a misspelt v5-libero) used to start and run prompt_v5 under v3's rules; v5 was an alias.
	for (const name of ["v5-lite", "v5", "v6-libero"]) {
		const p = fakePi({ "ft-prompt": name }, "libero");
		await p.emit("session_start");
		assert.match(
			p.warnings.join("\n"),
			new RegExp(`--ft-prompt ${name}: not a prompt version \\(.*v5-libero.*\\) and not a template file`),
			name,
		);
	}
	assert.equal(isV5("v5"), false);
	assert.equal(isV5("v5-libero"), true);
	assert.equal(PRESETS.v5, undefined, "no alias of v5-libero");
	// A known version, and a template file by path, start.
	const ok = fakePi({ "ft-prompt": "v5-libero" }, "libero");
	await ok.emit("session_start");
	assert.doesNotMatch(ok.warnings.join("\n"), /not a prompt version/);
	const file = join(mkdtempSync(join(tmpdir(), "ft-")), "prompt.txt");
	writeFileSync(file, `${V5_TEMPLATE}\n`);
	const byPath = fakePi({ "ft-prompt": file }, "libero");
	await byPath.emit("session_start");
	assert.doesNotMatch(byPath.warnings.join("\n"), /not a prompt version/);
});

// ---------------------------------------------------------------------------
// a request that is not a turn of the session's own loop (the dashboard's model check, a question to a VLM, a check
// that asks the session's model, pi's own summaries)

/** What a request to a fine-tuned model gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"finetuned runs a fine-tuned policy and answers no other request; --aux-model names a model for the checks that take one";
const NO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
/** `m` is that answer: an error of the model asked (`id`), with no content and no usage. */
function assertNotATurn(m: any, id = ADAPTER) {
	assert.equal(m.stopReason, "error");
	assert.equal(m.errorMessage, NOT_A_TURN);
	assert.deepEqual(m.content, []);
	assert.deepEqual([m.api, m.provider, m.model], ["finetuned", "finetuned", id]);
	assert.deepEqual(m.usage, { ...NO_COST, totalTokens: 0, cost: { ...NO_COST, total: 0 } });
}

/**
 * What the policy takes for no turn of its session: a request without an id, under another id, or with pi's mark on
 * a one-off summary under the session's id. Each one as it is, and with a signal that is aborted when it arrives.
 */
const NOT_TURNS = Object.entries<Record<string, unknown>>({
	"without an id": {},
	"under another id": { sessionId: "0193e1c0-0000-7000-8000-000000000000" },
	"as a one-off summary under the session's id": { sessionId: SESSION, cacheRetention: "none" },
}).flatMap(([kind, options]): [string, Record<string, unknown>][] => [
	[kind, options],
	[`${kind}, arriving aborted`, { ...options, signal: AbortSignal.abort() }],
]);

/** A context of a request's own, as a tool's agent loop or pi's summaries send one: a system prompt and tools. */
const OWN = normalizeContext({
	systemPrompt: "You are a policy.",
	tools: [{ name: "policy_act", description: "act", parameters: { type: "object", properties: {} } as never }],
	messages: [CHECK as never],
});
/**
 * A request of this kind gets the error with each transcript a check sends: one user message, the loop's own as it is
 * at this place, and a context of its own; and with a signal that is live when it arrives, as the dashboard's model
 * check sends one (a request that arrives aborted keeps its own).
 */
async function refused(p: ReturnType<typeof fakePi>, options: Record<string, unknown>, transcript: unknown[]) {
	for (const context of [undefined, transcript, OWN]) assertNotATurn(await p.side(options, context));
	assertNotATurn(await p.side({ signal: new AbortController().signal, ...options }));
}

const said = (m: any) =>
	m.content
		.filter((c: any) => c.type === "text")
		.map((c: any) => c.text)
		.join("\n");
/** A turn as the loop got it: its stop reason, its text (or its error) without the latency, and its calls. */
const seen = (m: any) => [
	m.stopReason,
	m.errorMessage ?? said(m).replace(/ · \d+ ms/g, ""),
	calls(m).map((c: any) => [c.id, c.name, c.arguments]),
];
/** The places of an episode where a request from elsewhere can arrive, in the order an episode first reaches them. */
const PLACES = [
	"before the first step",
	"while a step's call runs",
	"between two steps",
	"while the service answers a step",
	"after the end",
];
type At = (place: string, transcript: unknown[], p: ReturnType<typeof fakePi>) => Promise<void>;

/**
 * One episode as pi's agent loop runs it: the calls of each turn get `results` (by turn), until a turn calls nothing;
 * then the loop asks once more, as for a second prompt. `at` runs at each place where a request from elsewhere can
 * arrive, with the transcript the loop has there; while `held` holds back the service's reply to a step, `at` runs
 * inside the loop's request that waits for it. Returns each turn as the loop got it.
 */
async function episode(p: ReturnType<typeof fakePi>, results: (m: any, i: number) => unknown[], at?: At, held?: Hold) {
	const visit = async (place: string, transcript: unknown[]) => at?.(place, [...transcript], p);
	const turns: unknown[] = [];
	const history: unknown[] = [];
	await visit("before the first step", history);
	for (let i = 0; i < 40; i++) {
		const asked = p.turn([...history]);
		const h = held;
		if (h && (await Promise.race([asked.then(() => false), h.arrived.then(() => true)]))) {
			held = undefined;
			try {
				await visit("while the service answers a step", history);
			} finally {
				h.release();
			}
		}
		const m = await asked;
		turns.push(seen(m));
		if (!calls(m).length) break;
		await visit("while a step's call runs", [...history, m]);
		history.push(m, ...results(m, i));
		await visit("between two steps", history);
	}
	await visit("after the end", history);
	turns.push(seen(await p.turn([...history])));
	return turns;
}
/** Small camera images for these episodes: the policy's transform still runs on them, at a fraction of the time. */
const VIEWS = [pattern(32, 32, 7), pattern(24, 32, 11)].map((v) => png(v).toString("base64"));
/** The units `act` result of each call of `m`: its header with the TASK line, the robot's state and two views. */
const seenBy = (m: any, details: Record<string, unknown> = {}) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [
			{ type: "text", text: `units: ${c.arguments.unit} x1\nRecent units: none\nTASK: ${TASK}` },
			{ type: "text", text: "{}" },
			...VIEWS.map((data) => ({ type: "image", data, mimeType: "image/png" })),
		],
		details,
		isError: false,
	}));
/** The same on two arms: the front view and each arm's wrist view. */
const seenByBoth = (m: any) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [
			{ type: "text", text: `units: ${c.arguments.unit} x1\nTASK: fold the towel` },
			...[VIEWS[0], VIEWS[1], VIEWS[1]].map((data) => ({ type: "image", data, mimeType: "image/png" })),
		],
		details: {},
		isError: false,
	}));
/** The steps the policy wrote, without their latency and the endpoint's address, which differ from run to run. */
const written = (p: ReturnType<typeof fakePi>) =>
	p.entries.map((e) => [
		e.type,
		Object.fromEntries(Object.entries(e.data).filter(([k]) => k !== "latency_ms" && k !== "endpoint")),
	]);

/**
 * One arm (v5, a robot that reports its TCP): MV_DOWN, MV_DOWN again (1 mm each: the third decision is asked without
 * it), RT_YAW_CW, DONE asked again (GRASP), DONE asked again (MV_UP), an out-of-vocabulary reply asked again (MV_LEFT),
 * then the environment's success. The service holds back its reply to the second decision. Returns what the loop got,
 * the steps the policy wrote and what the service was asked.
 */
async function oneArm(at?: At) {
	const held = hold();
	const ep = await endpoint([
		"MV_DOWN",
		{ held, reply: "MV_DOWN" },
		"RT_YAW_CW",
		"DONE",
		"GRASP",
		"DONE",
		"MV_UP",
		"MV_STRETCH",
		"MV_LEFT",
	]);
	const tcp = [0.2, 0.199, 0.198, 0.198, 0.198, 0.198].map((z) => [0.5, 0, z]);
	const p = v5Run(ep, tcp);
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		const turns = await episode(p, (m, i) => seenBy(m, i === 6 ? { terminated: true } : {}), at, held);
		return { turns, steps: written(p), asked: ep.bodies.map((b) => b.body) };
	} finally {
		held.release();
		await ep.close();
	}
}

/**
 * A subgoal prompt whose phase moves with the policy's GRASP and RELEASE, two steps to a phase: GRASP, MV_UP, MV_LEFT
 * (the service holds back this reply), RELEASE, MV_UP, DONE.
 */
async function subgoal(at?: At) {
	const dir = mkdtempSync(join(tmpdir(), "ft-cond-"));
	writeFileSync(join(dir, "task_config.json"), JSON.stringify({ task: TASK, subgoals: PLAN }));
	const held = hold();
	const ep = await endpoint(["GRASP", "MV_UP", { held, reply: "MV_LEFT" }, "RELEASE", "MV_UP", "DONE"]);
	const p = fakePi({
		"ft-endpoint": ep.url,
		"ft-prompt": "v3-subgoal",
		"ft-task-config": join(dir, "task_config.json"),
		"ft-phase-steps": "2",
	});
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		const turns = await episode(p, (m) => seenBy(m), at, held);
		return { turns, steps: written(p), asked: ep.bodies.map((b) => b.body) };
	} finally {
		held.release();
		await ep.close();
		rmSync(dir, { recursive: true, force: true });
	}
}

/** Two arms: MV_UP and MV_FWD, STILL and GRASP (the service holds back this reply), a lone DONE, garbage, DONE twice. */
async function twoArms(at?: At) {
	const held = hold();
	const ep = await endpoint(["MV_UP MV_FWD", { held, reply: "STILL GRASP" }, "DONE MV_UP", "garbage", "DONE DONE"]);
	const p = fakePi(
		{ "ft-endpoint": ep.url, "ft-prompt": "v4-dual-once" },
		"piper_dual",
		["act", "finish"],
		DUAL_UNITS,
	);
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		const turns = await episode(p, (m) => seenByBoth(m), at, held);
		return { turns, steps: written(p), asked: ep.bodies.map((b) => b.body) };
	} finally {
		held.release();
		await ep.close();
	}
}

/** Each episode without a request from elsewhere, run once for all the tests that compare with it. */
const alone = new Map<string, ReturnType<typeof oneArm>>();
const without = (run: typeof oneArm) => alone.get(run.name) ?? alone.set(run.name, run()).get(run.name);

for (const [kind, options] of NOT_TURNS)
	test(`a request that is not the session's turn (${kind}) gets the error and leaves the policy as it is: before the first step, while a step's call runs, between two steps, while the service answers a step and after the end, the session's own turns, the steps written and what the service is asked are those of an episode without it (one arm with the stuck guard and replies asked again, a subgoal prompt's phase, two arms' histories)`, async () => {
		for (const run of [oneArm, subgoal, twoArms]) {
			const places: string[] = [];
			const got = await run(async (place, transcript, p) => {
				places.push(place);
				await refused(p, options, transcript);
			});
			assert.deepEqual([...new Set(places)], PLACES, run.name);
			assert.deepEqual(got, await without(run), run.name);
		}
	});

test("before a session starts no request is a session's turn, of whatever kind, with or without options; after it starts, one that is not its turn does not take the opening step", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
	// Each request as it was answered: of each kind, without options, and under the id the session will have.
	const answered = async (what: string, m: Promise<any>) => [what, (await m).stopReason, (await m).errorMessage];
	const before = [];
	for (const [kind, options] of NOT_TURNS) before.push(await answered(kind, p.side(options)));
	before.push(await answered("without options", p.bare()));
	before.push(await answered("under the session's id", p.turn([])));
	assert.deepEqual(
		before,
		[...NOT_TURNS.map(([kind]) => kind), "without options", "under the session's id"].map((what) => [
			what,
			"error",
			NOT_A_TURN,
		]),
	);
	await p.emit("session_start");
	for (const [, options] of NOT_TURNS) await refused(p, options, []);
	assertNotATurn(await p.bare());
	// The session's first step is the opening one, under the first call id; nothing was asked or written.
	assert.deepEqual(seen(await p.turn([])), [
		"toolUse",
		"opening RELEASE (the runner's first step; it returns the first observation)",
		[["ft_1", "act", { unit: "RELEASE" }]],
	]);
	assert.deepEqual(p.entries, []);
});

test("once the policy has refused this robot at the start, a request that is not the session's turn gets the error, and the session's own turn still says why", async () => {
	const why = /^The fine-tuned policy refused this robot: --ft-prompt v5-lite: not a prompt version/;
	// One kind at a time, each in a session of its own: the session's own turn says why after each.
	for (const [kind, options] of NOT_TURNS) {
		const p = fakePi({ "ft-prompt": "v5-lite" }, "libero");
		await p.emit("session_start");
		await refused(p, options, []);
		const t = await p.turn([]);
		assert.deepEqual([kind, calls(t), why.test(said(t))], [kind, [], true]);
		await refused(p, options, [t]);
		assert.deepEqual([kind, why.test(said(await p.turn([t])))], [kind, true]);
		assert.deepEqual(p.entries, []);
	}
	const p = fakePi({ "ft-prompt": "v5-lite" }, "libero");
	await p.emit("session_start");
	assertNotATurn(await p.bare());
	assert.match(said(await p.turn([])), why);
});

test("a request of the session's own loop is its turn whatever retention it names other than pi's one-off mark; a new session's turns are those under its own id", async () => {
	const ep = await endpoint(["MV_DOWN", "MV_UP"]);
	const p = fakePi({ "ft-endpoint": ep.url });
	try {
		await p.emit("session_start");
		const t0 = await p.side({ sessionId: SESSION, cacheRetention: "short" }, []);
		const history = [t0, ...observation(t0)];
		const t1 = await p.side({ sessionId: SESSION, cacheRetention: "long" }, history);
		assert.deepEqual(
			[t0, t1].map((m) => m.stopReason),
			["toolUse", "toolUse"],
			"a turn under short retention, then one under long",
		);
		assert.deepEqual(calls(t0)[0].arguments, { unit: "RELEASE" });
		assert.deepEqual(calls(t1)[0].arguments, { unit: "MV_DOWN" });
		// A new session has an id of its own: a request under the one before is not its turn.
		p.session.id = "session-2";
		await p.emit("session_start");
		assertNotATurn(await p.side({ sessionId: SESSION }, [...history, t1, ...observation(t1)]));
		assert.deepEqual(
			calls(await p.turn([])).map((c: any) => [c.id, c.arguments]),
			[["ft_1", { unit: "RELEASE" }]],
		);
		assert.equal(ep.bodies.length, 1);
	} finally {
		await ep.close();
	}
});

test("an id that only resembles the session's is another id, and a one-off summary is told by its mark alone", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
	await p.emit("session_start");
	const ids = [
		`${SESSION}0`,
		SESSION.toUpperCase(),
		SESSION.slice(0, -1),
		SESSION.slice(1),
		` ${SESSION}`,
		`${SESSION} `,
		"",
	];
	const answers = [];
	for (const sessionId of ids) answers.push([sessionId, (await p.side({ sessionId })).stopReason]);
	// The stub's other requests set a token limit; pi's mark counts without one.
	const mark = { sessionId: SESSION, cacheRetention: "none", maxTokens: undefined };
	answers.push(["the mark without a token limit", (await p.side(mark)).stopReason]);
	assert.deepEqual(
		answers,
		[...ids, "the mark without a token limit"].map((what) => [what, "error"]),
	);
	for (const sessionId of ids) assertNotATurn(await p.side({ sessionId }));
	assertNotATurn(await p.side(mark));
	assert.deepEqual(
		calls(await p.turn([])).map((c: any) => c.id),
		["ft_1"],
	);
});

/** What else a request can carry, as pi's loop or another asker sends it: none of it makes a request a turn or not. */
const CARRIED = {
	reasoning: "low",
	temperature: 0,
	metadata: { user_id: "u" },
	headers: { "x-request": "1" },
	onPayload: () => undefined,
	maxTokens: 64,
	transport: "sse",
};

test("whatever else a request carries (a reasoning level, a temperature, metadata, headers, a payload hook, a token limit, a retention other than pi's mark), the id and the mark decide: without the id, under another id or with the mark it gets the error, and the session's own turn is answered", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
	await p.emit("session_start");
	// Each kind with all of it, under each retention a request names (a one-off summary's is pi's mark).
	const answers = [];
	const expected = [];
	for (const [kind, options] of NOT_TURNS)
		for (const cacheRetention of options.cacheRetention ? [options.cacheRetention] : [undefined, "short", "long"]) {
			const m = await p.side({ ...CARRIED, cacheRetention, ...options });
			answers.push([kind, cacheRetention, m.stopReason, m.errorMessage]);
			expected.push([kind, cacheRetention, "error", NOT_A_TURN]);
		}
	assert.deepEqual(answers, expected);
	// The session's first step, asked with all of it and a retention: the opening one, under the first call id.
	const own = { ...CARRIED, cacheRetention: "long", sessionId: SESSION };
	const opening = within<any>(2000, p.ask(own, []).result(), "the session's own turn");
	assert.deepEqual(seen(await opening), [
		"toolUse",
		"opening RELEASE (the runner's first step; it returns the first observation)",
		[["ft_1", "act", { unit: "RELEASE" }]],
	]);
});

test("a session whose model at its start is another one (the fallback planner over this policy) keeps its id too: the policy's turns are those under it", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
	p.ctx.model = { provider: "fallback" };
	await p.emit("session_start");
	assertNotATurn(await p.side());
	assert.deepEqual(
		calls(await p.turn([])).map((c: any) => [c.id, c.arguments]),
		[["ft_1", { unit: "RELEASE" }]],
	);
});

test("the answer to a request that is not the session's turn reaches its asker as one error event, stamped with the model it asked", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
	await p.emit("session_start");
	const stream = p.ask();
	// Its events and its result, each awaited two seconds at most (a stream that never ends fails the test).
	const events: any[] = [];
	const read = async () => {
		for await (const event of stream) events.push(event);
	};
	await within(2000, read(), "its events");
	assert.deepEqual(
		events.map((e) => [e.type, e.reason]),
		[["error", "error"]],
	);
	assertNotATurn(events[0].error);
	assert.equal(await within(2000, stream.result(), "its result"), events[0].error);
	for (const id of ["local", "gemma4_e4b_showharness_ft"]) {
		assertNotATurn(await p.side({}, undefined, id), id);
		assertNotATurn(await p.bare(id), id);
	}
});

test("the answer to a request that is not the session's turn is no error pi sends again, and none the fallback planner by default changes models for", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
	await p.emit("session_start");
	const m = await p.side();
	assertNotATurn(m);
	// pi sends a request again after an error it takes for a passing one, and after one it takes for a full context.
	assert.equal(isRetryableAssistantError(m), false);
	assert.equal(isContextOverflow(m, p.model(ADAPTER).contextWindow), false);
	assert.doesNotMatch(m.errorMessage, RELAY_ERRORS);
});

test("the session's id is kept at every start pi announces (its first start, a new session, a resumed one, a fork, a reload), each in a provider of its own as pi builds one, and still when a run ends: that session's own request is its turn, and one without an id is not", async () => {
	for (const reason of ["startup", "new", "resume", "fork", "reload"]) {
		const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
		p.session.id = `session-${reason}`;
		await p.emit("session_start", { reason });
		// pi ends each run with agent_end; the session's next prompt is still its own.
		await p.emit("agent_end");
		assertNotATurn(await p.side());
		assert.deepEqual(
			calls(await p.turn([])).map((c: any) => c.name),
			["act"],
			reason,
		);
	}
});

test("the session's id and where its call ids go on from are kept ahead of the rest of the start: when the rest throws (here a robot's wrist hook), the session's own turns are still answered, after the file's call ids, and every other request refused", async () => {
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" }, "libero", ["act", "finish"], {
		vocabulary: [...UNITS],
		wrist: () => {
			throw new Error("the wrist camera is not up yet");
		},
	});
	p.file.push({
		type: "message",
		message: { role: "assistant", content: [{ type: "toolCall", id: "ft_3", name: "act", arguments: {} }] },
	});
	await assert.rejects(p.emit("session_start"), /the wrist camera is not up yet/);
	for (const [, options] of NOT_TURNS) assertNotATurn(await p.side(options));
	assert.deepEqual(
		calls(await p.turn([])).map((c: any) => c.id),
		["ft_4"],
	);
});

// a later session over a file that holds an earlier episode

/** A units `act` result for the task `task` with the views `views` (a finish's result says its status), and `details`. */
const seenIn = (m: any, task: string, views: string[], details: Record<string, unknown> = {}) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [
			{
				type: "text",
				text:
					c.name === "act"
						? `units: ${c.arguments.unit} x1\nRecent units: none\nTASK: ${task}`
						: c.arguments.status,
			},
			...views.map((data) => ({ type: "image", data, mimeType: "image/png" })),
		],
		details,
		isError: false,
	}));
/** Other camera images than VIEWS, for a later episode. */
const LATER_VIEWS = [pattern(32, 32, 13), pattern(24, 32, 17)].map((v) => png(v).toString("base64"));
/** A transcript's messages as pi's session file holds them, one entry each. */
const fileOf = (messages: unknown[]) =>
	messages.map((message, i) => ({
		type: "message",
		id: `e${i}`,
		parentId: i ? `e${i - 1}` : null,
		timestamp: "",
		message,
	}));

test("a later session over a file that holds an earlier episode (a fork or clone, a resumed or reloaded session, --continue), in a provider of its own as pi builds one and under either model at its start (the policy, the fallback planner over it), goes on from that episode's last call id, and its steps are a fresh session's: each reads its own call's result, not the earlier one's task, images or env success", async () => {
	// The earlier episode: the opening RELEASE, MV_LEFT, after which the env reports success, and finish.
	const earlier: any[] = [];
	const ep = await endpoint(["MV_LEFT"]);
	try {
		const first = fakePi({ "ft-endpoint": ep.url });
		await first.emit("session_start");
		for (const details of [{}, { success: true }, {}]) {
			const m = await first.turn([...earlier]);
			earlier.push(m, ...seenIn(m, "stack the red cube", VIEWS, details));
		}
	} finally {
		await ep.close();
	}
	assert.deepEqual(
		earlier.filter((m) => m.role === "assistant").map((m) => calls(m).map((c: any) => [c.id, c.name])),
		[[["ft_1", "act"]], [["ft_2", "act"]], [["ft_3", "finish"]]],
	);
	/** The later episode in a session whose file holds `file`: its turns and the steps it wrote. */
	const later = async (file: any[], model = "finetuned", reason?: string) => {
		const service = await endpoint(["MV_FWD", "DONE"]);
		try {
			const p = fakePi({ "ft-endpoint": service.url });
			p.ctx.model = { provider: model };
			p.file.push(...file);
			// The same start announced again leaves the planner as it was.
			await p.emit("session_start", reason ? { reason } : {});
			await p.emit("session_start", reason ? { reason } : {});
			const history = file.map((e) => e.message);
			const turns: any[] = [];
			for (let i = 0; i < 3; i++) {
				const m = await p.turn([...history]);
				turns.push(seen(m));
				history.push(m, ...seenIn(m, "put the cube in the bowl", LATER_VIEWS));
			}
			return { turns, steps: written(p) };
		} finally {
			await service.close();
		}
	};
	const ids = (turns: any[]) => turns.map((t) => t[2].map((c: any[]) => c[0]));
	const withoutIds = (turns: any[]) => turns.map((t) => [t[0], t[1], t[2].map((c: any[]) => c.slice(1))]);
	const alone = await later([]);
	assert.deepEqual(
		alone.turns.map((t) => t[2]),
		[
			[["ft_1", "act", { unit: "RELEASE" }]],
			[["ft_2", "act", { unit: "MV_FWD" }]],
			[["ft_3", "finish", { status: "success", summary: "the policy emitted DONE after 1 steps" }]],
		],
	);
	for (const model of ["finetuned", "fallback"])
		for (const reason of ["startup", "reload", "resume", "fork"]) {
			const what = `${model}, ${reason}`;
			const resumed = await later(fileOf(earlier), model, reason);
			assert.deepEqual(ids(resumed.turns), [["ft_4"], ["ft_5"], ["ft_6"]], what);
			assert.deepEqual(withoutIds(resumed.turns), withoutIds(alone.turns), what);
			assert.deepEqual(resumed.steps, alone.steps, what);
		}
});

test("the policy makes no call id past the largest that is counted: over a file at that id its turn fails with the error, and so does the next; one short of it the opening makes that id, and the next step fails before the service is asked or a step is written", async () => {
	/** A file whose one call is ft_<n>. */
	const at = (n: number) => ({
		type: "message",
		id: "e",
		parentId: null,
		timestamp: "",
		message: { role: "assistant", content: [{ type: "toolCall", id: `ft_${n}`, name: "act", arguments: {} }] },
	});
	const ran = `the session's call ids reached ft_${LARGEST}; start a new session`;
	const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
	p.file.push(at(LARGEST));
	await p.emit("session_start");
	const m = await p.turn([]);
	assert.equal(m.stopReason, "error");
	assert.equal(m.errorMessage, ran);
	assert.deepEqual(seen(await p.turn([m])), ["error", ran, []]);
	// The service would answer with a move, or with DONE (a finish).
	for (const reply of ["MV_FWD", "DONE"]) {
		const ep = await endpoint([reply]);
		try {
			const q = fakePi({ "ft-endpoint": ep.url });
			q.file.push(at(LARGEST - 1));
			await q.emit("session_start");
			const opening = await q.turn([]);
			assert.deepEqual(seen(opening)[2], [[`ft_${LARGEST}`, "act", { unit: "RELEASE" }]], reply);
			assert.deepEqual(seen(await q.turn([opening, ...seenBy(opening)])), ["error", ran, []], reply);
			assert.deepEqual([ep.bodies.length, written(q)], [0, []], reply);
		} finally {
			await ep.close();
		}
	}
});

test("a session that went back on its tree (/tree) to before an earlier episode, or compacted it out of the context, and started again (a reload) goes on from that episode's last call id: an id the file holds names no new call", async () => {
	const prompt = { role: "user", content: "Solve the task.", timestamp: 0 };
	// The earlier episode's calls and results, as the file holds them after the prompt.
	const earlier = [1, 2, 3, 4].flatMap((n) => {
		const m = { role: "assistant", content: [{ type: "toolCall", id: `ft_${n}`, name: "act", arguments: {} }] };
		return [m, ...seenIn(m, "stack the red cube", VIEWS)];
	});
	const summary = {
		type: "compaction",
		id: "c",
		parentId: null,
		timestamp: "",
		summary: "an episode",
		tokensBefore: 9,
	};
	for (const [what, file, context] of [
		// The leaf back at the prompt: the context holds the prompt alone.
		["a branch the session left", fileOf([prompt, ...earlier]), [prompt]],
		// A compaction's summary in place of the episode: the context holds the summary (as a user message).
		[
			"entries a compaction keeps out of the context",
			[...fileOf([prompt, ...earlier]), summary],
			[{ role: "user", content: "an episode", timestamp: 0 }],
		],
	] as const) {
		const p = fakePi({ "ft-endpoint": "http://127.0.0.1:9/v1" });
		p.file.push(...file);
		await p.emit("session_start", { reason: "reload" });
		assert.deepEqual(
			calls(await p.turn([...context])).map((c: any) => c.id),
			["ft_5"],
			what,
		);
	}
});
