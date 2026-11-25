/**
 * HumanCLAW: the action mapping, clamps and prompts against golden texts that HumanCLAW's own Python
 * produced (services/tests/humanclaw_golden.py), the units round trip, and the PSV planner's
 * retries, verifier replacement and history on scripted replies.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
	isContextOverflow,
	isRetryableAssistantError,
	normalizeContext,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { RELAY_ERRORS } from "../src/planner/fallback.ts";
import {
	chooserAction,
	pyFixed,
	pyRound,
	skillCall,
	skillToText,
	toJson,
	UNITS,
	unitOf,
} from "../src/robots/humanclaw/actions.ts";
import { mountPsv, psvBase } from "../src/robots/humanclaw/provider.ts";
import { historyItem, PsvPlanner, parseJsonLoose } from "../src/robots/humanclaw/psv.ts";
import { within } from "./helpers/pi-session.ts";

type Json = Record<string, any>;
const golden = JSON.parse(readFileSync(new URL("./fixtures/humanclaw-golden.json", import.meta.url), "utf8")) as {
	chooser: Json[];
	episodes: Json[];
};

test("chooserAction reproduces _chooser_action on every golden case (clamps, rounding, recovery)", () => {
	for (const c of golden.chooser) {
		const a = chooserAction(c.input);
		assert.deepEqual(toJson(a), c.action, JSON.stringify(c.input));
		assert.equal(skillToText(a), c.text);
	}
});

test("Python rounding: round-half-even and %.2f on exact binary ties", () => {
	assert.equal(pyRound(36.5), 36);
	assert.equal(pyRound(37.5), 38);
	assert.equal(pyFixed(0.625, 2), "0.62");
	assert.equal(pyFixed(0.375, 2), "0.38");
	assert.equal(pyFixed(0.3, 2), "0.30");
});

test("a unit and its parameter run the SkillCall HumanCLAW would run, and back", () => {
	for (const c of golden.chooser) {
		const a = chooserAction(c.input);
		const { unit, param } = unitOf(a);
		assert.ok(UNITS.some((u) => u.name === unit));
		assert.deepEqual(toJson(skillCall(unit, param)), toJson(a), JSON.stringify(c.input));
	}
	assert.deepEqual(skillCall("TURN_RIGHT", 45).cond, -45);
	assert.deepEqual(skillCall("SIDE_LEFT", 0.25).cond, 0.25);
	assert.deepEqual(skillCall("STEP_BACK", 0.3).cond, [0, -0.3]);
	assert.equal(skillCall("STOP", undefined).skill, "stand");
	assert.ok(UNITS.find((u) => u.name === "STOP")?.terminal);
});

for (const ep of golden.episodes)
	test(`PSV planner replays golden episode ${ep.name}: prompts, retries, verifier, history`, async () => {
		const replies: unknown[] = ep.replies.flat();
		const prompts: string[] = [];
		const planner = new PsvPlanner(
			async (prompt) => {
				prompts.push(prompt);
				const r = replies.shift();
				if (r && typeof r === "object") throw new Error((r as Json).raise);
				return { text: r as string, usage: { prompt_tokens: 1, completion_tokens: 1 } };
			},
			async () => {},
		);
		planner.reset(ep.instruction);
		const history: Json[] = [];
		for (const [i, s] of (ep.steps as Json[]).entries()) {
			const before = prompts.length;
			const d = await planner.act(history);
			assert.deepEqual(prompts.slice(before), s.prompts, `step ${i} prompts`);
			assert.deepEqual(toJson(d.action), s.action, `step ${i} action`);
			assert.equal(skillToText(d.action), s.action_text);
			assert.deepEqual(d.raw_plan, s.raw_plan, `step ${i} raw_plan`);
			assert.deepEqual(d.verifier, s.verifier, `step ${i} verifier`);
			assert.deepEqual(planner.state(), s.planner_state, `step ${i} state`);
			assert.deepEqual(
				d.stages.map((x) => x.raw),
				s.stages.map((x: Json) => x.raw),
			);
			const item = historyItem(i, d);
			assert.deepEqual(item, s.history_item, `step ${i} history`);
			history.push(item);
		}
		assert.equal(replies.length, 0);
	});

test("parseJsonLoose takes fenced, prefixed and <think> replies and refuses prose", () => {
	assert.deepEqual(parseJsonLoose('<think>x</think>```json\n{"a": 1}\n```'), { a: 1 });
	assert.deepEqual(parseJsonLoose('Sure: {"a": {"b": "}"}} trailing'), { a: { b: "}" } });
	assert.throws(() => parseJsonLoose("no json"), /Could not parse JSON/);
});

test("psvBase reads --model humanclaw-psv/<base> from argv only", () => {
	assert.equal(psvBase(["pi", "--model", "humanclaw-psv/selfhost/muse"]), "selfhost/muse");
	assert.equal(psvBase(["pi", "--model=humanclaw-psv/a/b"]), "a/b");
	assert.equal(psvBase(["pi", "--model", "selfhost/muse"]), undefined);
});

import humanclaw from "../src/robots/humanclaw/index.ts";
import { fakeEnv, rgb, stubPi } from "./sim-stub.ts";

test("pi mode: act records target_visible and runs the unit's SkillCall; STOP ends and finishes the rollout", async (t) => {
	let step = 0;
	const obsOf = (done = false, stopped = false) => ({
		ego: rgb(4, 4),
		instruction: "Find the bed.",
		step,
		max_steps: 100,
		done,
		stopped,
		episode: {
			scene_id: "s",
			episode_id: "0",
			object_category: "bed",
			rollout: 0,
			key: "s_ep0_bed",
			output_dir: "/x",
		},
	});
	const env = await fakeEnv((c) => {
		if (c.method === "env.reset") return obsOf();
		if (c.method === "env.step") {
			step++;
			const stop = c.kwargs.skill === "stand";
			return { ...obsOf(stop, stop), collision: {} };
		}
		if (c.method === "env.finish")
			return {
				metrics: { nav_sr_20cm: true, find_sr: true },
				steps: step,
				active_stop: true,
				videos: [],
				metrics_path: "/x/metrics.json",
			};
		return undefined;
	});
	const s = stubPi({ "env-url": env.url, units: "both", "humanclaw-mode": "pi" });
	t.after(() => env.close());
	humanclaw(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active().sort(), ["act", "finish", "look", "plan"]);
	const look = await s.run("look", {});
	assert.equal(look.details.instruction, "Find the bed.");
	await s.run("act", { unit: "TURN_RIGHT", param: 200, target_visible: true });
	const rec = env.calls.find((c) => c.method === "env.record_decision")!;
	assert.equal(rec.kwargs.decision.planner_skill.visible_state, "The target is visible.");
	const st = env.calls.find((c) => c.method === "env.step")!;
	assert.deepEqual([st.kwargs.skill, st.kwargs.cond, st.kwargs.action_name], ["turn", -120, "Turn<right><120>"]);
	const r = await s.run("act", { unit: "STOP" });
	assert.equal(r.details.done, true);
	assert.ok(env.calls.some((c) => c.method === "env.finish"));
	const after = await s.run("act", { unit: "WALK", param: "fast" });
	assert.match(after.content[0].text, /not run/);
	assert.equal(env.calls.filter((c) => c.method === "env.step").length, 2);
});

test("paper mode needs humanclaw-psv and --units=both; pi-mode modules are refused", async () => {
	const s = stubPi({ units: "true", "humanclaw-mode": "pi", "env-url": "http://127.0.0.1:1" });
	humanclaw(s.pi);
	await s.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(s.active(), []);
	const p = stubPi({ units: "both", "humanclaw-mode": "paper", "env-url": "http://127.0.0.1:1" });
	humanclaw(p.pi);
	await p.emit("session_start");
	process.exitCode = undefined;
	assert.deepEqual(p.active(), []);
});

test("humanclaw-psv turns: look first, then one act per decision from the base model's JSON, finish when done", async () => {
	const { mountPsv, PSV_ENTRY } = await import("../src/robots/humanclaw/provider.ts");
	const s = stubPi();
	const emitted: unknown[] = [];
	(s.pi as any).events = { emit: (_: string, d: unknown) => emitted.push(d), on: () => () => {} };
	const replies = [
		"not json",
		JSON.stringify({
			visible_state: "The bed is visible.",
			mid_level_goal: "Go",
			action_id: 2,
			action_name: "Turn<left><45>",
		}),
	];
	const asked: any[] = [];
	const psv = mountPsv(s.pi, "selfhost/muse", async (content) => {
		asked.push(content);
		return { text: replies.shift() ?? "{}" };
	});
	await s.emit("session_start");
	const turn = async (messages: any[]) => {
		const model = { id: "selfhost/muse", api: "humanclaw-psv", provider: "humanclaw-psv" } as any;
		// As pi's agent loop asks: with the session's id (sim-stub's).
		const stream = psv.streamSimple(model, { messages } as any, { sessionId: "sess" });
		let done: any;
		for await (const ev of stream) if (ev.type === "done") done = ev.message;
		return done.content.filter((c: any) => c.type === "toolCall");
	};
	const [look] = await turn([]);
	assert.equal(look.name, "look");
	const img = { type: "image", data: "AAAA", mimeType: "image/png" };
	const result = (id: string, details: any) => ({
		role: "toolResult",
		toolCallId: id,
		content: [img],
		details,
		isError: false,
	});
	const [act] = await turn([result(look.id, { instruction: "Find the bed.", step: 0 })]);
	assert.deepEqual([act.name, act.arguments], ["act", { unit: "TURN_LEFT", param: 45 }]);
	assert.equal(asked.length, 2, "the unparsable reply is asked again at the same state");
	assert.deepEqual(asked[0][1], img, "prompt text first, then the ego image");
	assert.equal((emitted[0] as any).planner_skill.visible_state, "The bed is visible.");
	assert.equal(s.entries.filter((e) => e.type === PSV_ENTRY).length, 1);
	const [fin] = await turn([result(act.id, { done: true, stopped: true, step: 1 })]);
	assert.deepEqual([fin.name, fin.arguments.status], ["finish", "success"]);
});

// ---------------------------------------------------------------------------
// humanclaw-psv and a request that is not a turn of the session's own loop (the dashboard's model check, a question to
// a VLM, a check that asks the session's model, pi's own summaries)

/** The stub's first session id. pi sends a session's id with every request of that session's own loop. */
const SESSION = "session-1";
/** What a request to humanclaw-psv gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"humanclaw-psv plans the episode's steps and answers no other request; --aux-model names a model for the checks that take one";
const NO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
/** `m` is that answer: an error of humanclaw-psv/<base>, with no content and no usage. */
function assertNotATurn(m: any) {
	assert.equal(m.stopReason, "error");
	assert.equal(m.errorMessage, NOT_A_TURN);
	assert.deepEqual(m.content, []);
	assert.deepEqual([m.api, m.provider, m.model], ["humanclaw-psv", "humanclaw-psv", "selfhost/muse"]);
	assert.deepEqual(m.usage, { ...NO_COST, totalTokens: 0, cost: { ...NO_COST, total: 0 } });
}
/**
 * What the planner takes for no turn of its session: a request without an id, under another id, or with pi's mark on
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
/** A user message to ask with, as the dashboard's model check asks. */
const CHECK = { role: "user", content: "Reply with the single word OK.", timestamp: 0 };
/** A context of a request's own, as a tool's agent loop or pi's summaries send one: a system prompt and tools. */
const OWN = normalizeContext({
	systemPrompt: "You are a policy.",
	tools: [{ name: "policy_act", description: "act", parameters: { type: "object", properties: {} } as never }],
	messages: [CHECK as never],
});

/** A base-model reply held back: `arrived` once the planner has asked; it is answered when the test calls `release`. */
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
type Reply = string | { held: Hold; reply: string };

/**
 * humanclaw-psv/selfhost/muse on a stub pi whose session's branch holds the entries the provider writes (a later
 * session resumes from them). The base model answers with `replies` in turn, through the `ask` hook, or through
 * `registry` when one is given (as pi's model registry streams a reply). `asked` holds what each base-model request
 * carried, `decisions` what the provider sent the robot.
 */
function psvPi(replies: Reply[], o: { branch?: any[]; registry?: (asked: Json[]) => unknown } = {}) {
	const handlers = new Map<string, ((event: any, ctx: any) => unknown)[]>();
	const branch = o.branch ?? [];
	const entries: { type: string; data: any }[] = [];
	const decisions: unknown[] = [];
	const asked: Json[] = [];
	/** The id of the session the stub is in: a test moves it on for a new session. */
	const session = { id: SESSION };
	const pi = {
		on: (name: string, fn: (event: any, ctx: any) => unknown) =>
			handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerProvider: () => {},
		getFlag: () => undefined,
		appendEntry: (type: string, data: unknown) => {
			entries.push({ type, data });
			branch.push({ type: "custom", customType: type, data });
		},
		events: { emit: (_channel: string, data: unknown) => decisions.push(data), on: () => () => {} },
	} as unknown as ExtensionAPI;
	const ctx = {
		modelRegistry: o.registry?.(asked),
		sessionManager: { getBranch: () => branch, getSessionId: () => session.id },
	};
	const psv = mountPsv(
		pi,
		"selfhost/muse",
		o.registry
			? undefined
			: async (content) => {
					asked.push(content);
					let r = replies.shift() ?? "{}";
					if (typeof r !== "string") {
						r.held.reached();
						await r.held.released;
						r = r.reply;
					}
					return { text: r };
				},
	);
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name, ...event }, ctx);
	};
	const model = { id: "selfhost/muse", api: "humanclaw-psv", provider: "humanclaw-psv" } as any;
	/** One decision as pi's agent loop asks for it: with the session's id. It is awaited ten seconds at most. */
	const turn = (messages: unknown[]) =>
		within(
			10_000,
			psv.streamSimple(model, { messages } as any, { sessionId: session.id }).result(),
			"the session's own turn",
		);
	/**
	 * A request that is not the loop's, as ctx.modelRegistry.streamSimple hands one over (the dashboard's model check,
	 * a question to a VLM): one user message unless a transcript or a whole context is given, and no session id
	 * unless `options` holds one.
	 */
	const ask = (options: Record<string, unknown> = {}, context?: unknown[] | object) =>
		psv.streamSimple(
			model,
			(Array.isArray(context) ? { messages: context } : (context ?? { messages: [CHECK] })) as any,
			{ maxTokens: 1, ...options } as any,
		);
	/** The answer to such a request, awaited two seconds at most (one taken for a turn may wait on the base model). */
	const side = (options: Record<string, unknown> = {}, context?: unknown[] | object) =>
		within(2000, ask(options, context).result(), "a request that is not the session's turn");
	/** The answer to one from a caller that passes no options at all. */
	const bare = () =>
		within(2000, psv.streamSimple(model, { messages: [] } as any).result(), "a request without options");
	return { emit, turn, ask, side, bare, session, branch, entries, decisions, asked, psv };
}
type Psv = ReturnType<typeof psvPi>;

/**
 * A request of this kind gets the error with each transcript a check sends: one user message, the loop's own as it is
 * at this place, and a context of its own; and with a signal that is live when it arrives, as the dashboard's model
 * check sends one (a request that arrives aborted keeps its own).
 */
async function refused(p: Psv, options: Record<string, unknown>, transcript: unknown[]) {
	for (const context of [undefined, transcript, OWN]) assertNotATurn(await p.side(options, context));
	assertNotATurn(await p.side({ signal: new AbortController().signal, ...options }));
}

const plan = (action_id: number, action_name: string, goal = "Find the bed") =>
	JSON.stringify({ visible_state: "A hallway.", mid_level_goal: goal, action_id, action_name });
const TURN = plan(2, "Turn<left><45>");
/** Walk forward: the planner, then the verifier, asked at the same observation. */
const WALK = plan(0, "Walk<forward><slow>");
/** The verifier's answer: the proposed action stands. */
const ACCEPT = JSON.stringify({ verdict: "accept", reason: "the lane is clear" });
/** Stop/Stand, which the verifier is asked about too. */
const STOP = plan(1, "Stop/Stand", "Stop/Stand");
/** The ego image of the n-th observation: each decision's requests show which one they carry. */
const ego = (n: number) => ({ type: "image", data: `EGO${n}`, mimeType: "image/png" });
/** The robot's result of each call of `m`, the n-th observation: the episode is done at `done`. */
const observed = (m: any, n: number, done: number) =>
	m.content
		.filter((c: any) => c.type === "toolCall")
		.map((c: any) => ({
			role: "toolResult",
			toolCallId: c.id,
			toolName: c.name,
			content: [{ type: "text", text: `step ${n}` }, ego(n)],
			details: n >= done ? { done: true, stopped: true, step: n } : { instruction: "Find the bed.", step: n },
			isError: false,
		}));
/** A turn as the loop got it: its stop reason, its text (or its error) and its calls. */
const seen = (m: any) => [
	m.stopReason,
	m.errorMessage ??
		m.content
			.filter((c: any) => c.type === "text")
			.map((c: any) => c.text)
			.join("\n"),
	m.content.filter((c: any) => c.type === "toolCall").map((c: any) => [c.id, c.name, c.arguments]),
];
type At = (place: string, transcript: unknown[], p: Psv) => Promise<void>;

/**
 * One episode as pi's agent loop runs it: the calls of each turn get the next observation, done at the `done`-th,
 * until a turn calls nothing; then the loop asks once more, as for a second prompt. `at` runs at each place where a
 * request from elsewhere can arrive, with the transcript the loop has there; while `held` holds back a base-model
 * reply, `at` runs inside the loop's request that waits for it. Returns each turn as the loop got it.
 */
async function decide(p: Psv, done: number, at?: At, held?: Hold) {
	const visit = async (place: string, transcript: unknown[]) => at?.(place, [...transcript], p);
	const turns: unknown[] = [];
	const history: unknown[] = [];
	await visit("before the first decision", history);
	for (let n = 0; n < 20; n++) {
		const asked = p.turn([...history]);
		const h = held;
		if (h && (await Promise.race([asked.then(() => false), h.arrived.then(() => true)]))) {
			held = undefined;
			try {
				await visit("while the base model answers a decision", history);
			} finally {
				h.release();
			}
		}
		const m = await asked;
		turns.push(seen(m));
		if (!m.content.some((c: any) => c.type === "toolCall")) break;
		await visit("while a call runs", [...history, m]);
		history.push(m, ...observed(m, n, done));
		await visit("between two decisions", history);
	}
	await visit("after the end", history);
	turns.push(seen(await p.turn([...history])));
	return turns;
}
/** What the planner holds and wrote, and what it sent: its history, its state, its entries, decisions and requests. */
const held = (p: Psv) => ({
	history: p.psv.history(),
	planner: [p.psv.planner.instruction, p.psv.planner.state()],
	entries: p.entries,
	decisions: p.decisions,
	asked: p.asked,
});

/**
 * look, a turn, a walk (the planner's request held back, then the verifier at the same observation), Stop/Stand (and
 * its verifier), finish.
 */
async function episode(at?: At) {
	const h = hold();
	const p = psvPi([TURN, { held: h, reply: WALK }, ACCEPT, STOP, ACCEPT]);
	try {
		await p.emit("session_start");
		return { turns: await decide(p, 3, at, h), ...held(p) };
	} finally {
		h.release();
	}
}
/**
 * A later session on the branch of one that made two decisions (a turn and a walk): it starts with look again, then
 * a turn and Stop/Stand, its planner going on from the state and history the first session wrote.
 */
async function resumed(at?: At) {
	const first = psvPi([TURN, WALK, ACCEPT]);
	await first.emit("session_start");
	const history: unknown[] = [];
	for (let n = 0; n < 3; n++) {
		const m = await first.turn([...history]);
		history.push(m, ...observed(m, n, 9));
	}
	const p = psvPi([TURN, STOP, ACCEPT], { branch: first.branch });
	await p.emit("session_start");
	return { turns: await decide(p, 2, at), ...held(p) };
}
/**
 * The base model through pi's model registry (an OpenAI-compatible server): it rejects the JSON response format at
 * the second decision, which is asked again as text, as is every request after it.
 */
async function negotiated(at?: At) {
	const replies = [TURN, "", TURN, STOP, ACCEPT];
	const p = psvPi([], {
		registry: (asked) => ({
			find: (provider: string, id: string) => ({ provider, id, api: "openai-completions" }),
			async *streamSimple(model: unknown, context: { messages: { content: Json[] }[] }, opts: SimpleStreamOptions) {
				const payload = await opts.onPayload?.({}, model as never);
				asked.push({ payload, content: context.messages[0].content });
				const text = replies.shift() ?? "{}";
				if (!text) {
					yield { type: "error", error: { errorMessage: '400: {"code":"response_format_not_supported"}' } };
					return;
				}
				yield { type: "done", message: { content: [{ type: "text", text }], usage: { input: 1, output: 1 } } };
			},
		}),
	});
	await p.emit("session_start");
	return { turns: await decide(p, 3, at), ...held(p) };
}
/** Each episode without a request from elsewhere, run once for all the tests that compare with it. */
const alone = new Map<string, ReturnType<typeof episode>>();
const without = (run: typeof episode) => alone.get(run.name) ?? alone.set(run.name, run()).get(run.name);

for (const [kind, options] of NOT_TURNS)
	test(`humanclaw-psv: a request that is not the session's turn (${kind}) gets the error and leaves the planner as it is: before the first decision, while a call runs, between two decisions, while the base model answers a decision and after the end, the session's own turns, the history, the planner's state, the entries, the decisions sent to the robot and the base model's requests are those of an episode without it (an ego image asked twice, a resumed session, the JSON format turned down)`, async () => {
		for (const run of [episode, resumed, negotiated]) {
			const places: string[] = [];
			const got = await run(async (place, transcript, p) => {
				places.push(place);
				await refused(p, options, transcript);
			});
			const expected = [
				"before the first decision",
				"while a call runs",
				"between two decisions",
				...(run === episode ? ["while the base model answers a decision"] : []),
				"after the end",
			];
			assert.deepEqual([...new Set(places)], expected, run.name);
			assert.deepEqual(got, await without(run), run.name);
		}
	});

test("humanclaw-psv: before a session starts no request is a session's turn, of whatever kind, with or without options; after it starts, one that is not its turn does not take the opening look", async () => {
	const p = psvPi([TURN]);
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
	// The session's first turn is the opening look, under the first call id; the base model was not asked.
	assert.deepEqual(seen(await p.turn([])), ["toolUse", "look (the first ego image)", [["psv_1", "look", {}]]]);
	assert.deepEqual([p.asked, p.entries, p.decisions], [[], [], []]);
});

test("humanclaw-psv: a request of the session's own loop is its turn whatever retention it names other than pi's one-off mark, and whether its signal is aborted; a new session's turns are those under its own id", async () => {
	const p = psvPi([TURN, TURN]);
	await p.emit("session_start");
	const look = await p.side({ sessionId: SESSION, cacheRetention: "short" }, []);
	const history = [look, ...observed(look, 0, 9)];
	// The planner has no abort branch: a turn of the loop that arrives aborted is taken as any other turn (this base
	// model does not read the signal).
	const act = await p.side({ sessionId: SESSION, cacheRetention: "long", signal: AbortSignal.abort() }, history);
	history.push(act, ...observed(act, 1, 9));
	const again = await p.side({ sessionId: SESSION, cacheRetention: "long" }, history);
	assert.deepEqual(
		[look, act, again].map((m) => m.stopReason),
		["toolUse", "toolUse", "toolUse"],
		"a turn under short retention, one under long arriving aborted, one under long",
	);
	assert.deepEqual(seen(look)[2], [["psv_1", "look", {}]]);
	assert.deepEqual(seen(act)[2], [["psv_2", "act", { unit: "TURN_LEFT", param: 45 }]]);
	assert.deepEqual(seen(again)[2], [["psv_3", "act", { unit: "TURN_LEFT", param: 45 }]]);
	// A new session has an id of its own: a request under the one before is not its turn.
	p.session.id = "session-2";
	await p.emit("session_start");
	assertNotATurn(await p.side({ sessionId: SESSION }, [...history, again, ...observed(again, 2, 9)]));
	const opened = await p.turn([]);
	assert.deepEqual(
		opened.content.filter((c: any) => c.type === "toolCall").map((c: any) => c.name),
		["look"],
	);
	assert.equal(p.asked.length, 2);
});

test("humanclaw-psv: an id that only resembles the session's is another id, and a one-off summary is told by its mark alone", async () => {
	const p = psvPi([TURN]);
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
	assert.deepEqual(seen(await p.turn([]))[2], [["psv_1", "look", {}]]);
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

test("humanclaw-psv: whatever else a request carries (a reasoning level, a temperature, metadata, headers, a payload hook, a token limit, a retention other than pi's mark), the id and the mark decide: without the id, under another id or with the mark it gets the error, and the session's own turn is answered", async () => {
	const p = psvPi([TURN]);
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
	// The session's first decision, asked with all of it and a retention: the opening look, under the first call id.
	const own = { ...CARRIED, cacheRetention: "long", sessionId: SESSION };
	const look = within(2000, p.ask(own, []).result(), "the session's own turn");
	assert.deepEqual(seen(await look), ["toolUse", "look (the first ego image)", [["psv_1", "look", {}]]]);
	assert.deepEqual([p.asked, p.entries, p.decisions], [[], [], []]);
});

test("humanclaw-psv: the answer to a request that is not the session's turn reaches its asker as one error event; it is no error pi sends again, and none the fallback planner by default changes models for", async () => {
	const p = psvPi([TURN]);
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
	const m = events[0].error;
	// pi sends a request again after an error it takes for a passing one, and after one it takes for a full context.
	assert.equal(isRetryableAssistantError(m), false);
	assert.equal(isContextOverflow(m, 100_000_000), false);
	assert.doesNotMatch(m.errorMessage, RELAY_ERRORS);
});

test("humanclaw-psv: the session's id is kept at every start pi announces (its first start, a new session, a resumed one, a fork, a reload), each in a planner of its own as pi builds one, and still when a run ends: that session's own request is its turn, and one without an id is not", async () => {
	for (const reason of ["startup", "new", "resume", "fork", "reload"]) {
		const p = psvPi([TURN]);
		p.session.id = `session-${reason}`;
		await p.emit("session_start", { reason });
		// pi ends each run with agent_end; the session's next prompt is still its own.
		await p.emit("agent_end");
		assertNotATurn(await p.side());
		const own = seen(await p.turn([]))[2] as [string, string, unknown][];
		assert.deepEqual(
			own.map(([, name, args]) => [name, args]),
			[["look", {}]],
			reason,
		);
	}
});

test("humanclaw-psv: the session's id is kept ahead of the rest of the start: when the rest throws (here reading the branch), the session's own turns are still answered and every other request refused", async () => {
	const unreadable = Object.assign([], {
		filter: () => {
			throw new Error("the branch could not be read");
		},
	});
	const p = psvPi([TURN], { branch: unreadable });
	await assert.rejects(p.emit("session_start"), /the branch could not be read/);
	for (const [, options] of NOT_TURNS) assertNotATurn(await p.side(options));
	assert.deepEqual(seen(await p.turn([]))[2], [["psv_1", "look", {}]]);
});
