import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { isContextOverflow, isRetryableAssistantError } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import replay, { loadRecording, REPLAY_ENTRY, taskDiff } from "../src/capabilities/replay/index.ts";
import { RELAY_ERRORS } from "../src/planner/fallback.ts";
import { strayArguments, strayRefusal } from "../src/primitives/arguments.ts";
import { deployFlags } from "./helpers/deployment.ts";

type Handler = (event: any, ctx: any) => unknown;

/** A fresh folder for a recorded session, removed when this file's tests are done. */
const folders: string[] = [];
const folder = () => {
	const dir = mkdtempSync(join(tmpdir(), "replay-"));
	folders.push(dir);
	return dir;
};
after(() => {
	for (const dir of folders) rmSync(dir, { recursive: true, force: true });
});

const assistant = (id: string, parentId: string, content: unknown[], stopReason = "toolUse") => ({
	type: "message",
	id,
	parentId,
	message: { role: "assistant", content, stopReason },
});
let callIds = 0;
const call = (name: string, args: Record<string, unknown>) => ({
	type: "toolCall",
	id: `rec_${name}_${++callIds}`,
	name,
	arguments: args,
});
/** A turn and one result entry per call, chained. */
function turnWithResults(id: string, parentId: string, content: unknown[], errors: Record<string, string> = {}) {
	const a = assistant(id, parentId, content);
	const calls = content.filter((c: any) => c.type === "toolCall") as { id: string; name: string }[];
	const rs = calls.map((c, i) => ({
		type: "message",
		id: `${id}_r${i}`,
		parentId: i ? `${id}_r${i - 1}` : id,
		message: {
			role: "toolResult",
			toolCallId: c.id,
			toolName: c.name,
			content: [{ type: "text", text: errors[c.name] ?? "{}" }],
			isError: c.name in errors,
		},
	}));
	return { entries: [a, ...rs], leaf: rs.at(-1)?.id ?? id };
}
const STALE =
	"The operator took over and executed 2 step(s): FORWARD GRASP. This move call was decided on an older observation and was not executed; decide again from the current observation.";

/**
 * A recorded session: a task, an abandoned branch, an errored reply, a multi-call turn, a move the
 * operator's takeover dropped as stale, a turn refused by the robot base next to a grip that
 * really failed, a write, and (optionally) finish.
 */
function recordSession(finish = true) {
	const dir = folder();
	const t1 = turnWithResults("a1", "u", [{ type: "thinking", thinking: "look first" }, call("view", {})]);
	const t2 = turnWithResults("a2", "err", [
		{ type: "text", text: "two at once" },
		call("move", { xyz: [1, 2, 3] }),
		call("grip", { v: 1 }),
	]);
	const t3 = turnWithResults("a3", t2.leaf, [call("move", { xyz: [5, 5, 5] })], { move: STALE });
	const t4 = turnWithResults("a4", t3.leaf, [call("move", { xyz: [6, 6, 6] }), call("grip", { v: 0 })], {
		move: "Planner turns budget exhausted; the episode is over.",
		grip: "gripper jammed",
	});
	const t5 = turnWithResults("a5", t4.leaf, [call("write", { path: "/tmp/x", content: "y" })]);
	const lines: unknown[] = [
		{ type: "session", version: 3, id: "s", cwd: dir },
		{
			type: "custom",
			id: "t",
			parentId: null,
			customType: "robot_task",
			data: { robot: "arm", task: "3", seed: "1" },
		},
		{ type: "message", id: "u", parentId: "t", message: { role: "user", content: "Solve." } },
		...t1.entries,
		assistant("dead", t1.leaf, [call("move", { xyz: [9, 9, 9] })]), // abandoned: the leaf's branch skips it
		assistant("err", t1.leaf, [call("move", { xyz: [8, 8, 8] })], "error"),
		...t2.entries,
		...t3.entries,
		...t4.entries,
		...t5.entries,
		...turnWithResults(
			"a6",
			t5.leaf,
			finish ? [call("finish", { status: "success", summary: "ok" })] : [call("view", {})],
		).entries,
	];
	writeFileSync(join(dir, "2026-01-01T00-00-00-000Z_s.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	writeFileSync(join(dir, "arm_recipe.jsonl"), `${JSON.stringify({ action: "view" })}\n`);
	return dir;
}

/** The stub's first session id. pi sends a session's id with every request of that session's own loop. */
const SESSION = "session-1";

/** `gate`: another extension's tool_call handler, loaded before the replay (the robot's gate). */
function fakePi(flags: Record<string, unknown>, task: Record<string, string>, gate?: Handler) {
	flags = deployFlags(flags);
	const handlers = new Map<string, Handler[]>();
	const entries: { type: string; data: any }[] = [];
	const stderr: string[] = [];
	/** The id of the session the stub is in: a test moves it on for a new session. */
	const session = { id: SESSION };
	let provider: any;
	const pi = {
		on: (name: string, fn: Handler) => handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerFlag: () => {},
		getFlag: (name: string) => flags[name],
		registerProvider: (p: any) => {
			provider = p;
		},
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
	} as unknown as ExtensionAPI;
	const ctx = {
		hasUI: false,
		cwd: "/",
		model: { provider: "replay", id: "session" },
		ui: { notify: () => {} },
		sessionManager: {
			getBranch: () => [{ type: "custom", customType: "robot_task", data: task }],
			getSessionId: () => session.id,
		},
	};
	/** Run `name`'s handlers; the first defined handler result. */
	const emit = async (name: string, event: Record<string, unknown> = {}) => {
		let out: unknown;
		for (const fn of handlers.get(name) ?? []) out ??= await fn({ type: name, ...event }, ctx);
		return out as { block?: boolean; reason?: string } | undefined;
	};
	const log = console.error;
	console.error = (line: string) => stderr.push(line);
	if (gate) handlers.set("tool_call", [gate]);
	replay(pi);
	const model = provider.getModels()[0];
	/** One model turn over `messages`, as pi's agent loop asks for it: with the session's id. */
	const turn = (messages: unknown[] = [], signal?: AbortSignal) =>
		provider.streamSimple(model, { messages }, { signal, sessionId: session.id }).result();
	/**
	 * A request that is not the loop's, as ctx.modelRegistry.streamSimple hands one over (the dashboard's model check,
	 * a question to a VLM): one user message unless `messages` is given, and no session id unless `options` holds one.
	 */
	const ask = (options: Record<string, unknown> = {}, messages?: unknown[]) =>
		provider.streamSimple(
			model,
			{ messages: messages ?? [{ role: "user", content: "Reply with the single word OK.", timestamp: 0 }] },
			{ maxTokens: 1, ...options },
		);
	/** The answer to such a request. */
	const side = (options: Record<string, unknown> = {}, messages?: unknown[]) => ask(options, messages).result();
	/** The answer to one from a caller that passes no options at all. */
	const bare = () => provider.streamSimple(model, { messages: [] }).result();
	return {
		emit,
		turn,
		ask,
		side,
		bare,
		session,
		entries,
		stderr,
		model,
		restore: () => {
			console.error = log;
		},
	};
}

const calls = (m: any) => m.content.filter((c: any) => c.type === "toolCall");
const text = (m: any) =>
	m.content
		.filter((c: any) => c.type === "text")
		.map((c: any) => c.text)
		.join("\n");
/** This run's results of turn `m`'s calls; `errors` maps a call name to its error text. */
const results = (m: any, errors: Record<string, string> = {}) =>
	calls(m).map((c: any) => ({
		role: "toolResult",
		toolCallId: c.id,
		toolName: c.name,
		content: [{ type: "text", text: errors[c.name] ?? "{}" }],
		isError: c.name in errors,
	}));

test("loadRecording follows the leaf's branch, drops errored replies and calls that never ran, and picks the session file in a dir", () => {
	const rec = loadRecording(recordSession());
	assert.match(rec.path, /2026-01-01T00-00-00-000Z_s\.jsonl$/);
	assert.deepEqual(rec.task, { robot: "arm", task: "3", seed: "1" });
	assert.deepEqual(
		rec.turns.map((t) => t.calls.map((c) => c.name)),
		[["view"], [], ["move", "grip"], [], ["grip"], ["write"], ["finish"]],
	);
	assert.deepEqual(
		rec.turns.map((t) => t.blocked.map((b) => b.name)),
		[[], ["move"], [], ["move"], ["move"], [], []],
	);
	assert.match(rec.turns[1].blocked[0].reason, /stopReason "error"/);
	assert.equal(rec.turns[4].calls[0].error, "gripper jammed");
	assert.equal(rec.turns[2].calls[0].error, undefined);
	assert.equal(rec.turns[0].thinking, "look first");
	assert.equal(rec.finished, true);
	assert.deepEqual(taskDiff({ robot: "arm", seed: "1" }, { robot: "arm", seed: 2 }), ["seed: 1 -> 2"]);
});

test("the memory guard's refusal of a grep or find that is not its own is a call that never ran", () => {
	const rec = loadRecording(
		sessionOf([
			{
				content: [call("grep", { pattern: "x", path: "." }), call("view", {})],
				errors: {
					grep: "file access is disabled: the session's grep is not the memory guard's, which reads no ignore files",
				},
			},
		]),
	);
	assert.deepEqual(
		rec.turns.map((t) => [t.calls.map((c) => c.name), t.blocked.map((b) => b.name)]),
		[[["view"], ["grep"]]],
	);
});

async function start(p: ReturnType<typeof fakePi>) {
	await p.emit("session_start");
	await p.emit("before_agent_start");
	const t1 = await p.turn();
	const history = [t1, ...results(t1)];
	const t2 = await p.turn(history);
	return { t1, t2, history };
}

test("turns replay in order with recorded arguments; blocked and stale calls and writes skipped; recorded errors continue", async () => {
	const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "1" });
	try {
		const { t1, t2, history } = await start(p);
		assert.equal(p.model.id, "session");
		assert.equal(t1.stopReason, "toolUse");
		assert.deepEqual(t1.usage.totalTokens, 0);
		assert.deepEqual(
			calls(t1).map((c: any) => c.name),
			["view"],
		);
		assert.deepEqual(p.entries[0].type, REPLAY_ENTRY);
		assert.deepEqual(
			calls(t2).map((c: any) => [c.name, c.arguments]),
			[
				["move", { xyz: [1, 2, 3] }],
				["grip", { v: 1 }],
			],
		);
		assert.match(text(t2), /two at once/);
		// The stale-dropped move and the budget-refused move never ran in the recording: not sent.
		history.push(t2, ...results(t2));
		const t3 = await p.turn(history);
		assert.deepEqual(
			calls(t3).map((c: any) => [c.name, c.arguments]),
			[["grip", { v: 0 }]],
		);
		assert.match(text(t3), /skipped move \(it never ran in the recording: The operator took over/);
		assert.match(text(t3), /skipped move \(it never ran in the recording: Planner turns budget exhausted/);
		// The grip fails again, as it did in the recording: noted, and the replay goes on.
		history.push(t3, ...results(t3, { grip: "gripper jammed" }));
		const t4 = await p.turn(history);
		assert.deepEqual(
			calls(t4).map((c: any) => [c.name, c.arguments]),
			[["finish", { status: "success", summary: "ok" }]],
		);
		assert.match(text(t4), /grip failed, as in the recording: gripper jammed; continuing/);
		assert.match(text(t4), /skipped write/);
		assert.ok(p.stderr.some((l) => /\[replay\] grip failed/.test(l)));
		assert.ok(!p.stderr.some((l) => /differs/.test(l)));
		// pi stops after finish; if asked again, the recording is spent.
		history.push(t4, ...results(t4));
		const t5 = await p.turn(history);
		assert.equal(t5.stopReason, "stop");
		assert.equal(calls(t5).length, 0);
		await p.emit("agent_end");
		assert.match(text(await p.turn()), /already ran/);
	} finally {
		p.restore();
	}
});

for (const error of ["robot error: joint limit", "http://127.0.0.1:1/call: connect ECONNREFUSED"]) {
	test(`a call that fails where the recording's succeeded stops the replay (${error})`, async () => {
		const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "1" });
		try {
			const { t2, history } = await start(p);
			history.push(t2, ...results(t2, { move: error }));
			const t3 = await p.turn(history);
			assert.equal(t3.stopReason, "stop");
			assert.equal(calls(t3).length, 0);
			assert.match(text(t3), /move failed: .*; it succeeded in the recording/);
			assert.match(text(t3), /Replay stopped: .*3 of 7 turns replayed/);
			// No further recorded motion, however often pi asks.
			assert.equal(calls(await p.turn([...history, t3])).length, 0);
		} finally {
			p.restore();
		}
	});
}

test("a recording without finish ends with a text turn", async () => {
	const p = fakePi({ replay: recordSession(false) }, { robot: "arm", task: "3", seed: "1" });
	try {
		await p.emit("session_start");
		let m = await p.turn();
		const history: unknown[] = [];
		for (let i = 0; i < 5 && m.stopReason === "toolUse"; i++) {
			// The recording's grip {v: 0} jammed; this run's does too.
			const jammed = calls(m).some((c: any) => c.name === "grip" && c.arguments.v === 0);
			history.push(m, ...results(m, jammed ? { grip: "gripper jammed" } : {}));
			m = await p.turn(history);
		}
		assert.equal(m.stopReason, "stop");
		assert.match(text(m), /no finish/);
	} finally {
		p.restore();
	}
});

test("an abort ends the replay, and a task mismatch is warned on stderr", async () => {
	const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "7" });
	try {
		await p.emit("session_start");
		await p.emit("before_agent_start");
		assert.ok(p.stderr.some((l) => /differs from the recorded robot_task: seed: 1 -> 7/.test(l)));
		await p.turn();
		const ac = new AbortController();
		ac.abort();
		const aborted = await p.turn([], ac.signal);
		assert.equal(aborted.stopReason, "aborted");
		const after = await p.turn();
		assert.equal(calls(after).length, 0);
		assert.match(text(after), /already ran/);
		// A new session replays from the start.
		await p.emit("session_start");
		assert.deepEqual(
			calls(await p.turn()).map((c: any) => c.name),
			["view"],
		);
	} finally {
		p.restore();
	}
});

test("without --replay the model says so and calls nothing", async () => {
	const p = fakePi({}, { robot: "arm" });
	try {
		await p.emit("session_start");
		const m = await p.turn();
		assert.equal(m.stopReason, "stop");
		assert.match(text(m), /no --replay/);
	} finally {
		p.restore();
	}
});

/** A session of `turns` (content, stopReason, recorded errors by call name), each followed by its results. */
function sessionOf(turns: { content: unknown[]; stopReason?: string; errors?: Record<string, string> }[]) {
	const dir = folder();
	const lines: unknown[] = [
		{ type: "session", version: 3, id: "s", cwd: dir },
		{ type: "message", id: "u", parentId: null, message: { role: "user", content: "Solve." } },
	];
	let leaf = "u";
	turns.forEach((t, i) => {
		const r = turnWithResults(`x${i}`, leaf, t.content, t.errors);
		(r.entries[0] as any).message.stopReason = t.stopReason ?? "toolUse";
		lines.push(...r.entries);
		leaf = r.leaf;
	});
	writeFileSync(join(dir, "2026-01-01T00-00-00-000Z_s.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	return dir;
}

/**
 * What a recording holds for a call: an error result (a list: one text part each), pi's time of the tool's execute()
 * (`ms`), a result whose content is `raw` as given, or no result (`none`).
 */
type Outcome = { error?: string | string[]; ms?: number; none?: true; raw?: unknown };
/**
 * A session of `turns` whose calls have `outcomes` by position (default: a result "{}" without a time). `ms` on a
 * turn is pi's time of the reply. pi records how long a reply took, and how long the tool ran for every call whose
 * tool it entered; a hand-written recording has neither.
 */
function sessionWith(turns: { content: unknown[]; outcomes?: Outcome[]; ms?: number; stopReason?: string }[]) {
	const dir = folder();
	const lines: unknown[] = [
		{ type: "session", version: 3, id: "s", cwd: dir },
		{ type: "message", id: "u", parentId: null, message: { role: "user", content: "Solve." } },
	];
	let leaf = "u";
	turns.forEach((t, i) => {
		const a = assistant(`y${i}`, leaf, t.content, t.stopReason ?? "toolUse");
		if (t.ms !== undefined) Object.assign(a.message, { durationMs: t.ms });
		lines.push(a);
		leaf = a.id;
		(t.content.filter((c: any) => c.type === "toolCall") as { id: string; name: string }[]).forEach((c, k) => {
			const o = t.outcomes?.[k] ?? {};
			if (o.none) return;
			const id = `y${i}_r${k}`;
			lines.push({
				type: "message",
				id,
				parentId: leaf,
				message: {
					role: "toolResult",
					toolCallId: c.id,
					toolName: c.name,
					content: "raw" in o ? o.raw : [o.error ?? "{}"].flat().map((text) => ({ type: "text", text })),
					isError: o.error !== undefined,
					...(o.ms === undefined ? {} : { durationMs: o.ms }),
				},
			});
			leaf = id;
		});
	});
	writeFileSync(join(dir, "2026-01-01T00-00-00-000Z_s.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	return dir;
}
/** What a recording's turns hold for a replay: the calls it sends and the calls it skips, by name. */
const sentAndSkipped = (dir: string) =>
	loadRecording(dir).turns.map((t) => [t.calls.map((c) => c.name), t.blocked.map((b) => b.name)]);

/**
 * Execute turn `m`'s calls as pi does for a sequential batch: tool_call hooks, then the result; `errors` fail calls
 * by name. Every call ends with tool_execution_end; one a handler blocked gets no tool_result.
 */
async function runBatch(p: ReturnType<typeof fakePi>, m: any, errors: Record<string, string> = {}) {
	const out: any[] = [];
	const sent: string[] = [];
	for (const c of calls(m)) {
		const blocked = await p.emit("tool_call", { toolCallId: c.id, toolName: c.name, input: c.arguments });
		const failed = blocked?.block ? (blocked.reason ?? "blocked") : errors[c.name];
		if (!blocked?.block) sent.push(c.name);
		const content = [{ type: "text", text: failed ?? "{}" }];
		if (!blocked?.block)
			await p.emit("tool_result", { toolCallId: c.id, input: c.arguments, content, isError: failed !== undefined });
		await p.emit("tool_execution_end", {
			toolCallId: c.id,
			toolName: c.name,
			result: { content },
			isError: failed !== undefined,
		});
		out.push({ role: "toolResult", toolCallId: c.id, toolName: c.name, content, isError: failed !== undefined });
	}
	return { results: out, sent };
}

test("a call the robot's tool set refused never ran: it is skipped, not sent again; a tool failing with another's refusal ran", () => {
	const dir = sessionOf([
		{
			content: [call("codemode", { code: "return 1" }), call("view", {}), call("relay", { tool: "codemode" })],
			errors: {
				codemode: "codemode is not one of the tools arm gives the planner.",
				relay: "codemode is not one of the tools arm gives the planner.",
			},
		},
	]);
	const rec = loadRecording(dir);
	assert.deepEqual(
		rec.turns.map((t) => [t.calls.map((c) => c.name), t.blocked.map((b) => b.name)]),
		[[["view", "relay"], ["codemode"]]],
	);
});

test("a call refused for an argument its tool does not take never ran: it is skipped; a tool failing with another's refusal ran", () => {
	// The robot base's own sentence (../src/robot.ts builds it the same way), whatever the stray name holds: a line
	// separator is escaped, so the refusal stays one line.
	const key = `step${String.fromCharCode(0x2028)}m`;
	const refusal = strayRefusal("move", strayArguments({ [key]: -0.05 }, [], () => ["xyz", "gripper"]) as string);
	assert.equal(refusal, 'move has no argument "step\\u2028m"; it takes xyz, gripper.');
	const dir = sessionOf([
		{
			content: [
				call("move", { xyz: [1, 2, 3], [key]: -0.05 }),
				call("view", {}),
				call("relay", { tool: "move" }),
				call("grip", {}),
			],
			// relay failed with move's refusal, and grip with its own after other text: both ran.
			errors: { move: refusal, relay: refusal, grip: `robot error: ${refusal.replace("move", "grip")}` },
		},
	]);
	const rec = loadRecording(dir);
	assert.deepEqual(
		rec.turns.map((t) => [t.calls.map((c) => c.name), t.blocked.map((b) => b.name)]),
		[[["view", "relay", "grip"], ["move"]]],
	);
});

test("a reply cut at the output token limit (or otherwise abnormal) has its calls skipped with a note", async () => {
	const dir = sessionOf([
		// A reply that ended with "stop" and carries a call ended normally, like one that ended with "toolUse".
		{ content: [call("view", {})], stopReason: "stop" },
		{
			content: [call("move", { xyz: [1, 2] })],
			stopReason: "length",
			errors: {
				move: 'Tool call "move" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.',
			},
		},
		{ content: [call("grip", { v: 1 })], stopReason: "pause_turn" },
		{ content: [call("move", { xyz: [1, 2, 3] })] },
	]);
	const rec = loadRecording(dir);
	assert.deepEqual(
		rec.turns.map((t) => [t.calls.map((c) => c.name), t.blocked.map((b) => b.name)]),
		[
			[["view"], []],
			[[], ["move"]],
			[[], ["grip"]],
			[["move"], []],
		],
	);
	assert.match(rec.turns[1].blocked[0].reason, /stopReason "length", so its arguments may be truncated/);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const t2 = await p.turn([t1, ...(await runBatch(p, t1)).results]);
		// The truncated move is never sent; the next real turn is.
		assert.deepEqual(
			calls(t2).map((c: any) => [c.name, c.arguments]),
			[["move", { xyz: [1, 2, 3] }]],
		);
		assert.match(text(t2), /skipped move \(it never ran in the recording: the reply ended with stopReason "length"/);
		assert.match(
			text(t2),
			/skipped grip \(it never ran in the recording: the reply ended with stopReason "pause_turn"/,
		);
	} finally {
		p.restore();
	}
});

test("a failure inside a multi-call turn stops the replay at that call: the rest of the turn is not sent", async () => {
	const dir = sessionOf([
		{ content: [call("move", { xyz: [1, 0, 0] }), call("move", { xyz: [2, 0, 0] }), call("grasp", {})] },
		{ content: [call("finish", { status: "success" })] },
	]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		assert.equal(calls(t1).length, 3);
		const { results: r1, sent } = await runBatch(p, t1, { move: "robot error: IK failed" });
		assert.deepEqual(sent, ["move"]);
		assert.match(
			r1[1].content[0].text,
			/replay stopped \(move failed: robot error: IK failed; it succeeded in the recording\)/,
		);
		const t2 = await p.turn([t1, ...r1]);
		assert.equal(calls(t2).length, 0);
		assert.match(text(t2), /move not sent/);
		assert.match(text(t2), /grasp not sent/);
		assert.match(text(t2), /Replay stopped/);
		assert.equal(calls(await p.turn([t1, ...r1, t2])).length, 0);
	} finally {
		p.restore();
	}
});

test("a recorded call the robot refuses now stops the replay at that call: the rest of the turn is not sent", async () => {
	// The recording's move ran with an argument its tool no longer takes. This run's gate blocks it, and a blocked
	// call gets no tool_result: the replay learns of it when the call ends.
	const dir = sessionOf([
		{
			content: [
				call("move", { xyz: [1, 0, 0], step_m: 0.005 }),
				call("grasp", {}),
				call("move", { xyz: [2, 0, 0] }),
			],
		},
		{ content: [call("finish", { status: "success" })] },
	]);
	const refusal = 'move has no argument "step_m"; it takes xyz.';
	const p = fakePi({ replay: dir }, {}, (e) =>
		e.toolName === "move" && "step_m" in e.input ? { block: true, reason: refusal } : undefined,
	);
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		assert.equal(calls(t1).length, 3);
		const { results: r1, sent } = await runBatch(p, t1);
		assert.deepEqual(sent, []);
		assert.deepEqual(
			r1.map((r) => r.content[0].text),
			[
				refusal,
				...Array(2).fill(
					`replay stopped (move failed: ${refusal}; it succeeded in the recording); this recorded call was not sent`,
				),
			],
		);
		const t2 = await p.turn([t1, ...r1]);
		assert.equal(calls(t2).length, 0);
		assert.match(text(t2), /grasp not sent/);
		assert.match(text(t2), /Replay stopped/);
	} finally {
		p.restore();
	}
});

test("a multi-call turn whose calls end as recorded is sent whole, a failure the recording also had included", async () => {
	const dir = sessionOf([
		{ content: [call("move", { xyz: [1, 0, 0] }), call("grasp", {}), call("move", { xyz: [2, 0, 0] })] },
		{ content: [call("grip", { v: 1 }), call("move", { xyz: [3, 0, 0] })], errors: { grip: "gripper jammed" } },
		{ content: [call("finish", { status: "success" })] },
	]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const { results: r1, sent } = await runBatch(p, t1);
		assert.deepEqual(sent, ["move", "grasp", "move"]);
		const t2 = await p.turn([t1, ...r1]);
		const { results: r2, sent: sent2 } = await runBatch(p, t2, { grip: "gripper jammed" });
		assert.deepEqual(sent2, ["grip", "move"]);
		const t3 = await p.turn([t1, ...r1, t2, ...r2]);
		assert.match(text(t3), /grip failed, as in the recording/);
		assert.deepEqual(
			calls(t3).map((c: any) => c.name),
			["finish"],
		);
	} finally {
		p.restore();
	}
});

test("only the recorded call's own end counts: a call it made of the same tool, refused, does not stop the turn", async () => {
	const dir = sessionOf([{ content: [call("relay", { tool: "move" }), call("move", { xyz: [1, 0, 0] })] }]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const [relay, move] = calls(await p.turn());
		// pi ends a nested call under the id <caller>/<n>; relay reports the refusal and succeeds, as recorded.
		await p.emit("tool_execution_end", {
			toolCallId: `${relay.id}/1`,
			parentToolCallId: relay.id,
			toolName: "move",
			result: { content: [{ type: "text", text: "refused" }] },
			isError: true,
		});
		await p.emit("tool_execution_end", {
			toolCallId: relay.id,
			toolName: "relay",
			result: { content: [{ type: "text", text: "error: refused" }] },
			isError: false,
		});
		assert.equal(
			await p.emit("tool_call", { toolCallId: move.id, toolName: "move", input: move.arguments }),
			undefined,
		);
	} finally {
		p.restore();
	}
});

test("a result without content is judged when its call ends", async () => {
	// An untyped tool may return none. It failed in the recording and succeeds now: the rest is withheld.
	const dir = sessionOf([{ content: [call("bare", {}), call("move", { xyz: [1, 0, 0] })], errors: { bare: "boom" } }]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const [bare, move] = calls(await p.turn());
		await p.emit("tool_execution_end", {
			toolCallId: bare.id,
			toolName: "bare",
			result: { details: {} },
			isError: false,
		});
		const blocked = await p.emit("tool_call", { toolCallId: move.id, toolName: "move", input: move.arguments });
		assert.equal(
			blocked?.reason,
			"replay stopped (bare succeeded; it failed in the recording (boom)); this recorded call was not sent",
		);
	} finally {
		p.restore();
	}
});

test("a call an earlier replay withheld never ran: replaying that replay's session does not send it", () => {
	const sentence =
		"replay stopped (move failed: robot error: IK failed; it succeeded in the recording); this recorded call was not sent";
	const dir = sessionOf([
		{
			content: [call("move", { xyz: [1, 0, 0] }), call("grasp", {}), call("relay", {}), call("grip", {})],
			// relay and grip failed with the sentence inside other text: both ran.
			errors: {
				move: "robot error: IK failed",
				grasp: sentence,
				relay: `relay failed: ${sentence}`,
				grip: `${sentence}\nmore`,
			},
		},
	]);
	assert.deepEqual(
		loadRecording(dir).turns.map((t) => [t.calls.map((c) => c.name), t.blocked.map((b) => b.name)]),
		[[["move", "relay", "grip"], ["grasp"]]],
	);
});

test("a code-mode session replays every run_code program verbatim", async () => {
	// A program with quotes, newlines, indentation and unicode, as the model wrote it.
	const program =
		'obs = get_observation()\nfor i in range(3):\n    move_to([0.1, -0.2, 0.3], gripper=+1)  # "carry"\nRESULT = {"done": True, "note": "héllo\\n"}\n';
	const dir = sessionOf([
		{ content: [call("run_code", { code: "print(get_state())" })] },
		{
			content: [call("run_code", { code: program, timeout_s: 45 })],
			errors: { run_code: 'run_code: {\n "status": "error"' },
		},
		{ content: [call("run_code", { code: "set_gripper(False)" })] },
		{ content: [call("finish", { status: "success", summary: "ok" })] },
	]);
	const rec = loadRecording(dir);
	assert.deepEqual(
		rec.turns.map((t) => t.calls.map((c) => c.name)),
		[["run_code"], ["run_code"], ["run_code"], ["finish"]],
	);
	assert.equal(rec.turns[1].calls[0].arguments.code, program);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		assert.deepEqual(
			calls(t1).map((c: any) => c.arguments),
			[{ code: "print(get_state())" }],
		);
		const history = [t1, ...(await runBatch(p, t1)).results];
		const t2 = await p.turn(history);
		assert.deepEqual(
			calls(t2).map((c: any) => c.arguments),
			[{ code: program, timeout_s: 45 }],
			"the program is sent byte for byte, with its timeout",
		);
		// A run_code that errored in the recording (the program raised) errors again: noted, replay goes on.
		history.push(t2, ...(await runBatch(p, t2, { run_code: 'run_code: {\n "status": "error"' })).results);
		const t3 = await p.turn(history);
		assert.deepEqual(
			calls(t3).map((c: any) => c.arguments),
			[{ code: "set_gripper(False)" }],
		);
		assert.match(text(t3), /run_code failed, as in the recording/);
		history.push(t3, ...(await runBatch(p, t3)).results);
		assert.deepEqual(
			calls(await p.turn(history)).map((c: any) => c.name),
			["finish"],
		);
	} finally {
		p.restore();
	}
});

test("a call that failed in the recording but succeeds now stops the replay", async () => {
	const dir = sessionOf([
		{ content: [call("grip", { v: 1 }), call("move", { xyz: [1, 0, 0] })], errors: { grip: "gripper jammed" } },
		{ content: [call("move", { xyz: [2, 0, 0] })] },
	]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const { results: r1, sent } = await runBatch(p, t1);
		assert.deepEqual(sent, ["grip"]);
		const t2 = await p.turn([t1, ...r1]);
		assert.equal(calls(t2).length, 0);
		assert.match(text(t2), /grip succeeded; it failed in the recording \(gripper jammed\)/);
		assert.match(text(t2), /Replay stopped/);
	} finally {
		p.restore();
	}
});

/** What a request gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"replay/session replays recorded turns and answers no other request; --aux-model names a model for the checks that take one";
/**
 * `m` is that answer: an error of the replay's own model, with nothing of the recording and nothing of the replay's
 * own report.
 */
function assertNotATurn(m: any) {
	assert.equal(m.stopReason, "error");
	assert.equal(m.errorMessage, NOT_A_TURN);
	assert.deepEqual(m.content, []);
	assert.deepEqual([m.api, m.provider, m.model], ["replay", "replay", "session"]);
	assert.equal(m.usage.totalTokens, 0);
}
/** One call of a sequential batch as pi runs it: its hook, `during` while it runs, then its end; `error` fails it. */
async function runCall(p: ReturnType<typeof fakePi>, c: any, error?: string, during?: () => Promise<unknown>) {
	const blocked = await p.emit("tool_call", { toolCallId: c.id, toolName: c.name, input: c.arguments });
	if (!blocked?.block) await during?.();
	const said = blocked?.block ? (blocked.reason ?? "blocked") : (error ?? "{}");
	await p.emit("tool_execution_end", {
		toolCallId: c.id,
		toolName: c.name,
		result: { content: [{ type: "text", text: said }] },
		isError: blocked?.block === true || error !== undefined,
	});
	return said;
}
const twoTurns = () =>
	sessionOf([
		{ content: [call("move", { xyz: [1, 0, 0] }), call("grip", { v: 1 })] },
		{ content: [{ type: "text", text: "two at once" }, call("move", { xyz: [2, 0, 0] }), call("finish", {})] },
	]);

test("a request that is not the session's turn, before the first turn: the replay has not started and its first turn is whole", async () => {
	const p = fakePi({ replay: recordSession() }, { robot: "arm", task: "3", seed: "7" });
	try {
		await p.emit("session_start");
		assertNotATurn(await p.side());
		assert.equal(p.entries.length, 0);
		assert.equal(p.stderr.length, 0);
		// The warning of a task that differs belongs to the replay's start: a request before it does not skip it.
		await p.emit("before_agent_start");
		assert.ok(p.stderr.some((l) => /differs from the recorded robot_task: seed: 1 -> 7/.test(l)));
		assertNotATurn(await p.side());
		const t1 = await p.turn();
		assert.deepEqual(
			calls(t1).map((c: any) => [c.id, c.name]),
			[["replay_1", "view"]],
		);
		assert.match(text(t1), /^replay: 7 recorded turns from /);
		assert.deepEqual(
			p.entries.map((e) => e.type),
			[REPLAY_ENTRY],
		);
	} finally {
		p.restore();
	}
});

/**
 * What the replay takes for no turn of its session: a request without an id, under another id, or with pi's mark on
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
for (const [kind, options] of NOT_TURNS) {
	test(`a request that is not the session's turn (${kind}) leaves the replay as it is: before the first turn, while a call of a batch runs and behind it; the call's divergence still withholds the rest and stops the replay`, async () => {
		const p = fakePi({ replay: twoTurns() }, {});
		try {
			await p.emit("session_start");
			assertNotATurn(await p.side(options));
			assert.equal(p.entries.length, 0);
			assert.equal(p.stderr.length, 0);
			// The replay starts with the session's own first turn: its opening note, its entry, the recording's first calls.
			const t1 = await p.turn();
			assert.match(text(t1), /^replay: 2 recorded turns from /);
			assert.deepEqual(
				p.entries.map((e) => e.type),
				[REPLAY_ENTRY],
			);
			const [move, grip] = calls(t1);
			assert.deepEqual(
				[move, grip].map((c: any) => [c.id, c.name]),
				[
					["replay_1", "move"],
					["replay_2", "grip"],
				],
			);
			// A request while the move runs; the move then fails where the recording's succeeded.
			const failed = "robot error: IK failed";
			await runCall(p, move, failed, async () => assertNotATurn(await p.side(options)));
			// One more request once the call has ended, before the next call is prepared. Its transcript is like the
			// loop's (the prompt, the replay's own turn, the move's result): the transcript is not read.
			const prompt = { role: "user", content: "Replay the session.", timestamp: 0 };
			assertNotATurn(await p.side(options, [prompt, t1, results(t1, { move: failed })[0]]));
			// Nor does one that brings a result for both calls before the second has run undo the stop.
			assertNotATurn(await p.side(options, [prompt, t1, ...results(t1, { move: failed })]));
			const withheld = `replay stopped (move failed: ${failed}; it succeeded in the recording); this recorded call was not sent`;
			assert.equal(await runCall(p, grip), withheld);
			// And one with the very transcript the session's next request brings.
			const next = [prompt, t1, ...results(t1, { move: failed, grip: withheld })];
			assertNotATurn(await p.side(options, next));
			const t2 = await p.turn(next);
			assert.deepEqual(calls(t2), []);
			assert.deepEqual(text(t2).split("\n"), [
				`replay: move failed: ${failed}; it succeeded in the recording`,
				"replay: grip not sent",
				"Replay stopped: a call's outcome diverged from the recording's; no further recorded motion is sent (1 of 2 turns replayed).",
			]);
			// The replay printed its opening note and these two and wrote its one entry: nothing for the other requests.
			assert.equal(p.stderr.length, 3);
			assert.deepEqual(
				p.entries.map((e) => e.type),
				[REPLAY_ENTRY],
			);
		} finally {
			p.restore();
		}
	});
}

test("a request that is not the session's turn, between two turns: the next turn is the recording's next, with the notes of the turn before", async () => {
	const dir = sessionOf([
		{ content: [call("grip", { v: 1 })], errors: { grip: "gripper jammed" } },
		{ content: [{ type: "text", text: "two at once" }, call("move", { xyz: [2, 0, 0] })] },
		{ content: [call("finish", {})] },
	]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const history = [t1, ...results(t1, { grip: "gripper jammed" })];
		// The request brings the transcript of the session's next one.
		assertNotATurn(await p.side({}, history));
		// The grip failed as in the recording: the session's next turn says so, not the other request's answer.
		const t2 = await p.turn(history);
		assert.deepEqual(
			calls(t2).map((c: any) => [c.id, c.name, c.arguments]),
			[["replay_2", "move", { xyz: [2, 0, 0] }]],
		);
		assert.deepEqual(text(t2).split("\n"), [
			"replay: grip failed, as in the recording: gripper jammed; continuing",
			"two at once",
		]);
		// A call that diverged still stops the replay at the session's next turn.
		assertNotATurn(await p.side());
		const t3 = await p.turn([...history, t2, ...results(t2, { move: "robot error: IK failed" })]);
		assert.deepEqual(calls(t3), []);
		assert.match(
			text(t3),
			/^replay: move failed: robot error: IK failed; it succeeded in the recording\nReplay stopped: /,
		);
		assert.match(text(t3), /\(2 of 3 turns replayed\)\.$/);
	} finally {
		p.restore();
	}
});

test("a request that is not the session's turn gets the same answer before a session starts, before the replay's first turn, between turns and once the replay is over", async () => {
	const p = fakePi({ replay: twoTurns() }, {});
	try {
		// No session yet: no request is a session's turn, of whatever kind, with or without options.
		for (const [, options] of NOT_TURNS) assertNotATurn(await p.side(options));
		assertNotATurn(await p.bare());
		assertNotATurn(await p.turn());
		await p.emit("session_start");
		assertNotATurn(await p.side());
		const t1 = await p.turn();
		assert.deepEqual(
			calls(t1).map((c: any) => [c.id, c.name]),
			[
				["replay_1", "move"],
				["replay_2", "grip"],
			],
		);
		assertNotATurn(await p.side());
		assertNotATurn(await p.bare());
		const t2 = await p.turn([t1, ...results(t1)]);
		assert.deepEqual(
			calls(t2).map((c: any) => [c.id, c.name]),
			[
				["replay_3", "move"],
				["replay_4", "finish"],
			],
		);
		await p.turn([t1, ...results(t1), t2, ...results(t2)]);
		await p.emit("agent_end");
		for (const [, options] of NOT_TURNS) assertNotATurn(await p.side(options));
		assertNotATurn(await p.bare());
		assert.deepEqual(
			p.entries.map((e) => e.type),
			[REPLAY_ENTRY],
		);
		// The session's own request after the end is still told that the replay ran.
		assert.match(text(await p.turn()), /already ran/);
	} finally {
		p.restore();
	}
});

test("a request that is not the session's turn, once a run has ended with calls in flight, prints and writes nothing and leaves them remembered", async () => {
	const p = fakePi({ replay: twoTurns() }, {});
	try {
		await p.emit("session_start");
		await p.turn();
		await p.emit("agent_end");
		const printed = p.stderr.length;
		for (const [, options] of NOT_TURNS) assertNotATurn(await p.side(options));
		assert.equal(p.stderr.length, printed);
		assert.deepEqual(
			p.entries.map((e) => e.type),
			[REPLAY_ENTRY],
		);
		assert.deepEqual(text(await p.turn()).split("\n"), [
			"replay: move: no result; continuing",
			"replay: grip: no result; continuing",
			"The replay already ran in this session.",
		]);
	} finally {
		p.restore();
	}
});

test("a request under another id is not the session's turn, nor is a one-off summary pi sends under the session's id", async () => {
	const p = fakePi({ replay: twoTurns() }, {});
	try {
		await p.emit("session_start");
		// pi's compaction and branch summaries send a fresh id each; its bug-report summary sends the session's.
		// pi marks them all as not to be cached.
		const other = { sessionId: "0193e1c0-0000-7000-8000-000000000000" };
		const summary = { sessionId: SESSION, cacheRetention: "none" };
		assertNotATurn(await p.side(other));
		assertNotATurn(await p.side(summary));
		const t1 = await p.turn();
		assertNotATurn(await p.side(other));
		assertNotATurn(await p.side(summary));
		// pi's compaction and branch summaries carry both, an id of their own and the mark; nor does the mark need an id.
		assertNotATurn(await p.side({ ...other, cacheRetention: "none" }));
		assertNotATurn(await p.side({ cacheRetention: "none" }));
		// A request of the session's own loop is its turn whatever else it carries: here a retention that is not the mark.
		// It brings the move's result alone, and the turn says so before the recording's text.
		const t2 = await p.side({ sessionId: SESSION, cacheRetention: "long" }, [t1, results(t1)[0]]);
		assert.deepEqual(text(t2).split("\n"), ["replay: grip: no result; continuing", "two at once"]);
		assert.deepEqual(
			[t1, t2].map((m) => calls(m).map((c: any) => c.id)),
			[
				["replay_1", "replay_2"],
				["replay_3", "replay_4"],
			],
		);
		// So is one with yet another retention. This one brings no result of the two calls in flight, and the replay
		// notes that before it closes.
		assert.deepEqual(text(await p.side({ sessionId: SESSION, cacheRetention: "short" })).split("\n"), [
			"replay: move: no result; continuing",
			"replay: finish: no result; continuing",
			"Replay complete: all recorded turns were replayed.",
		]);
		// A new session has an id of its own: a request under the one before is not its turn.
		p.session.id = "session-2";
		await p.emit("session_start");
		assertNotATurn(await p.side({ sessionId: SESSION }));
		assert.deepEqual(
			calls(await p.turn()).map((c: any) => c.id),
			["replay_1", "replay_2"],
		);
	} finally {
		p.restore();
	}
});

test("a request that is not the session's turn and arrives aborted does not end the replay", async () => {
	const p = fakePi({ replay: twoTurns() }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const gone = new AbortController();
		gone.abort();
		assertNotATurn(await p.side({ signal: gone.signal }));
		for (const [kind, options] of NOT_TURNS) if (kind.endsWith("aborted")) assertNotATurn(await p.side(options));
		assert.deepEqual(
			calls(await p.turn([t1, ...results(t1)])).map((c: any) => [c.id, c.name]),
			[
				["replay_3", "move"],
				["replay_4", "finish"],
			],
		);
	} finally {
		p.restore();
	}
});

test("the answer to a request that is not the session's turn is no error pi sends again, and none the fallback planner by default changes models for", async () => {
	const p = fakePi({ replay: twoTurns() }, {});
	try {
		await p.emit("session_start");
		const m = await p.side();
		assertNotATurn(m);
		// pi sends a request again after an error it takes for a passing one, and after one it takes for a full context.
		assert.equal(isRetryableAssistantError(m), false);
		assert.equal(isContextOverflow(m, p.model.contextWindow), false);
		assert.doesNotMatch(m.errorMessage, RELAY_ERRORS);
	} finally {
		p.restore();
	}
});

test("the answer to a request that is not the session's turn reaches its asker as one error event", async () => {
	const p = fakePi({ replay: twoTurns() }, {});
	try {
		await p.emit("session_start");
		const stream = p.ask();
		const events: any[] = [];
		for await (const event of stream) events.push(event);
		assert.deepEqual(
			events.map((e) => [e.type, e.reason]),
			[["error", "error"]],
		);
		assertNotATurn(events[0].error);
		assert.equal(await stream.result(), events[0].error);
		// A caller that passes no options at all gets the same answer.
		assertNotATurn(await p.bare());
	} finally {
		p.restore();
	}
});

test("a request that is not the session's turn gets the same answer when there is nothing to replay, and the session's own turn still says why, in a later session too", async () => {
	for (const [flags, why] of [
		[{}, /^no --replay session given/],
		[{ replay: join(folder(), "gone.jsonl") }, /^cannot read --replay /],
	] as const) {
		const p = fakePi(flags, {});
		try {
			await p.emit("session_start");
			for (const [, options] of NOT_TURNS) assertNotATurn(await p.side(options));
			const m = await p.turn();
			assert.equal(m.stopReason, "stop");
			assert.match(text(m), why);
			p.session.id = "session-2";
			await p.emit("session_start");
			for (const [, options] of NOT_TURNS) assertNotATurn(await p.side(options));
			assert.match(text(await p.turn()), why);
		} finally {
			p.restore();
		}
	}
});

test("an id that only resembles the session's is another id, and a one-off summary is told by its mark alone", async () => {
	const p = fakePi({ replay: twoTurns() }, {});
	try {
		await p.emit("session_start");
		for (const sessionId of [
			`${SESSION}0`,
			SESSION.toUpperCase(),
			SESSION.slice(0, -1),
			SESSION.slice(1),
			` ${SESSION}`,
			`${SESSION} `,
			"",
		])
			assertNotATurn(await p.side({ sessionId }));
		// The stub's other requests set a token limit; pi's mark counts without one.
		assertNotATurn(await p.side({ sessionId: SESSION, cacheRetention: "none", maxTokens: undefined }));
		assert.deepEqual(
			calls(await p.turn()).map((c: any) => c.id),
			["replay_1", "replay_2"],
		);
	} finally {
		p.restore();
	}
});

const NO_RESULT = "the recording holds no result for it";

test("a call the recording holds no result for is not sent, and the replay goes on with the calls that have one", async () => {
	const dir = sessionWith([
		{ content: [call("view", {})] },
		// Aborted after the first move: the second move and the grip never started.
		{
			content: [call("move", { xyz: [1, 0, 0] }), call("move", { xyz: [2, 0, 0] }), call("grip", { v: 1 })],
			outcomes: [{}, { none: true }, { none: true }],
		},
		// The session went on. A result is missing between two that are there: that call alone is skipped.
		{
			content: [call("grip", { v: 0 }), call("move", { xyz: [3, 0, 0] }), call("view", {})],
			outcomes: [{}, { none: true }, {}],
		},
		// The run was cut while finish ran: a finish without a result is no finish.
		{ content: [call("finish", { status: "success" })], outcomes: [{ none: true }] },
	]);
	const rec = loadRecording(dir);
	assert.deepEqual(sentAndSkipped(dir), [
		[["view"], []],
		[["move"], ["move", "grip"]],
		[["grip", "view"], ["move"]],
		[[], ["finish"]],
	]);
	assert.deepEqual(rec.turns[1].blocked, [
		{ name: "move", reason: NO_RESULT, unfinished: true },
		{ name: "grip", reason: NO_RESULT, unfinished: true },
	]);
	assert.equal(rec.finished, false);
	// In a recording with pi's times too: the loader looks for the result before it reads the times.
	const cut = sessionWith([
		{
			content: [call("move", { xyz: [1, 0, 0] }), call("grip", { v: 1 })],
			outcomes: [{ ms: 4 }, { none: true }],
			ms: 900,
		},
	]);
	assert.deepEqual(sentAndSkipped(cut), [[["move"], ["grip"]]]);
	assert.deepEqual(loadRecording(cut).turns[0].blocked, [{ name: "grip", reason: NO_RESULT, unfinished: true }]);
	const p = fakePi({ replay: dir }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		const b1 = await runBatch(p, t1);
		const t2 = await p.turn([t1, ...b1.results]);
		assert.deepEqual(
			calls(t2).map((c: any) => [c.name, c.arguments]),
			[["move", { xyz: [1, 0, 0] }]],
		);
		// The note does not say such a call never ran: the recording does not tell.
		assert.ok(
			text(t2).includes(`replay: skipped move (${NO_RESULT})\nreplay: skipped grip (${NO_RESULT})`),
			text(t2),
		);
		const b2 = await runBatch(p, t2);
		const t3 = await p.turn([t1, ...b1.results, t2, ...b2.results]);
		assert.deepEqual(
			calls(t3).map((c: any) => [c.name, c.arguments]),
			[
				["grip", { v: 0 }],
				["view", {}],
			],
		);
		const b3 = await runBatch(p, t3);
		const t4 = await p.turn([t1, ...b1.results, t2, ...b2.results, t3, ...b3.results]);
		assert.equal(calls(t4).length, 0);
		assert.ok(text(t4).includes(`replay: skipped finish (${NO_RESULT})`), text(t4));
		assert.match(text(t4), /Replay complete: the recording has no finish/);
		assert.deepEqual([...b1.sent, ...b2.sent, ...b3.sent], ["view", "move", "grip", "view"]);
	} finally {
		p.restore();
	}
});

test("in a recording with pi's times, a result without one is read as a call pi answered itself: skipped whatever its text says", async () => {
	// The approval gate's and the closed-loop gate's refusals, a failed validation and a hook that threw: none is
	// a text of the replay's list, and none entered its tool.
	const denied = "move was not approved by the operator: the operator declined this motion. It did not run.";
	const unknown =
		"move refused: the last motion (move) has an unknown outcome (the call timed out). Re-observe with view before issuing another motion.";
	const invalid =
		'Validation failed for tool "grip":\n  - v: must be number\n\nReceived arguments:\n{\n  "v": "tight"\n}';
	/** One reply: four calls pi answered itself, a relay that entered its tool and failed with the same words, a grip that ran. */
	const reply = (ms?: number) => ({
		content: [
			call("move", { xyz: [1, 0, 0] }),
			call("move", { xyz: [2, 0, 0] }),
			call("grip", { v: "tight" }),
			call("view", {}),
			call("relay", { tool: "move" }),
			call("grip", { v: 1 }),
		],
		outcomes: [
			{ error: denied },
			{ error: unknown },
			{ error: invalid },
			{ error: "fetch failed" },
			{ error: denied, ms },
			{ ms },
		],
	});
	const timed = sessionWith([{ ...reply(12), ms: 900 }]);
	assert.deepEqual(sentAndSkipped(timed), [
		[
			["relay", "grip"],
			["move", "move", "grip", "view"],
		],
	]);
	const turn = loadRecording(timed).turns[0];
	assert.deepEqual(
		turn.blocked.map((b) => [b.reason, b.unfinished]),
		[
			[denied, undefined],
			[unknown, undefined],
			[invalid.replace(/\s+/g, " "), undefined],
			["fetch failed", undefined],
		],
	);
	assert.deepEqual([turn.calls[0].error, turn.calls[1].error], [denied, undefined]);
	// A timed result marks the recording by itself, and so does the reply's own time: a recording whose every call
	// was refused has no timed result.
	assert.deepEqual(sentAndSkipped(sessionWith([reply(12)])), sentAndSkipped(timed));
	const refused = { content: [call("move", { xyz: [1, 0, 0] })], outcomes: [{ error: denied }] };
	assert.deepEqual(sentAndSkipped(sessionWith([{ ...refused, ms: 900 }])), [[[], ["move"]]]);
	// The times are read for every result, error or not; a result that says nothing still gets a reason.
	assert.deepEqual(loadRecording(sessionWith([{ content: [call("view", {})], ms: 900 }])).turns[0].blocked, [
		{ name: "view", reason: "{}" },
	]);
	const mute = sessionWith([{ content: [call("view", {})], outcomes: [{ error: " " }], ms: 900 }]);
	assert.deepEqual(loadRecording(mute).turns[0].blocked, [{ name: "view", reason: "its result says nothing" }]);
	// A result that is no error and whose content is no list of parts (a recording written by hand) says nothing
	// either.
	const bare = sessionWith([
		{
			content: [call("move", { xyz: [1, 0, 0] }), call("grip", { v: 1 }), call("turn", {}), call("view", {})],
			outcomes: [{ raw: "moved" }, { raw: null }, { raw: undefined }, { raw: [null] }],
			ms: 900,
		},
	]);
	assert.deepEqual(
		loadRecording(bare).turns[0].blocked,
		["move", "grip", "turn", "view"].map((name) => ({ name, reason: "its result says nothing" })),
	);
	// A result's text parts are read as one text, a space between them: as a reason, and as a recorded error.
	const parts = sessionWith([
		{
			content: [call("move", { xyz: [1, 0, 0] }), call("grip", { v: 1 })],
			outcomes: [
				{ error: ["move was not approved by the operator:", "too close."] },
				{ error: ["gripper", "jammed"], ms: 7 },
			],
			ms: 900,
		},
	]);
	assert.deepEqual(
		[loadRecording(parts).turns[0].blocked[0].reason, loadRecording(parts).turns[0].calls[0].error],
		["move was not approved by the operator: too close.", "gripper jammed"],
	);
	// The texts are read in a recording with times too: a listed refusal that a tool raised itself has a time and
	// is still skipped by its text; a tool that failed with another's refusal is sent. A time of 0 ms is a time.
	const stray = 'move has no argument "step_m"; it takes xyz, gripper.';
	const inside = sessionWith([
		{
			content: [
				call("move", { xyz: [1, 0, 0] }),
				call("move", { xyz: [2, 0, 0], step_m: 0.005 }),
				call("relay", { tool: "move" }),
				call("grip", { v: 1 }),
			],
			outcomes: [
				{ error: "operator submitted a terminal verdict; stopping", ms: 3 },
				{ error: stray, ms: 2 },
				{ error: stray, ms: 2 },
				{ ms: 0 },
			],
			ms: 900,
		},
	]);
	assert.deepEqual(sentAndSkipped(inside), [
		[
			["relay", "grip"],
			["move", "move"],
		],
	]);
	// A result's 0 ms marks the recording by itself, and a time anywhere on the branch marks every turn of it: the
	// turns before it and the turns behind it.
	const zero = {
		content: [call("move", { xyz: [1, 0, 0] }), call("grip", { v: 1 })],
		outcomes: [{ error: denied }, { ms: 0 }],
	};
	assert.deepEqual(sentAndSkipped(sessionWith([zero])), [[["grip"], ["move"]]]);
	assert.deepEqual(
		sentAndSkipped(sessionWith([refused, { content: [call("grip", { v: 1 })], outcomes: [{ ms: 5 }] }])),
		[
			[[], ["move"]],
			[["grip"], []],
		],
	);
	assert.deepEqual(
		sentAndSkipped(sessionWith([{ content: [call("grip", { v: 1 })], outcomes: [{ ms: 5 }] }, refused])),
		[
			[["grip"], []],
			[[], ["move"]],
		],
	);
	// A recording without any time (written by hand, or by a pi that recorded none) is read by the texts alone, as
	// before: none of these is on the list.
	assert.deepEqual(sentAndSkipped(sessionWith([reply()])), [[["move", "move", "grip", "view", "relay", "grip"], []]]);
	assert.deepEqual(sentAndSkipped(sessionWith([refused])), [[["move"], []]]);
	const p = fakePi({ replay: timed }, {});
	try {
		await p.emit("session_start");
		const t1 = await p.turn();
		assert.deepEqual(
			calls(t1).map((c: any) => [c.name, c.arguments]),
			[
				["relay", { tool: "move" }],
				["grip", { v: 1 }],
			],
		);
		assert.ok(text(t1).includes(`replay: skipped move (it never ran in the recording: ${denied})`), text(t1));
		assert.ok(text(t1).includes("replay: skipped view (it never ran in the recording: fetch failed)"), text(t1));
	} finally {
		p.restore();
	}
});

test("a call whose arguments did not parse is never sent: pi stored {} for it", () => {
	const cut = {
		...call("open", {}),
		argumentsError: 'Unterminated string in JSON at position 12\n\nReceived arguments:\n{"force": "lo',
	};
	const said = `Invalid arguments for tool "open": ${cut.argumentsError}`;
	// In a recording without times too: the call itself says so.
	const dir = sessionWith([{ content: [cut, call("view", {})], outcomes: [{ error: said }, {}] }]);
	assert.deepEqual(sentAndSkipped(dir), [[["view"], ["open"]]]);
	assert.equal(loadRecording(dir).turns[0].blocked[0].reason, said.replace(/\s+/g, " "));
	// Whatever its result is: one that is no error, with a list of parts or (written by hand) without one; one with
	// a time in a recording with times; or none, noted as a call without a result.
	assert.deepEqual(sentAndSkipped(sessionWith([{ content: [cut, call("view", {})] }])), [[["view"], ["open"]]]);
	const bare = sessionWith([{ content: [cut, call("view", {})], outcomes: [{ raw: "opened" }, {}] }]);
	assert.deepEqual(sentAndSkipped(bare), [[["view"], ["open"]]]);
	assert.equal(loadRecording(bare).turns[0].blocked[0].reason, "its result says nothing");
	const clocked = { content: [cut, call("view", {})], outcomes: [{ error: said, ms: 3 }, { ms: 3 }], ms: 900 };
	assert.deepEqual(sentAndSkipped(sessionWith([clocked])), [[["view"], ["open"]]]);
	assert.deepEqual(loadRecording(sessionWith([{ content: [cut], outcomes: [{ none: true }] }])).turns[0].blocked, [
		{ name: "open", reason: NO_RESULT, unfinished: true },
	]);
});

test("the text of a result that is no error is not read for a call that is sent: a content that is no list of parts does not matter", () => {
	// A recording written by hand; pi writes a list of parts.
	const dir = sessionWith([
		{
			content: [call("move", { xyz: [1, 0, 0] }), call("grip", { v: 1 }), call("view", {})],
			outcomes: [{ raw: "moved" }, { raw: null }, { raw: undefined }],
		},
	]);
	assert.deepEqual(sentAndSkipped(dir), [[["move", "grip", "view"], []]]);
});

test("a call's result is its own turn's: a result under its id in another turn is not taken for it", () => {
	const first = call("move", { xyz: [1, 0, 0] });
	const again = { ...call("move", { xyz: [2, 0, 0] }), id: first.id };
	// The first ran; the second, under the same id, was refused.
	assert.deepEqual(
		sentAndSkipped(
			sessionWith([{ content: [first] }, { content: [again], outcomes: [{ error: "The episode is finished." }] }]),
		),
		[
			[["move"], []],
			[[], ["move"]],
		],
	);
	// The first has no result; the second's is not taken for it.
	const rec = loadRecording(sessionWith([{ content: [first], outcomes: [{ none: true }] }, { content: [again] }]));
	assert.deepEqual(
		rec.turns.map((t) => [t.calls.map((c) => c.arguments), t.blocked.map((b) => b.unfinished)]),
		[
			[[], [true]],
			[[{ xyz: [2, 0, 0] }], []],
		],
	);
	// The second has no result; the first's, a refusal in the turn before, is not taken for it.
	const late = loadRecording(
		sessionWith([
			{ content: [first], outcomes: [{ error: "The episode is finished." }] },
			{ content: [again], outcomes: [{ none: true }] },
		]),
	);
	assert.deepEqual(
		late.turns.map((t) => t.blocked.map((b) => [b.reason, b.unfinished])),
		[[["The episode is finished.", undefined]], [[NO_RESULT, true]]],
	);
	// Nor is a result that is not on the leaf's branch: the loader reads the leaf's branch, and on it the call has
	// no result.
	const dir = folder();
	const moved = call("move", { xyz: [4, 0, 0] });
	const lines = [
		{ type: "session", version: 3, id: "s", cwd: dir },
		{ type: "message", id: "u", parentId: null, message: { role: "user", content: "Solve." } },
		assistant("a", "u", [moved]),
		{
			type: "message",
			id: "r",
			parentId: "a",
			message: { role: "toolResult", toolCallId: moved.id, toolName: "move", content: [], isError: false },
		},
		{ type: "message", id: "u2", parentId: "a", message: { role: "user", content: "Go on." } },
	];
	writeFileSync(join(dir, "2026-01-01T00-00-00-000Z_s.jsonl"), `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	assert.deepEqual(loadRecording(dir).turns[0].blocked, [{ name: "move", reason: NO_RESULT, unfinished: true }]);
	// An entry a hook wrote before a call's result (the approval gate's as the call was prepared, the closed-loop
	// gate's as the result came in) sits between a reply and its results: it ends nothing.
	const gated = folder();
	const [approved, gripped] = [call("move", { xyz: [5, 0, 0] }), call("grip", { v: 1 })];
	const result = (id: string, parentId: string, c: { id: string; name: string }, error?: string) => ({
		type: "message",
		id,
		parentId,
		message: {
			role: "toolResult",
			toolCallId: c.id,
			toolName: c.name,
			content: [{ type: "text", text: error ?? "{}" }],
			isError: error !== undefined,
		},
	});
	const between = [
		{ type: "session", version: 3, id: "s", cwd: gated },
		{ type: "message", id: "u", parentId: null, message: { role: "user", content: "Solve." } },
		assistant("a", "u", [approved, gripped]),
		{ type: "custom", id: "c1", parentId: "a", customType: "approval", data: { decision: "approve" } },
		result("r1", "c1", approved),
		{ type: "custom", id: "c2", parentId: "r1", customType: "closed_loop", data: { kind: "unknown_outcome" } },
		result("r2", "c2", gripped, "robot error: the call timed out"),
	];
	writeFileSync(
		join(gated, "2026-01-01T00-00-00-000Z_s.jsonl"),
		`${between.map((l) => JSON.stringify(l)).join("\n")}\n`,
	);
	assert.deepEqual(
		loadRecording(gated).turns.map((t) => [t.calls.map((c) => [c.name, c.error]), t.blocked]),
		[
			[
				[
					["move", undefined],
					["grip", "robot error: the call timed out"],
				],
				[],
			],
		],
	);
});

test("in a recording without times, a call whose result is an error is also not sent when the last result under its id is a listed refusal", () => {
	const first = call("move", { xyz: [1, 0, 0] });
	const again = { ...call("move", { xyz: [2, 0, 0] }), id: first.id };
	const denied = "move was not approved by the operator: the operator declined this motion. It did not run.";
	const over = "The episode is finished.";
	// In a later turn and in the same reply. The note of each call is its own result's.
	const twice = [
		{ content: [first], outcomes: [{ error: denied }] },
		{ content: [again], outcomes: [{ error: over }] },
	];
	assert.deepEqual(sentAndSkipped(sessionWith(twice)), [
		[[], ["move"]],
		[[], ["move"]],
	]);
	assert.deepEqual(
		loadRecording(sessionWith(twice)).turns.map((t) => t.blocked.map((b) => b.reason)),
		[[denied], [over]],
	);
	assert.deepEqual(
		sentAndSkipped(sessionWith([{ content: [first, again], outcomes: [{ error: denied }, { error: over }] }])),
		[[[], ["move", "move"]]],
	);
	// Not in a recording with pi's times: there the first call's result has one, so its tool ran and failed.
	assert.deepEqual(
		sentAndSkipped(
			sessionWith([
				{ content: [first], outcomes: [{ error: denied, ms: 3 }] },
				{ content: [again], outcomes: [{ error: over }] },
			]),
		),
		[
			[["move"], []],
			[[], ["move"]],
		],
	);
	// Nor when the last result under the id is no error, whatever it says.
	assert.deepEqual(
		sentAndSkipped(
			sessionWith([
				{ content: [first], outcomes: [{ error: "gripper jammed" }] },
				{ content: [again], outcomes: [{ raw: [{ type: "text", text: over }] }] },
			]),
		),
		[
			[["move"], []],
			[["move"], []],
		],
	);
	// The id's last result, not its last error: here a call that ran comes behind the refusal, and the call that
	// failed before the refusal is sent. And the call's own id, whatever the tool: a refusal of the same tool under
	// another id does not touch the call, a refusal of another tool under its id does.
	const failed = { content: [first], outcomes: [{ error: "gripper jammed" }] };
	const third = { ...call("move", { xyz: [3, 0, 0] }), id: first.id };
	assert.deepEqual(
		sentAndSkipped(sessionWith([failed, { content: [again], outcomes: [{ error: over }] }, { content: [third] }])),
		[
			[["move"], []],
			[[], ["move"]],
			[["move"], []],
		],
	);
	const other = call("move", { xyz: [2, 0, 0] });
	assert.deepEqual(sentAndSkipped(sessionWith([failed, { content: [other], outcomes: [{ error: over }] }])), [
		[["move"], []],
		[[], ["move"]],
	]);
	const grip = { ...call("grip", { v: 1 }), id: first.id };
	assert.deepEqual(sentAndSkipped(sessionWith([twice[0], { content: [grip], outcomes: [{ error: over }] }])), [
		[[], ["move"]],
		[[], ["grip"]],
	]);
	// The last result's text is read as a call's own is: with white space around it, and a refusal that names a
	// tool counts for a call of that tool only.
	const ungranted = (tool: string) => `${tool} is not one of the tools arm gives the planner.`;
	const under = (name: string) => ({
		content: [{ ...call(name, {}), id: first.id }],
		outcomes: [{ error: ungranted(name) }],
	});
	assert.deepEqual(
		[
			[twice[0], { content: [again], outcomes: [{ error: `  ${over}\n` }] }],
			[twice[0], under("move")],
			[failed, under("codemode")],
		].map((turns) => sentAndSkipped(sessionWith(turns))),
		[
			[
				[[], ["move"]],
				[[], ["move"]],
			],
			[
				[[], ["move"]],
				[[], ["move"]],
			],
			[
				[["move"], []],
				[[], ["codemode"]],
			],
		],
	);
	// Every call of the id whose result is an error, one that says nothing too, and whatever the id is: here the
	// empty one of a server that sends none.
	const bare = [first, again, third].map((c) => ({ ...c, id: "" }));
	const all = sessionWith([{ content: bare, outcomes: [{ error: "" }, { error: denied }, { error: over }] }]);
	assert.deepEqual(
		loadRecording(all).turns[0].blocked.map((b) => b.reason),
		["its result says nothing", denied, over],
	);
});

test("calls of one reply that share an id take that id's results in the order of the calls", () => {
	// A server whose stream carries no tool-call ids: pi stores "" for every call of the reply and writes their
	// results under that id, in the order of the calls.
	const moves = [1, 2, 3].map((x) => ({ ...call("move", { xyz: [x, 0, 0] }), id: "" }));
	const own = call("grip", { v: 1 });
	const read = (turn: Parameters<typeof sessionWith>[0][number]) => {
		const t = loadRecording(sessionWith([turn])).turns[0];
		return { sent: t.calls.map((c) => [c.arguments.xyz ?? c.name, c.error]), blocked: t.blocked };
	};
	// The run was aborted after the first move: the other two have no result, and the first's is not theirs.
	assert.deepEqual(read({ content: moves, outcomes: [{ ms: 5 }, { none: true }, { none: true }], ms: 900 }), {
		sent: [[[1, 0, 0], undefined]],
		blocked: [
			{ name: "move", reason: NO_RESULT, unfinished: true },
			{ name: "move", reason: NO_RESULT, unfinished: true },
		],
	});
	// Whatever the shared id is, and behind the reply's own text.
	for (const id of ["", "call_0"]) {
		const twins = [1, 2].map((x) => ({ ...call("move", { xyz: [x, 0, 0] }), id }));
		const content = [{ type: "text", text: "two moves" }, ...twins];
		assert.deepEqual(read({ content, outcomes: [{ ms: 5 }, { none: true }], ms: 900 }), {
			sent: [[[1, 0, 0], undefined]],
			blocked: [{ name: "move", reason: NO_RESULT, unfinished: true }],
		});
	}
	// Each call reads its own result, a call with an id of its own between them too: the first ran, the second
	// was refused, the third failed inside its tool. With pi's times, and by the texts alone.
	for (const ms of [5, undefined]) {
		assert.deepEqual(
			read({
				content: [moves[0], moves[1], own, moves[2]],
				outcomes: [{ ms }, { error: "The episode is finished." }, { ms }, { error: "gripper jammed", ms }],
			}),
			{
				sent: [
					[[1, 0, 0], undefined],
					["grip", undefined],
					[[3, 0, 0], "gripper jammed"],
				],
				blocked: [{ name: "move", reason: "The episode is finished." }],
			},
		);
	}
	// The first was refused with words the list does not hold; the two behind it ran and are sent.
	const denied = "move was not approved by the operator: the operator declined this motion. It did not run.";
	assert.deepEqual(read({ content: moves, outcomes: [{ error: denied }, { ms: 5 }, { ms: 5 }], ms: 900 }), {
		sent: [
			[[2, 0, 0], undefined],
			[[3, 0, 0], undefined],
		],
		blocked: [{ name: "move", reason: denied }],
	});
});

test("the texts of calls that never ran, read in a recording without times: each one whole, with white space around it, after other text and before other text", () => {
	// [the call's tool, the text, the pattern is tied to the text's start, to its end]
	const TEXTS: [string, string, boolean, boolean][] = [
		[
			"move",
			"replay stopped (grip failed: jammed; it succeeded in the recording); this recorded call was not sent",
			true,
			true,
		],
		["move", "Tool execution was blocked", true, true],
		["move", "Operation aborted", true, true],
		["nosuch", "Tool nosuch not found", true, true],
		["move", 'Tool call "move" was not executed: the response hit the output token limit', true, false],
		["move", "arm is not available.", true, true],
		["move", "The robot failed: env.step: timed out. The episode is over.", true, true],
		["move", "The episode is finished.", true, true],
		["move", "Planner tool_calls budget exhausted; the episode is over.", true, true],
		["move", "operator submitted a terminal verdict; the run is closing", true, false],
		["move", "operator aborted this run; finish without further motion", true, false],
		["move", "refused; request_scene_reset and obtain operator confirmation first", true, false],
		[
			"finish",
			"finish refused: the verifier judged the task NOT complete (the cube is beside the bowl).",
			true,
			false,
		],
		["move", STALE, false, true],
		["grep", "file access is disabled: the memory guard has no memory, output dir or cell", true, false],
		["read", "file access denied: /etc/passwd cannot be named", true, false],
		["codemode", "codemode is not one of the tools arm gives the planner.", true, true],
		["move", 'move has no argument "step_m"; it takes xyz, gripper.', true, true],
	];
	/** Whether a call of `tool` whose error result says `error` is read as one that never ran. */
	const skipped = (tool: string, error: string) =>
		sentAndSkipped(sessionWith([{ content: [call(tool, {})], outcomes: [{ error }] }]))[0][1].length === 1;
	for (const [tool, said, start, end] of TEXTS) {
		assert.equal(skipped(tool, said), true, said);
		assert.equal(skipped(tool, `  ${said}\n`), true, `with white space around it: ${said}`);
		assert.equal(skipped(tool, `robot error: ${said}`), !start, `after other text: ${said}`);
		assert.equal(skipped(tool, `${said} (again)`), !end, `before other text: ${said}`);
	}
});
