// The replay extension as the planner of a real pi session (pi's agent loop, extension runner and tools), over the
// robot base with a toy robot (on robosuite's manifest, or in units mode): what a recorded turn does when its calls
// end as recorded, and when one does not; which calls of a session pi itself recorded reach the robot again; and what
// a request to the replay's model gets that is not a turn of the session's own loop.

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import replay, { loadRecording, REPLAY_ENTRY } from "../src/capabilities/replay/index.ts";
import { auxModel } from "../src/infra/config.ts";
import type { UnitsSpec } from "../src/modes/units/index.ts";
import { askVlm } from "../src/modes/units/vlm.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";
import { useDeployment } from "./helpers/deployment.ts";
import { ask, at, llm, PNG, reply, type Session, said, vlm, withSession } from "./helpers/pi-session.ts";

const FINISH = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p: { status: string }) => ({ content: [{ type: "text" as const, text: p.status }], details: p }),
};

type Recorded = { name: string; arguments: Record<string, unknown>; error?: string };
const move = (x: number, more: Record<string, unknown> = {}, error?: string): Recorded => ({
	name: "move_to",
	arguments: { xyz: [x, 0, 1], ...more },
	...(error === undefined ? {} : { error }),
});
const finish: Recorded = { name: "finish", arguments: { status: "success", summary: "done" } };
/** The toy robot cannot reach x = 9. */
const IK = "robot error: IK failed";
const TAKES = "it takes xyz, quat_xyzw, rotvec, gripper, tol_m, max_steps.";

type RecordedTurn = Recorded[] | { text: string; calls: Recorded[] };
/**
 * A recorded session of `turns` (each a batch of calls with their results, and the reply's text when given), as pi
 * writes one; `task` is its robot_task entry.
 */
function record(dir: string, turns: RecordedTurn[], task?: object) {
	const lines: unknown[] = [{ type: "session", version: 3, id: "s", cwd: dir }];
	if (task) lines.push({ type: "custom", id: "t", parentId: null, customType: "robot_task", data: task });
	lines.push({ type: "message", id: "u", parentId: task ? "t" : null, message: { role: "user", content: "Solve." } });
	let leaf = "u";
	turns.forEach((turn, i) => {
		const calls = Array.isArray(turn) ? turn : turn.calls;
		const content = [
			...(Array.isArray(turn) ? [] : [{ type: "text", text: turn.text }]),
			...calls.map((c, k) => ({ type: "toolCall", id: `rec_${i}_${k}`, name: c.name, arguments: c.arguments })),
		];
		lines.push({
			type: "message",
			id: `a${i}`,
			parentId: leaf,
			message: { role: "assistant", content, stopReason: "toolUse" },
		});
		leaf = `a${i}`;
		calls.forEach((c, k) => {
			lines.push({
				type: "message",
				id: `a${i}_r${k}`,
				parentId: leaf,
				message: {
					role: "toolResult",
					toolCallId: `rec_${i}_${k}`,
					toolName: c.name,
					content: [{ type: "text", text: c.error ?? "moved" }],
					isError: c.error !== undefined,
				},
			});
			leaf = `a${i}_r${k}`;
		});
	});
	const file = join(dir, "recorded.jsonl");
	writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
	return file;
}

/** What the toy's check_attached fails with when a test makes it fail. */
const NO_FRAMES = "robot error: the frames could not be rendered";

/**
 * The toy robot; `reached` gets every call its move_to ran. With `asks` it also has check_attached, which asks a
 * model while it runs as the real tool does (../src/primitives/grasp.ts: `aux.attach`, else --aux-model, else the
 * session's model) and returns the answer; with "then fails" it fails after asking. Every move returns `views` camera
 * images, and `spec` is added to the robot's.
 */
const toy =
	(reached: unknown[], o: { asks?: true | "then fails"; views?: number; spec?: Partial<RobotSpec> } = {}) =>
	(pi: ExtensionAPI) => {
		const r = defineRobot(pi, {
			name: "toy",
			task: [],
			keepImages: 1,
			manifest: "robosuite",
			vars: () => ({ arms: [] }),
			capabilities: (c) => o.asks !== undefined && c === "grasp",
			start: async () => ["move_to", ...(o.asks ? ["check_attached"] : []), "finish"],
			result: () => ({}),
			status: () => ({ step: 0 }),
			finish: FINISH,
			...o.spec,
		});
		r.primitive("move_to", async (p: { xyz: number[] }) => {
			reached.push(p);
			if (p.xyz[0] === 9) throw new Error(IK);
			return {
				content: [
					{ type: "text", text: "moved" },
					...Array.from({ length: o.views ?? 0 }, () => ({
						type: "image" as const,
						data: PNG,
						mimeType: "image/png",
					})),
				],
				details: {},
			};
		});
		if (!o.asks) return;
		r.primitive("check_attached", async (_p, signal, ctx) => {
			const answer = await said(() =>
				askVlm(ctx, auxModel(pi, "attach"), pi.getThinkingLevel(), "Held?", [], signal),
			);
			if (o.asks === "then fails") throw new Error(NO_FRAMES);
			return { content: [{ type: "text", text: `asked: ${answer}` }], details: {} };
		});
	};

/**
 * Replay `file` with `-e robot -e replay --model replay/session --replay <file>`: what reached the robot's tool,
 * each call's result, and what the replay said.
 */
async function replayedFile(base: string, file: string) {
	const reached: unknown[] = [];
	// The replay extension is loaded after the robot, as `-e robot -e replay` loads them.
	return withSession(
		base,
		[toy(reached), replay],
		async ({ session, runtime }) => {
			const planner = runtime.getModel("replay", "session");
			assert.ok(planner, "the replay provider's model is registered");
			await session.setModel(planner);
			await session.prompt("Replay the session.");
			const texts = (m: { content: { type: string; text?: string }[] }) =>
				m.content.map((c) => (c.type === "text" ? c.text : "")).join("");
			return {
				results: session.messages.flatMap((m) =>
					m.role === "toolResult" ? [`${m.toolName}${m.isError ? " ERROR" : ""}: ${texts(m)}`] : [],
				),
				reached,
				said: session.messages.flatMap((m) => (m.role === "assistant" ? texts(m).split("\n") : [])),
			};
		},
		{ flags: { replay: file } },
	);
}

/** Replay `turns`, a session written by hand: what reached the robot's tool and each call's result. */
async function replayed(turns: Recorded[][]) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "replay-session-")));
	try {
		const { results, reached } = await replayedFile(base, record(base, turns));
		return { results, reached };
	} finally {
		rmSync(base, { recursive: true, force: true });
	}
}

test("a real session: a recorded turn whose calls end as recorded is sent whole, a failure the recording also had included", async () => {
	assert.deepEqual(await replayed([[move(0.5), move(0.6)], [finish]]), {
		results: ["move_to: moved", "move_to: moved", "finish: success"],
		reached: [{ xyz: [0.5, 0, 1] }, { xyz: [0.6, 0, 1] }],
	});
	assert.deepEqual(await replayed([[move(9, {}, IK), move(0.6)], [finish]]), {
		results: [`move_to ERROR: ${IK}`, "move_to: moved", "finish: success"],
		reached: [{ xyz: [9, 0, 1] }, { xyz: [0.6, 0, 1] }],
	});
});

test("a real session: a recorded call the robot refuses now stops the replay at that call, finish included", async () => {
	// The recording's move_to ran with step_m, a program's option its tool no longer takes.
	const refusal = `move_to has no argument "step_m"; ${TAKES}`;
	const withheld = `replay stopped (move_to failed: ${refusal}; it succeeded in the recording); this recorded call was not sent`;
	assert.deepEqual(await replayed([[move(0.5, { step_m: 0.005 }), move(0.6)], [finish]]), {
		results: [`move_to ERROR: ${refusal}`, `move_to ERROR: ${withheld}`],
		reached: [],
	});
	assert.deepEqual(await replayed([[move(0.5, { step_m: 0.005 }), finish]]), {
		results: [`move_to ERROR: ${refusal}`, `finish ERROR: ${withheld}`],
		reached: [],
	});
});

test("a real session: a recorded call whose outcome differs now stops the replay at that call: a robot tool's turn runs one call at a time", async () => {
	// It fails where the recording succeeded.
	assert.deepEqual(await replayed([[move(9), move(0.6)], [finish]]), {
		results: [
			`move_to ERROR: ${IK}`,
			`move_to ERROR: replay stopped (move_to failed: ${IK}; it succeeded in the recording); this recorded call was not sent`,
		],
		reached: [{ xyz: [9, 0, 1] }],
	});
	// It succeeds where the recording failed.
	assert.deepEqual(await replayed([[move(0.5, {}, IK), move(0.6)], [finish]]), {
		results: [
			"move_to: moved",
			`move_to ERROR: replay stopped (move_to succeeded; it failed in the recording (${IK})); this recorded call was not sent`,
		],
		reached: [{ xyz: [0.5, 0, 1] }],
	});
});

test("a real session: of a session pi recorded, a call a hook refused and the calls an abort left without a result do not reach the robot again", async () => {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "replay-session-")));
	// The approval gate's refusal (../src/capabilities/operator.ts), a text the replay's list does not hold.
	const denied = "move_to was not approved by the operator: the operator declined this motion. It did not run.";
	const moved = (x: number, id: string) => fauxToolCall("move_to", { xyz: [x, 0, 1] }, { id });
	try {
		// The recording: a robot, a gate that refuses the move to x = 0.7 in the approval gate's words, and an abort
		// once the move to x = 0.8 has ended, which leaves the move to x = 0.9 and finish without a result.
		const ran: unknown[] = [];
		const file = await withSession(
			base,
			[
				toy(ran),
				(pi) => {
					// As the approval gate does, an entry for every call it judged, written before the call's result.
					pi.on("tool_call", (event) => {
						const no = event.toolName === "move_to" && (event.input as { xyz: number[] }).xyz[0] === 0.7;
						pi.appendEntry("approval", { tool: event.toolName, decision: no ? "reject" : "approve" });
						return no ? { block: true, reason: denied } : undefined;
					});
				},
			],
			async ({ session, faux }) => {
				session.subscribe((event) => {
					if (event.type === "tool_execution_end" && event.toolCallId === "cut") void session.abort();
				});
				faux.setResponses([
					fauxAssistantMessage([moved(0.5, "a"), moved(0.7, "b"), moved(0.6, "c")], { stopReason: "toolUse" }),
					fauxAssistantMessage(
						[
							moved(0.8, "cut"),
							moved(0.9, "d"),
							fauxToolCall("finish", { status: "success", summary: "done" }, { id: "e" }),
						],
						{ stopReason: "toolUse" },
					),
				]);
				await session.prompt("Solve.");
				const path = join(base, "recorded.jsonl");
				const lines = [session.sessionManager.getHeader(), ...session.sessionManager.getEntries()];
				writeFileSync(path, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
				return path;
			},
		);
		assert.deepEqual(ran, [{ xyz: [0.5, 0, 1] }, { xyz: [0.6, 0, 1] }, { xyz: [0.8, 0, 1] }]);
		// pi's own record: the hook's entry sits between a reply and the result of each call it judged,
		assert.deepEqual(
			readFileSync(file, "utf8")
				.trim()
				.split("\n")
				.map((line) => JSON.parse(line))
				.flatMap((e) =>
					e.type === "custom" && e.customType === "approval"
						? ["approval"]
						: e.type === "message" && (e.message.role === "assistant" || e.message.role === "toolResult")
							? [e.message.toolCallId ?? "reply"]
							: [],
				),
			["reply", "approval", "a", "approval", "b", "approval", "c", "reply", "approval", "cut"],
		);
		// the refused call has an error result without a time, and the two cut calls have none.
		const NO_RESULT = "the recording holds no result for it";
		assert.deepEqual(
			loadRecording(file).turns.map((t) => [t.calls.map((c) => c.arguments.xyz), t.blocked]),
			[
				[
					[
						[0.5, 0, 1],
						[0.6, 0, 1],
					],
					[{ name: "move_to", reason: denied }],
				],
				[
					[[0.8, 0, 1]],
					[
						{ name: "move_to", reason: NO_RESULT, unfinished: true },
						{ name: "finish", reason: NO_RESULT, unfinished: true },
					],
				],
			],
		);
		// The replay, with no gate now: the three moves that ran, and no other.
		const again = await replayedFile(base, file);
		assert.deepEqual(again.reached, ran);
		assert.deepEqual(again.results, ["move_to: moved", "move_to: moved", "move_to: moved"]);
		for (const note of [
			`replay: skipped move_to (it never ran in the recording: ${denied})`,
			`replay: skipped move_to (${NO_RESULT})`,
			`replay: skipped finish (${NO_RESULT})`,
			"Replay complete: the recording has no finish; the episode is reported at shutdown.",
		])
			assert.ok(again.said.includes(note), `${note}\n${again.said.join("\n")}`);
	} finally {
		rmSync(base, { recursive: true, force: true });
	}
});

/** What a request to the replay's model gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"replay/session replays recorded turns and answers no other request; --aux-model names a model for the checks that take one";
const stopped = (n: number, of: number) =>
	`Replay stopped: a call's outcome diverged from the recording's; no further recorded motion is sent (${n} of ${of} turns replayed).`;
/** A recorded call of the toy's check_attached. */
const check: Recorded = { name: "check_attached", arguments: { object: "cube" } };

/**
 * Replay `turns` with `-e robot -e replay [-e more] --model replay/session --replay <file>` and `flags`. `before` runs
 * once the session's model is replay/session and before the prompt, with the context pi gives an extension; `aux` are
 * the answers of the scripted model, which is asked only when a flag or the deployment names it. Returns the moves
 * that reached the robot, each call's result, what the replay said (the session's replies and its stderr lines), the
 * session's entries of a type, and the robot's result row (written at pi's quit when the episode did not finish).
 */
async function replaying(
	turns: RecordedTurn[],
	o: {
		task?: object;
		robot?: (reached: unknown[]) => (pi: ExtensionAPI) => void;
		flags?: Record<string, string | boolean>;
		aux?: string[];
		before?: (ctx: ExtensionContext, session: Session) => Promise<void>;
		more?: ((pi: ExtensionAPI) => void)[];
	} = {},
) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "replay-session-")));
	const reached: unknown[] = [];
	let ctx: ExtensionContext | undefined;
	try {
		const file = record(base, turns, o.task);
		return await withSession(
			base,
			[
				(o.robot ?? toy)(reached),
				replay,
				...(o.more ?? []),
				(pi) => {
					pi.on("session_start", (_event, c) => {
						ctx = c;
					});
				},
			],
			async ({ session, faux, runtime, stderr }) => {
				faux.appendResponses((o.aux ?? []).map((text) => fauxAssistantMessage(text)));
				const planner = runtime.getModel("replay", "session");
				assert.ok(planner, "the replay provider's model is registered");
				await session.setModel(planner);
				await o.before?.(ctx as ExtensionContext, session);
				await session.prompt("Replay the session.");
				await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
				const texts = (m: { content: { type: string; text?: string }[] }) =>
					m.content.map((c) => (c.type === "text" ? c.text : "")).join("");
				const entries = (type: string) =>
					(session.sessionManager.getEntries() as { type: string; customType?: string; data?: any }[]).flatMap(
						(e) => (e.type === "custom" && e.customType === type ? [e.data] : []),
					);
				return {
					file,
					reached: reached.map((p) => (p as { xyz: number[] }).xyz[0]),
					results: session.messages.flatMap((m) =>
						m.role === "toolResult" ? [`${m.toolName}${m.isError ? " ERROR" : ""}: ${texts(m)}`] : [],
					),
					said: session.messages.flatMap((m) => (m.role === "assistant" ? texts(m).split("\n") : [])),
					stderr: stderr.filter((l) => l.startsWith("[replay]")),
					entries,
					row: entries(RESULT_ENTRY).at(-1) as Record<string, unknown> | undefined,
				};
			},
			{ flags: { replay: file, ...o.flags } },
		);
	} finally {
		rmSync(base, { recursive: true, force: true });
	}
}

test("a real session: a request before the session's first turn (the dashboard's model check, a question to a VLM) does not start the replay or take its first turn", async () => {
	const answers: string[] = [];
	let sources = -1;
	const out = await replaying([{ text: "I move above the cube.", calls: [move(0.5)] }, [move(0.6)], [finish]], {
		// The recorded task differs from this run's: the replay warns of that when the session's first turn starts.
		task: { robot: "toy", seed: "1" },
		before: async (ctx, session) => {
			answers.push(await llm(ctx), await vlm(ctx));
			sources = (session.sessionManager.getEntries() as { type: string; customType?: string }[]).filter(
				(e) => e.type === "custom" && e.customType === REPLAY_ENTRY,
			).length;
		},
	});
	assert.deepEqual(answers, [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`]);
	assert.deepEqual(out.reached, [0.5, 0.6]);
	assert.deepEqual(out.results, ["move_to: moved", "move_to: moved", "finish: success"]);
	// The replay starts with the session's own first turn: its entry, its opening note, the warning.
	assert.equal(sources, 0);
	assert.equal(out.entries(REPLAY_ENTRY).length, 1);
	assert.deepEqual(out.said.slice(0, 2), [`replay: 3 recorded turns from ${out.file}`, "I move above the cube."]);
	assert.deepEqual(out.stderr, [
		"[replay] this run's task differs from the recorded robot_task: seed: 1 -> (none)",
		`[replay] 3 recorded turns from ${out.file}`,
	]);
	assert.equal(out.row?.claimed, "success");
});

test("a real session: a recorded tool that asks the session's model while it runs gets no recorded turn for an answer, and the next turn is sent", async () => {
	const asking = (reached: unknown[]) => toy(reached, { asks: true });
	const out = await replaying([[check], { text: "Now to the bowl.", calls: [move(0.5)] }, [move(0.6)], [finish]], {
		robot: asking,
	});
	assert.deepEqual(out.reached, [0.5, 0.6]);
	assert.deepEqual(out.results, [
		`check_attached: asked: error: ${NOT_A_TURN}`,
		"move_to: moved",
		"move_to: moved",
		"finish: success",
	]);
	assert.ok(out.said.includes("Now to the bowl."));
	// The replay has nothing to note: every call it sent has its result.
	assert.deepEqual(out.stderr, [`[replay] 4 recorded turns from ${out.file}`]);
	// With a model named for the question, that model answers it.
	const named = await replaying([[check], [move(0.5)], [finish]], {
		robot: asking,
		flags: { "aux-model": "faux/faux-1" },
		aux: ["held"],
	});
	assert.deepEqual(named.results, ["check_attached: asked: held", "move_to: moved", "finish: success"]);
	// The deployment's own role names that model too.
	const role = await replaying([[check], [move(0.5)], [finish]], {
		robot: asking,
		aux: ["held"],
		before: async () => void useDeployment({ aux: { attach: "faux/faux-1" } }),
	});
	assert.deepEqual(role.results, ["check_attached: asked: held", "move_to: moved", "finish: success"]);
});

test("a real session: a request while a recorded call runs, or behind it, does not make the replay forget that call: when it diverges, the rest of its turn is withheld and the replay stops", async () => {
	const failing = (reached: unknown[]) => toy(reached, { asks: "then fails" });
	const turns = [[check, move(0.7)], [move(0.5)], [finish]];
	const answers: string[] = [];
	let session: Session | undefined;
	// Behind the failed call: a question to a VLM, a request under another id, and pi's own summary of the session.
	const behind: ((ctx: ExtensionContext) => Promise<string>)[] = [
		vlm,
		(ctx) => reply(ctx, { messages: [ask] }, { sessionId: "0193e1c0-0000-7000-8000-000000000000" }),
		() =>
			(session as Session)
				.summarizeForBugReport({ signal: new AbortController().signal })
				.catch((err: Error) => `threw: ${err.message}`),
	];
	for (const more of [
		[],
		...behind.map((asked) => [at("tool_execution_end", 1, async (ctx) => void answers.push(await asked(ctx)))]),
	]) {
		const out = await replaying(turns, {
			robot: failing,
			more,
			before: async (_ctx, s) => {
				session = s;
			},
		});
		assert.deepEqual(out.reached, []);
		assert.deepEqual(out.results, [
			`check_attached ERROR: ${NO_FRAMES}`,
			`move_to ERROR: replay stopped (check_attached failed: ${NO_FRAMES}; it succeeded in the recording); this recorded call was not sent`,
		]);
		assert.equal(out.said.at(-1), stopped(1, 3));
		assert.equal(out.row?.claimed, null);
	}
	assert.deepEqual(answers, [
		`error: ${NOT_A_TURN}`,
		`error: ${NOT_A_TURN}`,
		`threw: Bug report summary failed: ${NOT_A_TURN}`,
	]);
});

test("a real session: a model check between two turns does not make the replay forget a call that diverged, or take the next turn", async () => {
	const answers: string[] = [];
	const checking = () => at("turn_end", 1, async (ctx) => void answers.push(await llm(ctx)));
	// The first call diverges: the replay stops there.
	const diverged = await replaying([[move(9)], [move(0.5)], [finish]], { more: [checking()] });
	assert.deepEqual(diverged.reached, [9]);
	assert.deepEqual(diverged.results, [`move_to ERROR: ${IK}`]);
	assert.equal(diverged.said.at(-1), stopped(1, 3));
	assert.equal(diverged.row?.claimed, null);
	// Nothing diverges: the turn behind the check is the recording's next.
	const whole = await replaying([[move(0.4)], { text: "Now to the bowl.", calls: [move(0.5)] }, [finish]], {
		more: [checking()],
	});
	assert.deepEqual(whole.reached, [0.4, 0.5]);
	assert.ok(whole.said.includes("Now to the bowl."));
	assert.deepEqual(answers, [`failed: ${NOT_A_TURN}`, `failed: ${NOT_A_TURN}`]);
});

test("a real session: a request from another extension's context handler, right before the session's own, does not take the turn; nor does one when the replay is over", async () => {
	const answers: string[] = [];
	// pi hands the transcript to the extensions' context handlers shortly before each request of its loop; this one
	// is loaded after the replay.
	const out = await replaying([[move(0.5)], { text: "Now to the bowl.", calls: [move(0.6)] }, [finish]], {
		more: [
			at("context", 2, async (ctx) => void answers.push(await vlm(ctx))),
			at("agent_end", 1, async (ctx) => void answers.push(await vlm(ctx))),
		],
	});
	assert.deepEqual(out.reached, [0.5, 0.6]);
	assert.deepEqual(out.results, ["move_to: moved", "move_to: moved", "finish: success"]);
	assert.deepEqual(answers, [`error: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`]);
	// The replay printed its opening note and nothing for the two requests.
	assert.deepEqual(out.stderr, [`[replay] 3 recorded turns from ${out.file}`]);
});

test("a real session: a request under another id, one that declares tools of its own, and pi's own summary of the session do not take a turn", async () => {
	const answers: unknown[] = [];
	let session: Session | undefined;
	const out = await replaying(
		[[move(0.4)], { text: "Now to the bowl.", calls: [move(0.5)] }, [move(0.6)], [move(0.7)], [finish]],
		{
			before: async (_ctx, s) => {
				session = s;
			},
			more: [
				// pi's compaction and branch summaries go out under an id of their own.
				at("turn_end", 1, async (ctx) =>
					answers.push(
						await reply(ctx, { messages: [ask] }, { sessionId: "0193e1c0-0000-7000-8000-000000000000" }),
					),
				),
				// A tool's own agent loop declares its tools, as the session's loop does.
				at("turn_end", 2, async (ctx) =>
					answers.push(
						await reply(ctx, {
							systemPrompt: "You are a policy.",
							tools: [{ name: "policy_act", description: "act", parameters: Type.Object({}) }],
							messages: [ask],
						}),
					),
				),
				// pi's summary for a bug report is asked under the session's own id.
				at("turn_end", 3, async () =>
					answers.push(
						await (session as Session)
							.summarizeForBugReport({ signal: new AbortController().signal })
							.catch((err: Error) => `threw: ${err.message}`),
					),
				),
			],
		},
	);
	assert.deepEqual(out.reached, [0.4, 0.5, 0.6, 0.7]);
	assert.deepEqual(out.results, [
		"move_to: moved",
		"move_to: moved",
		"move_to: moved",
		"move_to: moved",
		"finish: success",
	]);
	assert.deepEqual(answers, [
		`error: ${NOT_A_TURN}`,
		`error: ${NOT_A_TURN}`,
		`threw: Bug report summary failed: ${NOT_A_TURN}`,
	]);
});

const VECTORS: UnitsSpec["vectors"] = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
const act = (unit: string): Recorded => ({ name: "act", arguments: { unit } });
/** The flags of a robot in pure units mode with the finish verifier on (../src/modes/units). */
const VERIFIED = {
	units: "true",
	"units-plugins": "recovery,auto_release,proprioception,plan,mem_text",
	"units-verify": "true",
};
/** A toy arm in the real units mode with its real finish verifier; with `blocked` a forward move fails. */
const unitsToy =
	(o: { blocked?: boolean } = {}) =>
	() =>
	(pi: ExtensionAPI) => {
		const pos = [0.5, 0, 0.2];
		let width = 0.08;
		defineRobot(pi, {
			name: "toy",
			task: [],
			keepImages: 2,
			start: async () => ["finish"],
			result: () => ({}),
			finish: FINISH,
			units: {
				vectors: VECTORS,
				stepM: 0.02,
				apply: async (m) => {
					if (m.delta[0] > 0 && o.blocked) throw new Error("robot error: blocked");
					for (let i = 0; i < 3; i++) pos[i] += m.delta[i];
					if (m.gripper === "open") width = 0.08;
					if (m.gripper === "close") width = 0.03;
					return {
						content: [
							{ type: "text", text: "obs" },
							{ type: "image", data: PNG, mimeType: "image/png" },
						],
						details: {},
					};
				},
				state: async () => ({ eef_xyz: [...pos], gripper_width: width, table_z: 0 }),
				instruction: () => "put the cube in the bowl",
				emptyWidthM: 0.005,
			},
		});
	};

test("a real session: the units verifier's question at a recorded finish is not answered by the replay: without a model named for it the finish is refused, with one it is checked, and with the verifier off nothing is asked", async () => {
	const refusal = `finish refused: the verifier is unavailable (${NOT_A_TURN}), so the task could not be checked. Call \`finish\` again to retry the check. (This is not the verifier's NOT complete verdict.)`;
	const out = await replaying([[act("MV_LEFT")], [finish]], { robot: unitsToy(), flags: VERIFIED });
	assert.equal(out.results.at(-1), `finish ERROR: ${refusal}`);
	// What the replay says of the stop: the refusal, then its closing line.
	assert.deepEqual(out.said.slice(-2), [
		`replay: finish failed: ${refusal}; it succeeded in the recording`,
		stopped(2, 2),
	]);
	assert.equal(out.row?.claimed, null);
	// The row says why the finish was not verified, and holds no verdict.
	assert.equal(out.row?.verifier_error, NOT_A_TURN);
	assert.equal(out.row?.finish_verified, undefined);
	// The replay has no call without a result.
	assert.deepEqual(
		out.stderr.filter((l) => /no result/.test(l)),
		[],
	);
	const named = await replaying([[act("MV_LEFT")], [finish]], {
		robot: unitsToy(),
		flags: { ...VERIFIED, "aux-model": "faux/faux-1" },
		aux: ['{"complete": true, "reason": "the cube is in the bowl"}'],
	});
	assert.equal(named.results.at(-1), "finish: success");
	assert.equal(named.row?.claimed, "success");
	assert.equal(named.row?.finish_verified, true);
	// The deployment's own role names the verifier's model too.
	const role = await replaying([[act("MV_LEFT")], [finish]], {
		robot: unitsToy(),
		flags: VERIFIED,
		aux: ['{"complete": true, "reason": "the cube is in the bowl"}'],
		before: async () => void useDeployment({ aux: { verify: "faux/faux-1" } }),
	});
	assert.equal(role.results.at(-1), "finish: success");
	assert.equal(role.row?.claimed, "success");
	assert.equal(role.row?.finish_verified, true);
	// With the verifier off the finish is the recorded one, and no model is asked.
	const off = await replaying([[act("MV_LEFT")], [finish]], {
		robot: unitsToy(),
		flags: { ...VERIFIED, "units-verify": "false" },
	});
	assert.equal(off.results.at(-1), "finish: success");
	assert.equal(off.row?.claimed, "success");
	assert.equal(off.row?.finish_verified, undefined);
	assert.equal(off.row?.verifier_error, undefined);
	assert.deepEqual(off.entries("units_verify"), []);
});

test("a real session: with the units verifier on and no model named for it, a turn [act, finish] whose act diverges does not end as a claimed success", async () => {
	const out = await replaying([[act("MV_FWD"), finish]], { robot: unitsToy({ blocked: true }), flags: VERIFIED });
	assert.deepEqual(
		out.results.map((r) => r.split(":")[0]),
		["act ERROR", "finish ERROR"],
	);
	assert.equal(out.said.at(-1), stopped(1, 1));
	assert.equal(out.row?.claimed, null);
});

test("a real session: a reviewed approval's question is not answered by the replay: no motion is approved, and the replay stops at the first; with a model named for the reviewer that approves every motion, the replay runs whole", async () => {
	const out = await replaying([[move(0.5)], [move(0.6)], [move(0.7)], [move(0.8)], [finish]], {
		flags: { approval: "reviewed" },
	});
	assert.deepEqual(out.reached, []);
	assert.deepEqual(out.results, [
		`move_to ERROR: move_to was not approved by the reviewer (error): the reviewer failed: ${NOT_A_TURN}. It did not run.`,
	]);
	assert.equal(out.said.at(-1), stopped(1, 5));
	assert.equal(out.row?.claimed, null);
	// With a model named for the reviewer, by either flag, its decisions stand and the replay runs whole.
	const approve = '{"decision": "approve", "reason": "the move is safe"}';
	for (const flag of ["aux-model", "approval-model"]) {
		const named = await replaying([[move(0.5)], [move(0.6)], [finish]], {
			flags: { approval: "reviewed", [flag]: "faux/faux-1" },
			aux: [approve, approve],
		});
		assert.deepEqual(named.reached, [0.5, 0.6]);
		assert.deepEqual(named.results, ["move_to: moved", "move_to: moved", "finish: success"]);
		assert.equal(named.row?.claimed, "success");
	}
});

test("a real session: visual differencing's question is not answered by the replay: every recorded move reaches the robot, and each result it would have described says that it was not; with a model named for it, that model describes them", async () => {
	const out = await replaying([[move(0.5)], [move(0.6)], [move(0.7)], [finish]], {
		robot: (reached) => toy(reached, { views: 1, spec: { vdm: { views: 1 } } }),
		flags: { vdm: true },
	});
	assert.deepEqual(out.reached, [0.5, 0.6, 0.7]);
	assert.deepEqual(out.results, [
		...Array.from({ length: 3 }, () => `move_to: moved[visual differencing unavailable: ${NOT_A_TURN}]`),
		"finish: success",
	]);
	assert.equal(out.row?.claimed, "success");
	assert.equal(out.row?.vdm_errors, 3);
	// With a model named for it, by the flag or by the deployment's own role, that model describes each result.
	const described = async (o: Parameters<typeof replaying>[1]) => {
		const named = await replaying([[move(0.5)], [move(0.6)], [finish]], {
			robot: (reached) => toy(reached, { views: 1, spec: { vdm: { views: 1 } } }),
			aux: ["A cube and a bowl.", "The arm moved right."],
			...o,
		});
		assert.deepEqual(named.reached, [0.5, 0.6]);
		assert.match(named.results[0], /\nA cube and a bowl\.$/);
		assert.match(named.results[1], /\nThe arm moved right\.$/);
		assert.equal(named.row?.claimed, "success");
		assert.equal(named.row?.vdm_errors, 0);
	};
	await described({ flags: { vdm: true, "aux-model": "faux/faux-1" } });
	await described({ flags: { vdm: true }, before: async () => void useDeployment({ aux: { vdm: "faux/faux-1" } }) });
});
