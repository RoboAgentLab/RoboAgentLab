// humanclaw-psv/<base> as the model of a real pi session (pi's agent loop, extension runner and tools) over the real
// HumanCLAW robot in paper mode, its env server ./sim-stub.ts's fake one and its base model the scripted one, asked
// through pi's model registry: what a request to humanclaw-psv gets that is not a turn of the session's own loop, what
// the reviewed approval then does, and that the planner's own decisions stay whole.

import assert from "node:assert/strict";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fauxAssistantMessage } from "@earendil-works/pi-ai/providers/faux";
import type { ExtensionAPI, ExtensionContext, FileEntry } from "@earendil-works/pi-coding-agent";
import { RESULT_ENTRY } from "../src/robot.ts";
import humanclaw from "../src/robots/humanclaw/index.ts";
import { PSV_ENTRY } from "../src/robots/humanclaw/provider.ts";
import { at, llm, type Session, vlm, within, withSession } from "./helpers/pi-session.ts";
import { fakeEnv, nd } from "./sim-stub.ts";

/** What a request to humanclaw-psv gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"humanclaw-psv plans the episode's steps and answers no other request; --aux-model names a model for the checks that take one";
const MODEL = "humanclaw-psv/faux/faux-1";

const plan = (action_id: number, action_name: string, goal = "Find the bed") =>
	JSON.stringify({ visible_state: "A hallway.", mid_level_goal: goal, action_id, action_name });
/** The base model's decisions in turn: a turn, a walk (which the verifier is asked about), Stop/Stand. */
const DECISIONS = [plan(2, "Turn<left><45>"), plan(0, "Walk<forward><slow>"), plan(1, "Stop/Stand", "Stop/Stand")];
const ACCEPT = JSON.stringify({ verdict: "accept", reason: "the lane is clear" });

/**
 * The fake env server of one episode, its ego images all of the grey `shade`: Stop/Stand ends it; `steps` are the
 * actions env.step got.
 */
async function episode(shade = 0) {
	let step = 0;
	const steps: string[] = [];
	const obsOf = (done = false) => ({
		ego: nd("uint8", [4, 4, 3], Buffer.alloc(4 * 4 * 3, shade)),
		instruction: "Find the bed.",
		step,
		max_steps: 100,
		done,
		stopped: done,
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
			steps.push(String(c.kwargs.action_name));
			return { ...obsOf(c.kwargs.skill === "stand"), collision: {} };
		}
		if (c.method === "env.finish")
			return { metrics: null, steps: step, active_stop: true, videos: [], metrics_path: null };
		return undefined;
	});
	return { env, steps };
}

const texts = (m: { content: { type: string; text?: string }[] }) =>
	m.content.map((c) => (c.type === "text" ? c.text : "")).join("");

/**
 * One episode with `-e robots/humanclaw --units=both --humanclaw-mode paper --model humanclaw-psv/faux/faux-1` and
 * `flags`, its ego images of the grey `shade`, in a new session or one whose file is `entries`: the base model answers
 * the planner with DECISIONS and the verifier with ACCEPT; any other question to it (a check that names it) gets
 * `aux`. `before` runs before the prompt and `after` after it, with the context pi gives an extension. Returns the
 * env's steps and calls, the loop's replies, each call's result, the planner's entries, how often the base model was
 * asked as the planner, the ego image each planner and verifier request carried, the other questions it got, the
 * robot's result row (written at pi's quit when the episode did not finish) and the session's file.
 */
async function paper(
	o: {
		flags?: Record<string, string | boolean>;
		aux?: string;
		before?: (ctx: ExtensionContext, session: Session) => Promise<void>;
		after?: (ctx: ExtensionContext, session: Session) => Promise<void>;
		more?: ((pi: ExtensionAPI) => void)[];
		shade?: number;
		entries?: FileEntry[];
	} = {},
) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "humanclaw-session-")));
	const { env, steps } = await episode(o.shade);
	const decisions = [...DECISIONS];
	let planned = 0;
	const images: unknown[] = [];
	const prompts: string[] = [];
	const questions: string[] = [];
	try {
		return await withSession(
			base,
			[humanclaw, ...(o.more ?? [])],
			async ({ session, faux, runtime, ctx }) => {
				const answer = (context: { messages: { content: unknown }[] }) => {
					const last = context.messages.at(-1)?.content;
					const prompt = Array.isArray(last)
						? last.map((c: { text?: string }) => c.text ?? "").join(" ")
						: String(last);
					const image = Array.isArray(last) ? last.find((c: { type?: string }) => c.type === "image") : undefined;
					if (/You are the plan-and-skill chooser/.test(prompt)) {
						images.push(image?.data);
						prompts.push(prompt);
						planned++;
						return fauxAssistantMessage(decisions.shift() ?? plan(1, "Stop/Stand", "Stop/Stand"));
					}
					if (/You are the verifier for the proposed action/.test(prompt)) {
						images.push(image?.data);
						prompts.push(prompt);
						return fauxAssistantMessage(ACCEPT);
					}
					questions.push(prompt);
					return fauxAssistantMessage(o.aux ?? "no answer");
				};
				faux.setResponses(Array.from({ length: 40 }, () => answer as never));
				const planner = runtime.getModel("humanclaw-psv", "faux/faux-1");
				assert.ok(planner, `${MODEL} is registered`);
				await session.setModel(planner);
				await o.before?.(ctx(), session);
				// A session that hangs fails the test.
				await within(20_000, session.prompt("Solve the task."), "the session's prompt");
				await o.after?.(ctx(), session);
				await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
				const entries = (type: string) =>
					(session.sessionManager.getEntries() as { type: string; customType?: string; data?: any }[]).flatMap(
						(e) => (e.type === "custom" && e.customType === type ? [e.data] : []),
					);
				return {
					robot: steps,
					env: env.calls.map((c) => c.method).filter((m) => m !== "code.api"),
					replies: session.messages.flatMap((m) =>
						m.role === "assistant"
							? [
									[
										m.stopReason,
										m.errorMessage ?? texts(m),
										m.content.flatMap((c) => (c.type === "toolCall" ? [[c.id, c.name, c.arguments]] : [])),
									],
								]
							: [],
					),
					results: session.messages.flatMap((m) =>
						m.role === "toolResult"
							? [`${m.toolCallId} ${m.toolName}${m.isError ? " ERROR" : ""}: ${texts(m)}`]
							: [],
					),
					decided: entries(PSV_ENTRY).map((e: Record<string, unknown>) => [e.step, e.history_item, e.state]),
					planned,
					images,
					prompts,
					questions,
					row: entries(RESULT_ENTRY).at(-1) as Record<string, unknown> | undefined,
					file: [session.sessionManager.getHeader(), ...session.sessionManager.getEntries()] as FileEntry[],
				};
			},
			{
				argv: ["--model", MODEL],
				flags: { "env-url": env.url, units: "both", "humanclaw-mode": "paper", ...o.flags },
				...(o.entries ? { entries: o.entries } : {}),
			},
		);
	} finally {
		env.close();
		rmSync(base, { recursive: true, force: true });
	}
}
/** What an episode did that a request from elsewhere must leave alone. */
const whole = (out: Awaited<ReturnType<typeof paper>>) => ({
	robot: out.robot,
	env: out.env,
	replies: out.replies,
	results: out.results,
	decided: out.decided,
	planned: out.planned,
	claimed: out.row?.claimed,
});
/** The answer to a request from elsewhere, awaited five seconds at most. */
const answer = (asked: Promise<string>) => within(5000, asked, "a request that is not the session's turn");
/** pi's summary of the session for a bug report (the session's id, and pi's mark on a one-off summary). */
const bugReport = (session: Session) =>
	session
		.summarizeForBugReport({ signal: new AbortController().signal })
		.then((s) => `summary: ${JSON.stringify(s)}`)
		.catch((err: Error) => `threw: ${err.message}`);

test("a real session in paper mode: the dashboard's model check, a question to a VLM and pi's bug-report summary, before the first decision, while an act is about to run, between two decisions and after the end, get the error; the planner's decisions, the env's steps and the entries are those of a session without them", async () => {
	const alone = await paper();
	assert.deepEqual(alone.robot, ["Turn<left><45>", "Walk<forward><slow>", "Stop/Stand"]);
	assert.equal(alone.planned, 3);
	assert.equal(alone.row?.claimed, "success");
	const answers: string[] = [];
	let session: Session | undefined;
	/** The dashboard's model check, a question to a VLM and pi's summary for a bug report, in turn. */
	const ask = async (ctx: ExtensionContext) =>
		void answers.push(await answer(llm(ctx)), await answer(vlm(ctx)), await answer(bugReport(session as Session)));
	const out = await paper({
		before: async (ctx, s) => {
			session = s;
			await ask(ctx);
		},
		// The robot's first act, as the approval gate sees it before it runs; the end of the planner's second turn.
		more: [at("tool_call", 2, ask), at("turn_end", 2, ask)],
		after: ask,
	});
	// Before the first decision, while an act is about to run, between two decisions and after the end.
	const each = [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`, `threw: Bug report summary failed: ${NOT_A_TURN}`];
	assert.deepEqual(answers, Array(4).fill(each).flat());
	assert.deepEqual(whole(out), whole(alone));
	// None of them reached the base model.
	assert.deepEqual(out.questions, []);
});

test("a real session in paper mode: once its run has ended, the session's own next prompt is still its turn: it is told that the planner ended the episode, and the episode is as it was", async () => {
	const alone = await paper();
	const out = await paper({
		after: async (_ctx, session) => {
			await within(20_000, session.prompt("Go on."), "the second prompt");
		},
	});
	assert.deepEqual(out.replies, [...alone.replies, ["stop", "The HumanCLAW planner already ended this episode.", []]]);
	assert.deepEqual(whole({ ...out, replies: alone.replies }), whole(alone));
});

test("a real session in paper mode: a reviewed approval's question, with no model named for the reviewer, gets the error: the first act does not run, and the planner quotes the refusal whole and stops; with --aux-model or --approval-model approving, the episode is whole", async () => {
	const out = await paper({ flags: { approval: "reviewed" } });
	const refusal = `act was not approved by the reviewer (error): the reviewer failed: ${NOT_A_TURN}. It did not run.`;
	assert.deepEqual(out.robot, []);
	assert.equal(out.results[1], `psv_2 act ERROR: ${refusal}`);
	// The planner quotes the refusal whole, and stops.
	assert.deepEqual(out.replies.at(-1), ["stop", `the robot refused: ${refusal}; the planner stops.`, []]);
	assert.equal(out.row?.claimed, null);
	assert.equal(out.row?.approval_errors, 1);
	assert.equal(out.planned, 1);
	const approve = '{"decision": "approve", "reason": "the move is safe"}';
	for (const flag of ["aux-model", "approval-model"]) {
		const named = await paper({ flags: { approval: "reviewed", [flag]: "faux/faux-1" }, aux: approve });
		assert.deepEqual(named.robot, ["Turn<left><45>", "Walk<forward><slow>", "Stop/Stand"]);
		assert.equal(named.row?.claimed, "success");
		assert.equal(named.row?.approval_errors, 0);
	}
});

test("a real session in paper mode over the file of an earlier one, under its id (--continue, --session, /resume) or a new one (a fork or clone): the planner's next episode goes on from the earlier one's last call id, starts afresh and plans from its own ego images: its planner and verifier are sent what a new session's are, it records the decisions a new session records, and it is whole", async () => {
	const first = await paper();
	assert.deepEqual(
		first.results.map((r) => r.slice(0, r.indexOf(":"))),
		["psv_1 look", "psv_2 act", "psv_3 act", "psv_4 act", "psv_5 finish"],
	);
	// A new session over an env whose ego images are another grey: what its planner and verifier are sent.
	const alone = await paper({ shade: 200 });
	assert.notDeepEqual(alone.images, first.images);
	const [header, ...entries] = first.file as [Record<string, unknown>, ...FileEntry[]];
	for (const id of [header.id, "0193e1c0-0000-7000-8000-00000000f02c"]) {
		const what = id === header.id ? "the same id" : "a new id";
		const out = await paper({ shade: 200, entries: [{ ...header, id } as FileEntry, ...entries] });
		assert.deepEqual(out.robot, ["Turn<left><45>", "Walk<forward><slow>", "Stop/Stand"], what);
		assert.deepEqual(
			out.replies.slice(first.replies.length).map((r) => (r[2] as unknown[][]).map((c) => c.slice(0, 2))),
			[[["psv_6", "look"]], [["psv_7", "act"]], [["psv_8", "act"]], [["psv_9", "act"]], [["psv_10", "finish"]]],
			what,
		);
		assert.deepEqual(out.images, alone.images, what);
		assert.deepEqual(out.prompts, alone.prompts, what);
		assert.deepEqual(out.decided.slice(first.decided.length), alone.decided, what);
		assert.equal(out.row?.claimed, "success", what);
	}
});
