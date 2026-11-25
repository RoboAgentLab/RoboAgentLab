// MotorMind (`--mm`, `--model motormind/<base>`) as the planner of a real pi session (pi's agent loop, extension
// runner and tools) over the robot base with a toy robot whose MotorMind callbacks run in-process: what a request to
// motormind's model gets that is not a turn of the session's own loop (the dashboard's model check, /robot-check's
// planner row, a question to a VLM, visual differencing, a reviewed approval, pi's own summaries), and that the
// session's own turns play the mission whole around them.

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { fauxAssistantMessage } from "@earendil-works/pi-ai/providers/faux";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { llmCheck, plannerRow } from "../src/infra/check.ts";
import type { Deployment } from "../src/infra/config.ts";
import {
	type Cycle,
	type Executor,
	look,
	type MmCall,
	type MmReply,
	type ObserveWhy,
	type SegmentReport,
	type SubgoalInput,
	type SubgoalRun,
} from "../src/modes/motormind/contract.ts";
import { motormindProvider, STATIC_TEXT } from "../src/modes/motormind/provider.ts";
import type { Segment } from "../src/modes/motormind/types.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";
import { mmCaps } from "./helpers/motormind-robot.ts";
import {
	ask,
	at,
	llm,
	reply,
	type Session,
	type SessionOptions,
	vlm,
	within,
	withSession,
} from "./helpers/pi-session.ts";

type Json = Record<string, any>;

/** What a request to motormind's model gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"motormind plays its mission's own turns and answers no other request; --aux-model names a model for the checks that take one";

/** The base model the mission's roles are asked on (pi-ai's scripted faux model). */
const BASE = "faux/faux-1";
type RoleOf = "plan" | "replan" | "verify" | "other";
/** What the base model answers each role with, as the `n`-th request of that role. */
type Answers = (role: RoleOf, n: number) => string | Promise<string>;

const PLAN = {
	clauses: ["the black bowl is on the plate"],
	rationale: "one bowl, one plate",
	subgoals: [
		{
			name: "take",
			subgoal: "Grasp the black bowl",
			criterion: "the bowl is held",
			target: "the black bowl",
			target_boxes: { scene: [100, 100, 200, 200] },
			clause: 0,
			max_cycles: 8,
		},
		{
			name: "put",
			subgoal: "Put the bowl on the plate",
			criterion: "the bowl rests on the plate, jaws open",
			target: "the black bowl",
			target_boxes: { scene: [100, 100, 200, 200] },
			clause: 0,
			max_cycles: 6,
		},
	],
};
const DONE = {
	status: "done",
	because: "it is",
	evidence: "the bowl",
	identity: "",
	replan: false,
	clauses: [{ what: "the black bowl is on the plate", met: false, evidence: "" }],
};
const TASK_DONE = {
	status: "task_done",
	because: "the bowl sits on the plate",
	evidence: "plate",
	identity: "",
	replan: false,
	clauses: [{ what: "the black bowl is on the plate", met: true, evidence: "seen" }],
};
/** A mission's roles: the plan, two verdicts of a subgoal done, then the task done; `other` answers any other request. */
const mission =
	(other: (n: number) => string = () => "BASE MODEL ANSWER"): Answers =>
	(role, n) =>
		role === "plan" || role === "replan"
			? JSON.stringify(PLAN)
			: role === "verify"
				? JSON.stringify(n >= 2 ? TASK_DONE : DONE)
				: other(n);

/**
 * withSession (./helpers/pi-session.ts) with motormind/<BASE> as the session's model from its start, the base
 * model answering by `answers` (each request's role read from its system prompt), and `--mm=true` among the flags.
 */
async function withMotorMind<T>(
	extensions: ((pi: ExtensionAPI) => void)[],
	run: (o: { session: Session; ctx: () => ExtensionContext; stderr: string[]; roles: RoleOf[] }) => Promise<T>,
	o: {
		flags?: Record<string, string | boolean>;
		argv?: string[];
		deployment?: Deployment;
		settings?: SessionOptions["settings"];
		answers?: Answers;
	} = {},
): Promise<T> {
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "motormind-session-")));
	const roles: RoleOf[] = [];
	const counts: Record<RoleOf, number> = { plan: 0, replan: 0, verify: 0, other: 0 };
	const answers = o.answers ?? mission();
	// The provider side of pi-ai gets the normalized transcript: the system prompt is its leading system message.
	const route = async (context: { messages: { role: string; content: unknown }[] }) => {
		const first = context.messages[0];
		const system = first?.role === "system" && typeof first.content === "string" ? first.content : "";
		const role: RoleOf = /^You plan/.test(system)
			? "plan"
			: /^You replan/.test(system)
				? "replan"
				: /^You judge/.test(system)
					? "verify"
					: "other";
		roles.push(role);
		return fauxAssistantMessage(await answers(role, counts[role]++));
	};
	try {
		return await withSession(
			dir,
			extensions,
			async ({ session, faux, ctx, stderr }) => {
				faux.setResponses(Array.from({ length: 200 }, () => route as never));
				return run({ session, ctx, stderr, roles });
			},
			{
				flags: { mm: "true", ...o.flags },
				argv: o.argv,
				deployment: o.deployment,
				settings: o.settings,
				model: `motormind/${BASE}`,
			},
		);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

const FIX = JSON.parse(
	readFileSync(new URL("./fixtures/motormind/m1-report-extract.json", import.meta.url), "utf8"),
) as { scenes: Record<string, Json> };
const GOAL = FIX.scenes.libero_goal;
const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 0, 0xff, 0xd9]).toString("base64");

/** An env observation as `env.mm_observe` returns one, from the M1 fixture (as ./motormind-logic.test.ts builds it). */
function rawObservation(s: { obsId: number; clock: number; holding: boolean; images: boolean }): Json {
	const seg = s.holding ? GOAL.C6.blocked : GOAL.C6.contact;
	const cam = (name: string) => ({
		mujoco_camera: name,
		width: 768,
		height: 768,
		intrinsic: [
			[927.06, 0, 384],
			[0, 927.06, 384],
			[0, 0, 1],
		],
		cam2base: [
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 1],
			[0, 0, 0, 1],
		],
		marked768: JPEG,
		marked512: JPEG,
		raw512: JPEG,
		aim_px: [400, 500],
	});
	return {
		obs_id: s.obsId,
		frame_clock_s: s.clock,
		pose: { position_m: seg.pose_after.position_mm.map((v: number) => v / 1000), rotation: seg.pose_after.rotation },
		base_pose: { position_m: GOAL.C1.p_b, rotation: GOAL.C1.R_b },
		caps: mmCaps(),
		table_z_m: 0,
		gripper: { opening_mm: seg.grasp?.opening_mm ?? 79, commanded: s.holding ? "close" : "open", moving: false },
		grasp: { holding: s.holding, reason: seg.grasp?.reason ?? "", closed_fraction: seg.grasp?.closed_fraction },
		clearance: {
			seen: true,
			mm: 120,
			median_mm: 125,
			surface_z_m: 0,
			at_range_floor: false,
			near_min_range: false,
			jaws_down: true,
		},
		camera_names: ["scene", "side", "wrist"],
		terminated: false,
		truncated: false,
		episode_over: false,
		degraded: { table_z: false, side_mount: false },
		config: {},
		...(s.images
			? {
					cameras: { scene: cam("agentview"), side: cam("sideview"), wrist: cam("robot0_eye_in_hand") },
					mosaic_jpeg: JPEG,
				}
			: {}),
	};
}

const act = (label: string): MmCall => ({
	tool: "mm_act",
	args: { type: "move", direction: "down", distance_mm: 50, label },
	label,
});
/**
 * ./motormind-logic.test.ts's scripted executor: each subgoal's attempt looks (the cached frame when fresh), moves
 * down once, and ends: the first holding the bowl, the next with the model's done.
 */
function scriptedExecutor() {
	let attempts = 0;
	return (input: SubgoalInput): Executor => {
		const n = attempts++;
		return (async function* (): Executor {
			const observes: { why: ObserveWhy; yielded: boolean }[] = [];
			const first = yield* look(input.cache, "cycle_start", observes);
			const steps: SegmentReport[] = [];
			let episodeOver = first?.episode_over === true;
			const label = "move down 50.0 mm (base)";
			const call = act(label);
			const reply: MmReply = yield call;
			if (reply.segments) {
				steps.push(...reply.segments);
				for (const st of reply.segments) input.onEvent({ type: "step_end", cycle: 0, step: st });
			}
			episodeOver ||= reply.episode_over;
			const ending: SubgoalRun["ending"] = n === 0 ? "HELD" : "MODEL_DONE";
			const cycle: Cycle = {
				index: 0,
				steps,
				proposal: {
					assessment: "a",
					done: ending === "MODEL_DONE",
					expect: "e",
					confidence: 1,
					command: {},
					deltas: [{ kind: "translate" as const, label, push: false }],
				},
				timing: { wall_s: 1, motion_s: 0.5, calls: [] },
			};
			input.onEvent({ type: "proposed", cycle: 0, proposal: cycle.proposal! });
			return {
				cycles: [cycle],
				ending_text: ending.toLowerCase(),
				episode_over: episodeOver,
				carry: { ...(input.carry ?? {}), n },
				observes,
				timing: cycle.timing,
				ending,
				...(n === 0 ? { held: { gap_mm: 63 } } : {}),
			};
		})();
	};
}

const FINISH = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p: { status: string }) => ({ content: [{ type: "text" as const, text: p.status }], details: p }),
};

/** The motions that reached the toy robot, by label. */
type Log = { steps: string[] };

/**
 * The toy robot: the real robot base with MotorMind mode mounted on in-process callbacks; `spec` is added to the
 * robot's. motormind's provider is mounted first, as the robot base mounts its own (../src/robot.ts), with the
 * scripted executor; with `base` the toy mounts none, and the base's own (with the real executor) plays when the
 * command line names the model. `starting` runs while the robot starts: after the session started, before the robot
 * published its handle.
 */
const toy =
	(
		log: Log,
		o: { spec?: Partial<RobotSpec>; starting?: (ctx: ExtensionContext) => Promise<void>; base?: true } = {},
	) =>
	(pi: ExtensionAPI) => {
		if (!o.base)
			motormindProvider(pi, ["node", "pi", "--model", `motormind/${BASE}`], {
				runSubgoal: scriptedExecutor(),
				describers: { legend: () => "L", capabilities: () => "C" },
			});
		const s = { obsId: 0, holding: true, acts: 0 };
		const observation = (images = true) => {
			s.obsId++;
			return rawObservation({ obsId: s.obsId, clock: s.obsId * 0.05, holding: s.holding, images });
		};
		defineRobot(pi, {
			name: "toy",
			task: [],
			keepImages: 1,
			start: async (ctx) => {
				await o.starting?.(ctx);
				return ["finish"];
			},
			result: () => ({}),
			status: () => ({ step: 0 }),
			finish: FINISH,
			motormind: {
				observe: async () => observation(),
				runStep: async (seg: Segment, opt) => {
					log.steps.push(seg.label);
					// the second motion lets go: the "put" subgoal's frame
					if (++s.acts >= 2) s.holding = false;
					const obs = observation(opt.observe === "full");
					return {
						outcome: "done",
						abort_reason: "",
						moved_m: seg.distance_m ?? 0,
						along_axis_m: seg.distance_m ?? 0,
						lateral_m: 0,
						turned_rad: 0,
						kind: seg.kind,
						ticks: 1,
						elapsed_s: 0.01,
						pose_after: obs.pose,
						episode_over: false,
						terminated: false,
						observation: obs,
					};
				},
				task: () => "put the black bowl on the plate",
			},
			...o.spec,
		});
	};

const texts = (m: { content: string | { type: string; text?: string }[] }) =>
	typeof m.content === "string" ? m.content : m.content.map((c) => (c.type === "text" ? c.text : "")).join("");
/** The session's replies: stop reason, error, text and calls. */
const repliesOf = (session: Session) =>
	session.messages.flatMap((m) => {
		if (m.role !== "assistant") return [];
		const a = m as AssistantMessage;
		const calls = a.content.flatMap((c) => (c.type === "toolCall" ? [` [call ${c.name} ${c.id}]`] : []));
		return [`${a.stopReason}${a.errorMessage ? ` (${a.errorMessage})` : ""}: ${texts(a)}${calls.join("")}`];
	});
/** Each call's result: tool, whether it failed, text. */
const resultsOf = (session: Session) =>
	session.messages.flatMap((m) =>
		m.role === "toolResult" ? [`${m.toolName}${m.isError ? " ERROR" : ""}: ${texts(m)}`] : [],
	);
/** The session's entries of a type. */
const entriesOf = (session: Session, type: string) =>
	(session.sessionManager.getEntries() as { type: string; customType?: string; data?: any }[]).flatMap((e) =>
		e.type === "custom" && e.customType === type ? [e.data] : [],
	);
/** The robot's result row, written at pi's quit. */
const rowOf = async (session: Session) => {
	await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
	return entriesOf(session, RESULT_ENTRY).at(-1) as Json | undefined;
};
const solve = (session: Session) => within(30_000, session.prompt("Solve the task."), "the prompt");

/** The session's replies of the mission the toy's scripted executor plays. */
const MISSION = [
	"toolUse: [motormind] no role call → mm_observe [call mm_observe motormind-1]",
	"toolUse: [motormind] planner → mm_act move down 50.0 mm (base) [call mm_act motormind-2]",
	"toolUse: [motormind] verify → mm_act move down 50.0 mm (base) [call mm_act motormind-3]",
	"toolUse: [motormind] verify verify → finish → task_done: the bowl sits on the plate [call finish motormind-4]",
];
/** Both motions of that mission, as they reached the robot. */
const MOTIONS = ["move down 50.0 mm (base) [1/1]", "move down 50.0 mm (base) [1/1]"];

/** /robot-check's planner row: a one-token completion of the session's model through pi's registry. */
const robotCheck = async (ctx: ExtensionContext) =>
	plannerRow(undefined, undefined, undefined, () => llmCheck(ctx.modelRegistry, ctx.model as never));

test("a real session: the dashboard's model check, /robot-check's planner row, a question to a VLM and pi's bug-report summary before the session's first turn get motormind's error and start no mission; the session's first turn starts it, and it is played whole", async () => {
	const log: Log = { steps: [] };
	await withMotorMind([toy(log)], async ({ session, ctx, stderr }) => {
		assert.equal(await llm(ctx()), `failed: ${NOT_A_TURN}`);
		assert.deepEqual(await robotCheck(ctx()), {
			status: "FAIL",
			check: "planner",
			detail: `motormind/${BASE}: ${NOT_A_TURN}`,
		});
		assert.equal(await vlm(ctx()), `error: ${NOT_A_TURN}`);
		await assert.rejects(session.summarizeForBugReport({ signal: new AbortController().signal }), {
			message: `Bug report summary failed: ${NOT_A_TURN}`,
		});
		assert.deepEqual(entriesOf(session, "motormind_turn"), []);
		await solve(session);
		assert.deepEqual(repliesOf(session), MISSION);
		assert.deepEqual(log.steps, MOTIONS);
		assert.equal(entriesOf(session, "motormind_turn").length, 4);
		assert.equal((await rowOf(session))?.claimed, "success");
		// Nothing was printed but the robot's row.
		assert.deepEqual(
			stderr.filter((l) => !l.startsWith("[toy] {")),
			[],
		);
	});
});

test("a real session: the dashboard's model check and a question to a VLM while the robot is starting get motormind's error and start nothing: once the robot is up, the session's first turn starts the mission", async () => {
	// The toy's own provider with the scripted executor, and the base's own as the command line mounts it.
	for (const base of [false, true]) {
		const log: Log = { steps: [] };
		const checked: string[] = [];
		const starting = async (ctx: ExtensionContext) => void checked.push(await llm(ctx), await vlm(ctx));
		await withMotorMind(
			[toy(log, { starting, ...(base ? { base } : {}) })],
			async ({ session }) => {
				assert.deepEqual(checked, [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`]);
				assert.deepEqual(entriesOf(session, "motormind_turn"), []);
				await solve(session);
				assert.deepEqual(
					repliesOf(session),
					base
						? [
								MISSION[0],
								// the scripted base model gives the real planner no plan
								"toolUse: [motormind] planner✗(no_json) planner✗(no_json) planner✗(no_json) → finish → error: the planner gave no usable plan in 3 attempts [call finish motormind-2]",
							]
						: MISSION,
				);
				assert.deepEqual(log.steps, base ? [] : MOTIONS);
			},
			base
				? {
						argv: ["--model", `motormind/${BASE}`],
						answers: (role) => (role === "other" ? "OK" : "I cannot plan this."),
					}
				: {},
		);
	}
});

test("a real session: requests during the mission get motormind's error (the model check as a call starts, a question to a VLM after it, a request under another id, one with two user messages, one with a system prompt and tools of its own, a question right before the session's own request, pi's bug-report summary); the mission is played whole", async () => {
	const log: Log = { steps: [] };
	const answers: string[] = [];
	let session: Session | undefined;
	const note = (fn: (ctx: ExtensionContext) => Promise<string>) => async (ctx: ExtensionContext) =>
		void answers.push(await fn(ctx));
	await withMotorMind(
		[
			toy(log),
			at("tool_execution_start", 1, note(llm)),
			at("tool_execution_end", 1, note(vlm)),
			at(
				"turn_end",
				2,
				note((ctx) => reply(ctx, { messages: [ask] }, { sessionId: "0193e1c0-0000-7000-8000-000000000000" })),
			),
			at(
				"turn_end",
				2,
				note((ctx) => reply(ctx, { messages: [ask, ask] })),
			),
			at(
				"turn_end",
				2,
				note((ctx) =>
					reply(ctx, {
						systemPrompt: "You are a policy.",
						tools: [{ name: "policy_act", description: "act", parameters: Type.Object({}) }],
						messages: [ask],
					}),
				),
			),
			// pi hands the transcript to the extensions' context handlers right before each request of its loop.
			at("context", 3, note(vlm)),
			// pi's summary for a bug report is asked under the session's own id, with pi's mark.
			at(
				"turn_end",
				3,
				note(() =>
					(session as Session)
						.summarizeForBugReport({ signal: new AbortController().signal })
						.catch((err: Error) => `threw: ${err.message}`),
				),
			),
		],
		async (o) => {
			session = o.session;
			await solve(o.session);
			assert.deepEqual(answers, [
				`failed: ${NOT_A_TURN}`,
				`error: ${NOT_A_TURN}`,
				`error: ${NOT_A_TURN}`,
				`error: ${NOT_A_TURN}`,
				`error: ${NOT_A_TURN}`,
				`error: ${NOT_A_TURN}`,
				`threw: Bug report summary failed: ${NOT_A_TURN}`,
			]);
			assert.deepEqual(repliesOf(o.session), MISSION);
			assert.deepEqual(log.steps, MOTIONS);
			assert.equal(entriesOf(o.session, "motormind_turn").length, 4);
			assert.equal((await rowOf(o.session))?.claimed, "success");
		},
	);
});

test("a real session: a request while the session's own turn waits for the planner gets motormind's error at once, and the turn then goes on with the planner's reply", async () => {
	const log: Log = { steps: [] };
	let asked = () => {};
	const planning = new Promise<void>((resolve) => {
		asked = resolve;
	});
	let release = () => {};
	const released = new Promise<void>((resolve) => {
		release = resolve;
	});
	const answers = mission();
	await withMotorMind(
		[toy(log)],
		async ({ session, ctx }) => {
			const played = solve(session);
			try {
				// A prompt that ends without asking the planner is not this mission's course: its replies say so.
				await within(30_000, Promise.race([planning, played]), "the planner's request");
				assert.deepEqual(
					[
						await llm(ctx()),
						await vlm(ctx()),
						await reply(ctx(), { messages: [ask] }, { sessionId: "0193e1c0-0000-7000-8000-000000000000" }),
					],
					[`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`],
				);
			} finally {
				release();
			}
			await played;
			assert.deepEqual(repliesOf(session), MISSION);
			assert.deepEqual(log.steps, MOTIONS);
		},
		{
			answers: async (role, n) => {
				if (role === "plan" && n === 0) {
					asked();
					await released;
				}
				return answers(role, n);
			},
		},
	);
});

test("a real session: once the mission is over, the model check, a question to a VLM, pi's bug-report summary and its branch summary get motormind's error; a second prompt still gets the mission's static text, and compaction is still cancelled without asking a model", async () => {
	const log: Log = { steps: [] };
	await withMotorMind(
		[toy(log)],
		async ({ session, ctx, roles }) => {
			await solve(session);
			assert.deepEqual(repliesOf(session), MISSION);
			assert.equal(await llm(ctx()), `failed: ${NOT_A_TURN}`);
			assert.equal(await vlm(ctx()), `error: ${NOT_A_TURN}`);
			await assert.rejects(session.summarizeForBugReport({ signal: new AbortController().signal }), {
				message: `Bug report summary failed: ${NOT_A_TURN}`,
			});
			const first = (
				session.sessionManager.getEntries() as { id: string; type: string; message?: { role: string } }[]
			).find((e) => e.type === "message" && e.message?.role === "assistant");
			await assert.rejects(
				within(20_000, session.navigateTree(first!.id, { summarize: true }), "the branch summary"),
				{
					message: `Branch summarization failed: ${NOT_A_TURN}`,
				},
			);
			const asked = roles.length;
			await assert.rejects(within(20_000, session.compact(), "compaction"), { message: "Compaction cancelled" });
			assert.equal(roles.length, asked);
			const before = session.messages.length;
			await within(30_000, session.prompt("And now put it back."), "the second prompt");
			assert.deepEqual(
				session.messages.slice(before).map((m) => `${m.role}: ${texts(m as never)}`),
				["user: And now put it back.", `assistant: ${STATIC_TEXT}`],
			);
			assert.equal(entriesOf(session, "motormind_turn").length, 4);
			assert.deepEqual(log.steps, MOTIONS);
		},
		{ settings: { compaction: { enabled: true, reserveTokens: 16_384, keepRecentTokens: 1 } } },
	);
});

test("a real session: visual differencing's question (--vdm, no model named for it) is not answered by motormind: each result it would describe says so, and the mission is played whole; with a model named for it (--aux-model, the deployment's aux.vdm), that model describes them", async () => {
	const vdm = { spec: { vdm: { views: 1 } } };
	const log: Log = { steps: [] };
	await withMotorMind(
		[toy(log, vdm)],
		async ({ session, roles }) => {
			await solve(session);
			assert.deepEqual(repliesOf(session), MISSION);
			assert.deepEqual(log.steps, MOTIONS);
			const results = resultsOf(session);
			assert.deepEqual(
				results.map((r) => r.split(":")[0]),
				["mm_observe", "mm_act", "mm_act", "finish"],
			);
			for (const r of results.slice(0, 3))
				assert.ok(r.endsWith(`[visual differencing unavailable: ${NOT_A_TURN}]`), r);
			assert.deepEqual(
				entriesOf(session, "vdm").map((e) => [e.tool, e.error]),
				[
					["mm_observe", NOT_A_TURN],
					["mm_act", NOT_A_TURN],
					["mm_act", NOT_A_TURN],
				],
			);
			// The base model was asked the mission's roles only.
			assert.ok(!roles.includes("other"));
			const row = await rowOf(session);
			assert.deepEqual([row?.claimed, row?.vdm_calls, row?.vdm_errors], ["success", 3, 3]);
		},
		{ flags: { vdm: true } },
	);
	const scene = "The gripper holds the bowl above the plate.";
	const ways: { flags: Record<string, string | boolean>; deployment?: Deployment }[] = [
		{ flags: { vdm: true, "aux-model": BASE } },
		{ flags: { vdm: true }, deployment: { aux: { vdm: BASE } } },
	];
	for (const named of ways) {
		const steps: Log = { steps: [] };
		await withMotorMind(
			[toy(steps, vdm)],
			async ({ session }) => {
				await solve(session);
				assert.deepEqual(repliesOf(session), MISSION);
				assert.deepEqual(steps.steps, MOTIONS);
				for (const r of resultsOf(session).slice(0, 3)) assert.ok(r.endsWith(`\n${scene}`), r);
				assert.deepEqual(
					entriesOf(session, "vdm").map((e) => [e.model, e.text]),
					Array.from({ length: 3 }, () => [BASE, scene]),
				);
				const row = await rowOf(session);
				assert.deepEqual([row?.claimed, row?.vdm_errors], ["success", 0]);
			},
			{ ...named, answers: mission(() => scene) },
		);
	}
});

test("a real session: a reviewed approval's question (--approval reviewed, no model named for the reviewer) is not answered by motormind: no motion is approved and none reaches the robot; with a model named for the reviewer (--aux-model, --approval-model) that approves, both motions run and the mission is played whole", async () => {
	const refusal = `mm_act ERROR: mm_act was not approved by the reviewer (error): the reviewer failed: ${NOT_A_TURN}. It did not run.`;
	const log: Log = { steps: [] };
	await withMotorMind(
		[toy(log)],
		async ({ session }) => {
			await solve(session);
			assert.deepEqual(log.steps, []);
			const acts = resultsOf(session).filter((r) => r.startsWith("mm_act"));
			assert.ok(acts.length > 0);
			for (const r of acts) assert.equal(r, refusal);
			const approvals = entriesOf(session, "approval");
			assert.deepEqual(
				approvals.map((e) => [e.tool, e.decision, e.reason]),
				acts.map(() => ["mm_act", "error", `the reviewer failed: ${NOT_A_TURN}`]),
			);
		},
		{ flags: { approval: "reviewed" } },
	);
	const approve = '{"decision": "approve", "reason": "the move is safe"}';
	for (const flag of ["aux-model", "approval-model"]) {
		const steps: Log = { steps: [] };
		await withMotorMind(
			[toy(steps)],
			async ({ session }) => {
				await solve(session);
				assert.deepEqual(repliesOf(session), MISSION);
				assert.deepEqual(steps.steps, MOTIONS);
				assert.deepEqual(
					entriesOf(session, "approval").map((e) => [e.decision, e.model]),
					[
						["approve", BASE],
						["approve", BASE],
					],
				);
				assert.equal((await rowOf(session))?.claimed, "success");
			},
			{ flags: { approval: "reviewed", [flag]: BASE }, answers: mission(() => approve) },
		);
	}
});

test("a real session: a switch of model away from motormind and back keeps the session's own turns its own: the next prompt gets the mission's static text, and the model check gets motormind's error", async () => {
	const log: Log = { steps: [] };
	await withMotorMind([toy(log)], async ({ session, ctx }) => {
		await solve(session);
		assert.deepEqual(repliesOf(session), MISSION);
		const mm = session.model;
		const other = ctx().modelRegistry.find("faux", "faux-1");
		assert.ok(mm && other);
		// pi announces each switch to the extensions (model_select); the session stays the same.
		await session.setModel(other);
		await session.setModel(mm);
		assert.equal(await llm(ctx()), `failed: ${NOT_A_TURN}`);
		assert.equal(
			await reply(ctx(), { messages: [ask] }, { sessionId: "0193e1c0-0000-7000-8000-000000000000" }),
			`error: ${NOT_A_TURN}`,
		);
		const before = session.messages.length;
		await within(30_000, session.prompt("And now put it back."), "the prompt after the switch");
		assert.deepEqual(
			session.messages.slice(before).map((m) => `${m.role}: ${texts(m as never)}`),
			["user: And now put it back.", `assistant: ${STATIC_TEXT}`],
		);
		assert.deepEqual(log.steps, MOTIONS);
	});
});
