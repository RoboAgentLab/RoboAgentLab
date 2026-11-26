// The fine-tuned policy as the model of a real pi session (pi's agent loop, extension runner and tools), over the robot
// base with a toy robot (its own tools, or one arm or two in the real units mode, two with its finish verifier) and the
// policy's service a local HTTP stub: what a request to the policy's model gets that is not a turn of the session's own
// loop, what the checks that ask the session's model then do, and that the session's own steps stay whole.

import assert from "node:assert/strict";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fauxAssistantMessage } from "@earendil-works/pi-ai/providers/faux";
import type { ExtensionAPI, ExtensionContext, FileEntry } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { Deployment } from "../src/infra/config.ts";
import finetuned, { STEP_ENTRY } from "../src/modes/finetuned/index.ts";
import type { UnitsSpec } from "../src/modes/units/index.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";
import { at, llm, PNG, type Session, vlm, within, withSession } from "./helpers/pi-session.ts";

/** What a request to a fine-tuned model gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"finetuned runs a fine-tuned policy and answers no other request; --aux-model names a model for the checks that take one";

const FINISH = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p: { status: string }) => ({ content: [{ type: "text" as const, text: p.status }], details: p }),
};
const view = { type: "image" as const, data: PNG, mimeType: "image/png" };

/**
 * A toy robot with its own tools look and go, each returning a camera view and a text that ends with what `tag` says
 * then; `inside` runs while go runs.
 */
const toolsToy =
	(
		log: string[],
		o: {
			inside?: (ctx: ExtensionContext, x: number) => Promise<unknown>;
			spec?: Partial<RobotSpec>;
			tag?: () => string;
		} = {},
	) =>
	(pi: ExtensionAPI) => {
		const r = defineRobot(pi, {
			name: "toy",
			task: [],
			keepImages: 2,
			start: async () => ["look", "go", "finish"],
			result: () => ({}),
			status: () => ({ step: 0 }),
			finish: FINISH,
			...o.spec,
		});
		r.tool("look", "Look at the scene.", Type.Object({}), async () => {
			log.push("look");
			return { content: [{ type: "text", text: `scene${o.tag?.() ?? ""}` }, view], details: {} };
		});
		r.tool("go", "Go to x.", Type.Object({ x: Type.Number() }), async (p, _signal, ctx) => {
			log.push(`go ${p.x}`);
			await o.inside?.(ctx, p.x);
			return { content: [{ type: "text", text: `at ${p.x}${o.tag?.() ?? ""}` }, view], details: {} };
		});
	};

const VECTORS: UnitsSpec["vectors"] = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/**
 * A toy arm, or two (`arms`), in the real units mode; each observation carries a front view and a wrist view of each
 * arm.
 */
const unitsToy = (arms?: string[]) => (log: string[]) => (pi: ExtensionAPI) => {
	const pos: Record<string, number[]> = {};
	const tcp = (arm = "one") => {
		pos[arm] ??= [0.5, 0, 0.2];
		return pos[arm];
	};
	defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 4,
		start: async () => ["finish"],
		result: () => ({}),
		finish: FINISH,
		units: {
			vectors: VECTORS,
			stepM: 0.02,
			...(arms ? { arms } : {}),
			apply: async (m) => {
				log.push(`${m.arm ?? "arm"} ${m.delta.map((v) => Math.round(v * 100)).join(",")}`);
				const p = tcp(m.arm);
				for (let i = 0; i < 3; i++) p[i] += m.delta[i];
				return {
					content: [{ type: "text", text: "obs" }, view, ...(arms ?? ["one"]).map(() => view)],
					details: {},
				};
			},
			state: async (arm) => ({ eef_xyz: [...tcp(arm)], gripper_width: 0.08, table_z: 0 }),
			instruction: () => "put the cube in the bowl",
			emptyWidthM: 0.005,
		},
	});
};
const oneArm = unitsToy();
const twoArms = unitsToy(["left", "right"]);

/**
 * The policy's service (vLLM's chat completions): `replies` in turn, then DONE; `asked` gets each request's body. The
 * service answers the n-th request once `during(n)` has run.
 */
async function service(replies: string[], during?: (n: number) => Promise<void>) {
	const asked: any[] = [];
	const server = createServer(async (req: IncomingMessage, res) => {
		let data = "";
		for await (const chunk of req) data += chunk;
		asked.push(JSON.parse(data));
		await during?.(asked.length).catch(() => {});
		res.writeHead(200, { "content-type": "application/json" });
		res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: replies.shift() ?? "DONE" } }] }));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	return {
		url: `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`,
		asked,
		close: () =>
			new Promise<void>((resolve) => {
				server.closeAllConnections();
				server.close(() => resolve());
			}),
	};
}

const GO = (x: number) => JSON.stringify({ name: "go", arguments: { x } });
const DONE = JSON.stringify({ name: "finish", arguments: { status: "success", summary: "done" } });
/** The tools surface: the toy's look opens the episode and runs in place of an unparsable reply. */
const TOOLS = { "ft-surface": "tools", "ft-observe": "look", "ft-model": "student", "ft-task": "push the cube" };
/** One arm, one token a step (the units surface). */
const ONE = { units: "true", "ft-surface": "units", "ft-model": "student", "ft-task": "put the cube in the bowl" };
/** Two arms, one token each (the units surface with a dual prompt). */
const DUAL = { ...ONE, "ft-prompt": "v4-dual-once" };
const texts = (m: { content: { type: string; text?: string }[] }) =>
	m.content.map((c) => (c.type === "text" ? c.text : "")).join("");

/**
 * One episode with `-e robot -e modes/finetuned`, the session's model set to finetuned/local (`model` names another)
 * once the session has started, and `flags`, the service answering `replies` (default: go 1, go 2, finish), in a new
 * session or one whose file is `entries`. `before` runs before the prompt, `waiting` while the loop's request for the
 * second step waits for the service, and `after` after the prompt, with the context pi gives an extension; `aux` are
 * the answers of the scripted model, which is asked only when a flag or the deployment names it. Returns what reached
 * the robot, the loop's replies, each call's result, what the service was asked, the session's entries of a type, the
 * robot's result row (written at pi's quit when the episode did not finish) and the session's file.
 */
async function run(
	o: {
		robot?: (log: string[]) => (pi: ExtensionAPI) => void;
		replies?: string[];
		flags?: Record<string, string | boolean>;
		aux?: string[];
		deployment?: Deployment;
		before?: (ctx: ExtensionContext, session: Session) => Promise<void>;
		waiting?: (ctx: ExtensionContext) => Promise<void>;
		after?: (ctx: ExtensionContext, session: Session) => Promise<void>;
		more?: ((pi: ExtensionAPI) => void)[];
		model?: string;
		argv?: string[];
		entries?: FileEntry[];
	} = {},
) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "finetuned-session-")));
	let context: ExtensionContext | undefined;
	// A copy: the service takes its replies off the list, and a test hands one list to several runs.
	const stub = await service([...(o.replies ?? [GO(1), GO(2), DONE])], async (n) => {
		if (n === 2 && context) await o.waiting?.(context);
	});
	const log: string[] = [];
	try {
		return await withSession(
			base,
			[(o.robot ?? ((l) => toolsToy(l)))(log), finetuned, ...(o.more ?? [])],
			async ({ session, faux, runtime, stderr, ctx }) => {
				faux.setResponses((o.aux ?? []).map((text) => fauxAssistantMessage(text)));
				const [provider, ...id] = (o.model ?? "finetuned/local").split("/");
				const planner = runtime.getModel(provider, id.join("/"));
				assert.ok(planner, `${o.model ?? "finetuned/local"} is registered`);
				await session.setModel(planner);
				context = ctx();
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
					robot: log,
					replies: session.messages.flatMap((m) =>
						m.role === "assistant"
							? [
									[
										m.stopReason,
										m.errorMessage ?? texts(m).replace(/ · \d+ ms/g, ""),
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
					asked: stub.asked,
					entries,
					row: entries(RESULT_ENTRY).at(-1) as Record<string, unknown> | undefined,
					stderr,
					file: [session.sessionManager.getHeader(), ...session.sessionManager.getEntries()] as FileEntry[],
				};
			},
			{
				flags: { ...TOOLS, ...o.flags },
				deployment: { ...o.deployment, services: { finetuned: stub.url } },
				...(o.argv ? { argv: o.argv } : {}),
				...(o.entries ? { entries: o.entries } : {}),
			},
		);
	} finally {
		await stub.close();
		rmSync(base, { recursive: true, force: true });
	}
}
/** What an episode did that a request from elsewhere must leave alone. */
const whole = (out: Awaited<ReturnType<typeof run>>) => ({
	robot: out.robot,
	replies: out.replies,
	results: out.results,
	asked: out.asked,
	steps: out.entries(STEP_ENTRY).map((s: Record<string, unknown>) => [s.step, s.call, s.raw, s.prompt]),
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

test("a real session: the dashboard's model check, a question to a VLM and pi's bug-report summary, each before the first step, while a step's call runs, between two steps, while the service answers a step and after the end, get the error; the policy's steps are those of a session without them", async () => {
	const alone = await run();
	assert.deepEqual(alone.robot, ["look", "go 1", "go 2"]);
	assert.equal(alone.row?.claimed, "success");
	const answers: string[] = [];
	let session: Session | undefined;
	/** The dashboard's model check, a question to a VLM and pi's summary for a bug report, in turn. */
	const ask = async (ctx: ExtensionContext) =>
		void answers.push(await answer(llm(ctx)), await answer(vlm(ctx)), await answer(bugReport(session as Session)));
	const out = await run({
		robot: (log) => toolsToy(log, { inside: async (ctx, x) => x === 1 && (await ask(ctx)) }),
		before: async (ctx, s) => {
			session = s;
			await ask(ctx);
		},
		more: [at("turn_end", 2, ask)],
		waiting: ask,
		after: ask,
	});
	// Before the first step, while go 1 runs, when its turn has ended, while the loop's request for the next step waits
	// for the service, and after the end.
	const each = [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`, `threw: Bug report summary failed: ${NOT_A_TURN}`];
	assert.deepEqual(answers, Array(5).fill(each).flat());
	assert.deepEqual(whole(out), whole(alone));
});

test("a real session: once its run has ended, the session's own next prompt is still its turn: it is told that the policy ended the episode, and the episode is as it was", async () => {
	const alone = await run();
	const out = await run({
		after: async (_ctx, session) => {
			await within(20_000, session.prompt("Go on."), "the second prompt");
		},
	});
	assert.deepEqual(out.replies, [...alone.replies, ["stop", "The fine-tuned policy already ended this episode.", []]]);
	assert.deepEqual(whole({ ...out, replies: alone.replies }), whole(alone));
});

test("a real session: on two arms the units verifier is on (by default, or with --units-verify=true) and, with no model named for it, its question at the finish gets the error: the finish is refused, the policy has ended and the run has no claim; with --aux-model, the deployment's aux.verify or --units-verify=false the run ends as a claimed success", async () => {
	const replies = ["MV_LEFT MV_RIGHT", "DONE DONE"];
	const refusal = `finish refused: the verifier is unavailable (${NOT_A_TURN}), so the task could not be checked. Call \`finish\` again to retry the check. (This is not the verifier's NOT complete verdict.)`;
	for (const flags of [DUAL, { ...DUAL, "units-verify": "true" }]) {
		const out = await run({ robot: twoArms, replies, flags });
		assert.equal(out.results.at(-1), `ft_3 finish ERROR: ${refusal}`);
		// The policy ended the episode with its finish: asked again, it says so and calls nothing.
		assert.deepEqual(out.replies.at(-1), ["stop", "The fine-tuned policy already ended this episode.", []]);
		assert.equal(out.row?.claimed, null);
		assert.equal(out.row?.verifier_error, NOT_A_TURN);
		assert.equal(out.row?.finish_verified, undefined);
		assert.equal(out.asked.length, 2, "the service was asked for the two steps, not for the verifier's question");
	}
	// With a model named for the verifier, by the flag or by the deployment's own role, that model checks the finish.
	const verdict = '{"complete": true, "reason": "the cube is in the bowl"}';
	for (const o of [
		{ flags: { ...DUAL, "aux-model": "faux/faux-1" } },
		{ deployment: { aux: { verify: "faux/faux-1" } } },
	]) {
		const named = await run({ robot: twoArms, replies, flags: DUAL, aux: [verdict, verdict], ...o });
		assert.equal(named.results.at(-1), "ft_3 finish: success");
		assert.equal(named.row?.claimed, "success");
		assert.equal(named.row?.finish_verified, true);
		assert.equal(named.entries("units_verify").at(-1)?.model, "faux/faux-1");
	}
	// With the verifier off the finish is taken as it is, and no model is asked.
	const off = await run({ robot: twoArms, replies, flags: { ...DUAL, "units-verify": "false" } });
	assert.equal(off.results.at(-1), "ft_3 finish: success");
	assert.equal(off.row?.claimed, "success");
	assert.equal(off.row?.finish_verified, undefined);
	assert.equal(off.row?.verifier_error, undefined);
	assert.deepEqual(off.entries("units_verify"), []);
});

test("a real session: a reviewed approval's question, with no model named for the reviewer, gets the error: the first motion does not run and the policy stops; with --aux-model or --approval-model the run is whole", async () => {
	const out = await run({ flags: { approval: "reviewed" } });
	assert.deepEqual(out.robot, ["look"]);
	assert.equal(
		out.results[1],
		`ft_2 go ERROR: go was not approved by the reviewer (error): the reviewer failed: ${NOT_A_TURN}. It did not run.`,
	);
	assert.deepEqual(out.replies.at(-1)?.[2], [], "the policy calls nothing more");
	assert.equal(out.row?.claimed, null);
	assert.equal(out.row?.approval_errors, 1);
	const approve = '{"decision": "approve", "reason": "the move is safe"}';
	for (const flag of ["aux-model", "approval-model"]) {
		const named = await run({ flags: { approval: "reviewed", [flag]: "faux/faux-1" }, aux: [approve, approve] });
		assert.deepEqual(named.robot, ["look", "go 1", "go 2"]);
		assert.equal(named.row?.claimed, "success");
	}
});

test("a real session on the units surface: a reviewed approval's question, with no model named for the reviewer, gets the error: the first motion does not run, and the policy quotes the refusal whole and stops", async () => {
	const out = await run({ robot: oneArm, replies: ["MV_LEFT", "DONE"], flags: { ...ONE, approval: "reviewed" } });
	const refusal = `act was not approved by the reviewer (error): the reviewer failed: ${NOT_A_TURN}. It did not run.`;
	assert.deepEqual(out.robot, []);
	assert.equal(out.results[0], `ft_1 act ERROR: ${refusal}`);
	assert.deepEqual(out.replies.at(-1), ["stop", `act failed: ${refusal}; the policy stops.`, []]);
	assert.equal(out.row?.claimed, null);
});

test("a real session: visual differencing's question, with no model named for it, gets the error: every step still runs, and each result it would have described says that it was not; with a model named for it, that model describes them", async () => {
	const robot = (log: string[]) => toolsToy(log, { spec: { vdm: { views: 1 } } });
	const out = await run({ robot, flags: { vdm: true } });
	assert.deepEqual(out.robot, ["look", "go 1", "go 2"]);
	assert.deepEqual(out.results, [
		`ft_1 look: scene[visual differencing unavailable: ${NOT_A_TURN}]`,
		`ft_2 go: at 1[visual differencing unavailable: ${NOT_A_TURN}]`,
		`ft_3 go: at 2[visual differencing unavailable: ${NOT_A_TURN}]`,
		"ft_4 finish: success",
	]);
	assert.equal(out.row?.claimed, "success");
	assert.equal(out.row?.vdm_errors, 3);
	// With a model named for it, by the flag or by the deployment's own role, that model describes each result.
	const named: Parameters<typeof run>[0][] = [
		{ flags: { vdm: true, "aux-model": "faux/faux-1" } },
		{ flags: { vdm: true }, deployment: { aux: { vdm: "faux/faux-1" } } },
	];
	for (const o of named) {
		const described = await run({
			robot,
			aux: ["A cube and a bowl.", "The arm moved.", "The arm moved again."],
			...o,
		});
		assert.deepEqual(described.robot, ["look", "go 1", "go 2"]);
		assert.match(described.results[0], /\nA cube and a bowl\.$/);
		assert.equal(described.row?.claimed, "success");
		assert.equal(described.row?.vdm_errors, 0);
	}
});

test("a real session behind the fallback planner with default flags (--model fallback/finetuned/local): the dashboard's model check and a question to a VLM get the error, nothing switches to the backup model, and the policy's steps are whole", async () => {
	const model = "fallback/finetuned/local";
	const argv = ["--model", model, "--fallback-model", "faux/faux-1"];
	const alone = await run({ model, argv });
	assert.deepEqual(alone.robot, ["look", "go 1", "go 2"]);
	const answers: string[] = [];
	const out = await run({
		model,
		argv,
		robot: (log) => toolsToy(log, { inside: async (ctx, x) => x === 1 && answers.push(await answer(vlm(ctx))) }),
		before: async (ctx) => void answers.push(await answer(llm(ctx))),
		more: [at("turn_end", 2, async (ctx) => void answers.push(await answer(llm(ctx))))],
	});
	assert.deepEqual(answers, [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`, `failed: ${NOT_A_TURN}`]);
	assert.deepEqual(whole(out), whole(alone));
	// Every turn was the policy's: four of them, none of the backup model's.
	assert.deepEqual(out.row?.planner_models, { primary: 4, fallback: 0 });
	assert.deepEqual(alone.row?.planner_models, { primary: 4, fallback: 0 });
	assert.deepEqual(out.entries("planner_fallback"), []);
	assert.deepEqual(
		out.stderr.filter((l) => l.startsWith("[fallback]")),
		[],
	);
});

/** The last result each request to the service carried (the tools surface's prompt names it). */
const lastResults = (asked: any[]) =>
	asked.map((body) =>
		body.messages[0].content
			.filter((c: any) => c.type === "text")
			.map((c: any) => /^Last result: .*$/m.exec(c.text)?.[0])
			.join(""),
	);

test("a real session over the file of an earlier one, under its id (--continue, --session, /resume) or a new one (a fork or clone), with the robot's frame budget and with a larger one: the policy's next episode goes on from the earlier one's last call id and runs on its own observations: each step is asked with its own last result, and the episode is whole and claims success", async () => {
	for (const keepImages of [2, 4]) {
		const first = await run({ robot: (log) => toolsToy(log, { spec: { keepImages } }) });
		assert.deepEqual(first.robot, ["look", "go 1", "go 2"]);
		assert.deepEqual(first.results, ["ft_1 look: scene", "ft_2 go: at 1", "ft_3 go: at 2", "ft_4 finish: success"]);
		const [header, ...entries] = first.file as [Record<string, unknown>, ...FileEntry[]];
		for (const id of [header.id, "0193e1c0-0000-7000-8000-00000000f02c"]) {
			const out = await run({
				robot: (log) => toolsToy(log, { spec: { keepImages }, tag: () => " (later)" }),
				entries: [{ ...header, id } as FileEntry, ...entries],
			});
			const what = `keep ${keepImages}, ${id === header.id ? "the same id" : "a new id"}`;
			assert.deepEqual(out.robot, ["look", "go 1", "go 2"], what);
			assert.deepEqual(
				out.replies.slice(first.replies.length).map((r) => r[2]),
				[
					[["ft_5", "look", {}]],
					[["ft_6", "go", { x: 1 }]],
					[["ft_7", "go", { x: 2 }]],
					[["ft_8", "finish", { status: "success", summary: "done" }]],
				],
				what,
			);
			assert.deepEqual(
				out.results.slice(first.results.length),
				["ft_5 look: scene (later)", "ft_6 go: at 1 (later)", "ft_7 go: at 2 (later)", "ft_8 finish: success"],
				what,
			);
			assert.deepEqual(
				lastResults(out.asked),
				["Last result: scene (later)", "Last result: at 1 (later)", "Last result: at 2 (later)"],
				what,
			);
			assert.equal(out.row?.claimed, "success", what);
		}
	}
});

test("a real session reloaded after an episode (/reload), with the robot's frame budget and with a larger one: the policy's next episode goes on from the earlier one's last call id and runs on its own observations: each step is asked with its own last result, and the episode is whole and claims success", async () => {
	for (const keepImages of [2, 4]) {
		let starts = 0;
		const start = async () => {
			starts++;
			return ["look", "go", "finish"];
		};
		const out = await run({
			robot: (log) => toolsToy(log, { spec: { keepImages, start }, tag: () => ` (start ${starts})` }),
			replies: [GO(1), GO(2), DONE, GO(1), GO(2), DONE],
			after: async (_ctx, session) => {
				await session.reload();
				await within(20_000, session.prompt("Again."), "the prompt after the reload");
			},
		});
		const what = `keep ${keepImages}`;
		assert.deepEqual(out.robot, ["look", "go 1", "go 2", "look", "go 1", "go 2"], what);
		assert.deepEqual(
			out.results,
			[
				"ft_1 look: scene (start 1)",
				"ft_2 go: at 1 (start 1)",
				"ft_3 go: at 2 (start 1)",
				"ft_4 finish: success",
				"ft_5 look: scene (start 2)",
				"ft_6 go: at 1 (start 2)",
				"ft_7 go: at 2 (start 2)",
				"ft_8 finish: success",
			],
			what,
		);
		assert.deepEqual(
			lastResults(out.asked),
			[1, 2].flatMap((n) => [
				`Last result: scene (start ${n})`,
				`Last result: at 1 (start ${n})`,
				`Last result: at 2 (start ${n})`,
			]),
			what,
		);
		assert.equal(out.row?.claimed, "success", what);
	}
});
