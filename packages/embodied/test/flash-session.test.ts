// Flash (flash/replay) as the planner of a real pi session (pi's agent loop, extension runner and tools), over the
// robot base with a toy robot that has a Flash hook (or one in units mode): what a request to Flash's model gets that
// is not a turn of the session's own loop, and what the robot's modules that ask the session's model do with it.

import assert from "node:assert/strict";
import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fauxAssistantMessage } from "@earendil-works/pi-ai/providers/faux";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import dashboard from "../src/capabilities/dashboard/index.ts";
import type { FlashCall, FlashHook, FlashProgram } from "../src/capabilities/flash/index.ts";
import { auxModel, type Deployment } from "../src/infra/config.ts";
import type { UnitsSpec } from "../src/modes/units/index.ts";
import { askVlm } from "../src/modes/units/vlm.ts";
import { defineRobot, RESULT_ENTRY, type RobotSpec } from "../src/robot.ts";
import { at, llm, PNG, type Session, said, vlm, within, withSession } from "./helpers/pi-session.ts";

/** What a request to Flash's model gets that is not a turn of the session's own loop. */
const NOT_A_TURN =
	"flash/replay replays a recorded plan and answers no other request; --aux-model names a model for the checks that take one";
/** How long a session, or a wait inside one, may take: each ends in well under a second. */
const MS = 10_000;

const FINISH = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p: { status: string }) => ({ content: [{ type: "text" as const, text: p.status }], details: p }),
};

/** A point where the toy's plan waits for the test: `reached` once the plan is there, `open` lets it go on. */
type Hold = { reached: Promise<void>; open: () => void; wait: () => Promise<void> };
function hold(): Hold {
	let arrive = () => {};
	let go = () => {};
	const reached = new Promise<void>((resolve) => {
		arrive = resolve;
	});
	const gate = new Promise<void>((resolve) => {
		go = resolve;
	});
	return {
		reached,
		open: () => go(),
		wait: () => {
			arrive();
			return gate;
		},
	};
}

type ToyOptions = {
	/** Where the plan waits for the test: before its first call (as a re-anchoring service call would), before go 2's rewrite. */
	holds?: { start?: Hold; rewrite?: Hold };
	/** `go` to these x asks the session's model while it runs, as check_attached does (aux.attach, else --aux-model). */
	asksAt?: number[];
	/** Camera images every result carries. */
	views?: number;
	spec?: Partial<RobotSpec>;
};

/**
 * The toy robot with a Flash hook. Its plan is go 1, 2, 3; `look` locates the anchor at 10, each go is shifted by it,
 * so the robot gets look, go 11, go 12, go 13, and go 13 solves the task. `reached` gets every call that ran, and
 * `answers` what the tool's question got.
 */
const toy =
	(o: ToyOptions = {}) =>
	(reached: string[], answers: string[]) =>
	(pi: ExtensionAPI) => {
		const hook: FlashHook<FlashProgram> = {
			load: () => ({ name: "toy", plan: [1, 2, 3].map((x) => ({ action: "go", arguments: { x } })) }),
			async start(_program, robot) {
				await o.holds?.start?.wait();
				const seen = await robot.move({ name: "look", arguments: {} });
				const anchor = Number((seen.json as { anchor?: number }).anchor);
				robot.note(`anchor at ${anchor}`);
				return {
					localized: 1,
					rewrite: async (entry): Promise<FlashCall> => {
						if (entry.arguments.x === 2) await o.holds?.rewrite?.wait();
						return { name: entry.action, arguments: { x: Number(entry.arguments.x) + anchor } };
					},
				};
			},
			over: (latest) => (latest.json as { solved?: boolean }).solved === true,
			solved: (latest) => (latest.json as { solved?: boolean }).solved === true,
		};
		const images = () =>
			Array.from({ length: o.views ?? 0 }, () => ({ type: "image" as const, data: PNG, mimeType: "image/png" }));
		const r = defineRobot(pi, {
			name: "toy",
			task: [],
			keepImages: 1,
			start: async () => ["look", "go", "finish"],
			result: () => ({}),
			status: () => ({ step: 0 }),
			finish: FINISH,
			flash: hook,
			...o.spec,
		});
		r.tool("look", "look", Type.Object({}), async () => {
			reached.push("look");
			return { content: [{ type: "text", text: JSON.stringify({ anchor: 10 }) }, ...images()], details: {} };
		});
		r.tool("go", "go", Type.Object({ x: Type.Number() }), async (p: { x: number }, signal, ctx) => {
			reached.push(`go ${p.x}`);
			if (o.asksAt?.includes(p.x))
				answers.push(
					await said(() => askVlm(ctx, auxModel(pi, "attach"), pi.getThinkingLevel(), "Held?", [], signal)),
				);
			return {
				content: [{ type: "text", text: JSON.stringify({ x: p.x, solved: p.x === 13 }) }, ...images()],
				details: {},
			};
		});
	};

type Flashed = {
	reached: string[];
	answers: string[];
	/** The calls of the session's replies, as `id name(arguments)` (a duration in a summary as N s). */
	calls: string[];
	/** Each call's result, as `name[ ERROR]: text`. */
	results: string[];
	/** The text of the session's replies, line by line. */
	said: string[];
	entries: (type: string) => any[];
	row: Record<string, any> | undefined;
	stderr: string[];
	/** The requests the scripted model got. */
	scripted: number;
};

/**
 * One session with `-e toy [-e more] --model flash/replay` (or `planner`, with `argv` as pi's command line) and `flags`.
 * `before` runs once that model plans and before the prompt, with the context pi gives an extension and the session's
 * stderr lines so far; `aux` are the answers of the scripted model, which is asked only when a flag or the deployment
 * names it; `drive` replaces the prompt. Returns what reached the robot, what the askers got, the session's calls and
 * results, and the robot's result row (written at pi's quit, which also comes when the session fails).
 */
async function flashing(
	o: {
		robot?: (reached: string[], answers: string[]) => (pi: ExtensionAPI) => void;
		flags?: Record<string, string | boolean>;
		deployment?: Deployment;
		aux?: string[];
		before?: (ctx: ExtensionContext, session: Session, answers: string[], stderr: string[]) => Promise<void>;
		more?: (answers: string[]) => ((pi: ExtensionAPI) => void)[];
		drive?: (session: Session, ctx: ExtensionContext, answers: string[]) => Promise<void>;
		argv?: string[];
		/** The session's model, as provider and id (default flash/replay). */
		planner?: [string, string];
	} = {},
): Promise<Flashed> {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "flash-session-")));
	const reached: string[] = [];
	const answers: string[] = [];
	try {
		return await withSession(
			base,
			[(o.robot ?? toy())(reached, answers), ...(o.more?.(answers) ?? [])],
			async ({ session, faux, runtime, stderr, ctx }) => {
				faux.appendResponses((o.aux ?? []).map((text) => fauxAssistantMessage(text)));
				const [provider, id] = o.planner ?? ["flash", "replay"];
				const planner = runtime.getModel(provider, id);
				assert.ok(planner, `${provider}/${id} is registered`);
				await session.setModel(planner);
				const quit = () => session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
				try {
					await o.before?.(ctx(), session, answers, stderr);
					const run = o.drive ? o.drive(session, ctx(), answers) : session.prompt("Solve the task.");
					await within(MS, run, "the session");
				} catch (err) {
					await quit();
					throw err;
				}
				await quit();
				const texts = (m: { content: { type: string; text?: string }[] }) =>
					m.content.map((c) => (c.type === "text" ? c.text : "")).join("");
				const entries = (type: string) =>
					(session.sessionManager.getEntries() as { type: string; customType?: string; data?: any }[]).flatMap(
						(e) => (e.type === "custom" && e.customType === type ? [e.data] : []),
					);
				return {
					reached,
					answers,
					calls: session.messages.flatMap((m) =>
						m.role === "assistant"
							? m.content.flatMap((c) =>
									c.type === "toolCall"
										? [`${c.id} ${c.name}(${JSON.stringify(c.arguments)})`.replace(/\d+\.\d s"/, 'N s"')]
										: [],
								)
							: [],
					),
					results: session.messages.flatMap((m) =>
						m.role === "toolResult" ? [`${m.toolName}${m.isError ? " ERROR" : ""}: ${texts(m)}`] : [],
					),
					said: session.messages.flatMap((m) => (m.role === "assistant" ? texts(m).split("\n") : [])),
					entries,
					row: entries(RESULT_ENTRY).at(-1),
					stderr,
					scripted: faux.state.callCount,
				};
			},
			{ flags: o.flags, deployment: o.deployment, argv: o.argv },
		);
	} finally {
		rmSync(base, { recursive: true, force: true });
	}
}

/** The toy's plan as the session sends it with nothing else asking Flash: each call with its id, then finish. */
const CALLS = [
	"flash_1 look({})",
	'flash_2 go({"x":11})',
	'flash_3 go({"x":12})',
	'flash_4 go({"x":13})',
	'flash_5 finish({"status":"success","summary":"replayed the toy program: 3 actions, 1 anchors re-localized, N s"})',
];
/** The run is the plan's, whole: every call reached the robot once, under the same ids, and finished as a success. */
function assertWhole(out: Flashed) {
	assert.deepEqual(out.calls, CALLS);
	assert.deepEqual(out.reached, ["look", "go 11", "go 12", "go 13"]);
	assert.deepEqual(out.results.at(-1), "finish: success");
	assert.equal(out.row?.claimed, "success");
	assert.equal(out.row?.planner_error, null);
}

/** A question to the session's model whose asker gives up once it has the answer: the answer, then the asker aborts. */
const askThenAbort = async (ctx: ExtensionContext) => {
	const asker = new AbortController();
	const answer = await said(() => askVlm(ctx, "", "off", "Is the cube in the bowl?", [], asker.signal));
	asker.abort();
	return answer;
};

test("a real session: a request before the session's first turn (the dashboard's model check, a question to a VLM, pi's bug-report summary) gets the error, and the plan runs whole from its first call", async () => {
	const out = await flashing({
		before: async (ctx, session, answers) =>
			void answers.push(
				await llm(ctx),
				await vlm(ctx),
				await session
					.summarizeForBugReport({ signal: new AbortController().signal })
					.catch((err: Error) => `threw: ${err.message}`),
			),
	});
	assert.deepEqual(out.answers, [
		`failed: ${NOT_A_TURN}`,
		`error: ${NOT_A_TURN}`,
		`threw: Bug report summary failed: ${NOT_A_TURN}`,
	]);
	assertWhole(out);
});

test("a real session: a request while the session's turn waits for the plan's next call is answered at once, and that turn then gets the call: the plan runs whole", async () => {
	const holds = { start: hold(), rewrite: hold() };
	const out = await flashing({
		robot: toy({ holds }),
		drive: async (session, ctx, answers) => {
			const run = session.prompt("Solve the task.");
			// The session's first turn waits while the hook locates the anchor: the dashboard's model check then.
			await within(MS, holds.start.reached, "the plan's start");
			answers.push(await within(MS, llm(ctx), "the model check"));
			holds.start.open();
			// A later turn waits while the hook rewrites go 2: a question whose asker gives up once answered.
			await within(MS, holds.rewrite.reached, "go 2's rewrite");
			answers.push(await within(MS, askThenAbort(ctx), "the question"));
			holds.rewrite.open();
			await run;
		},
	});
	assert.deepEqual(out.answers, [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`]);
	assertWhole(out);
});

test("a real session: a request while a call of the session runs (a tool that asks the session's model, a model check as a call starts, a question as one ends whose asker then gives up) gets the error, and the plan runs whole; a model named for the tool's question (--aux-model, the deployment's aux.attach) answers it", async () => {
	const out = await flashing({
		robot: toy({ asksAt: [12] }),
		more: (answers) => [
			at("tool_execution_start", 2, async (ctx) => void answers.push(await llm(ctx))),
			at("tool_execution_end", 3, async (ctx) => void answers.push(await askThenAbort(ctx))),
		],
	});
	// The check as go 11 starts, go 12's own question, the question as go 12 ends.
	assert.deepEqual(out.answers, [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`]);
	assertWhole(out);
	for (const named of [
		await flashing({ robot: toy({ asksAt: [12] }), flags: { "aux-model": "faux/faux-1" }, aux: ["held"] }),
		await flashing({ robot: toy({ asksAt: [12] }), deployment: { aux: { attach: "faux/faux-1" } }, aux: ["held"] }),
	]) {
		assert.deepEqual(named.answers, ["held"]);
		assertWhole(named);
	}
});

test("a real session: a request between two turns (a model check at a turn's end, a question from a context handler right before the session's next request) gets the error, and the session's next turn is the plan's next call", async () => {
	const out = await flashing({
		more: (answers) => [
			at("turn_end", 2, async (ctx) => void answers.push(await llm(ctx))),
			at("context", 3, async (ctx) => void answers.push(await vlm(ctx))),
		],
	});
	assert.deepEqual(out.answers, [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`]);
	assertWhole(out);
});

/** POST to `path` of the dashboard at `url` on a connection of its own: the status and the JSON answer. */
const post = (url: string, path: string) =>
	new Promise<{ status: number; json: any }>((resolve, reject) => {
		const req = request(
			`${url}${path}`,
			{ method: "POST", agent: false, headers: { "Content-Type": "application/json" } },
			(res) => {
				let raw = "";
				res.on("data", (chunk) => {
					raw += chunk;
				});
				res.on("end", () => resolve({ status: res.statusCode ?? 0, json: JSON.parse(raw) }));
			},
		);
		req.on("error", reject);
		req.end("{}");
	});

test("a real session with the dashboard: its model check (POST /llm-check) shows Flash's error before the session's first turn, between two turns and once the plan has finished, and the plan runs whole", async () => {
	let url = "";
	const check = async () => {
		const r = await within(MS, post(url, "llm-check"), "the dashboard's model check");
		return `${r.status} ${r.json.ok ? "ok" : "failed"} ${r.json.model}: ${r.json.detail}`;
	};
	const out = await flashing({
		flags: { dashboard: true },
		before: async (_ctx, _session, answers, stderr) => {
			// Without a UI the dashboard prints its address.
			url = (stderr.find((l) => l.startsWith("[dashboard] ")) ?? "").slice("[dashboard] ".length);
			answers.push(await check());
		},
		more: (answers) => [
			dashboard,
			at("turn_end", 2, async () => void answers.push(await check())),
			at("agent_end", 1, async () => void answers.push(await check())),
		],
	});
	assert.deepEqual(out.answers, Array(3).fill(`200 failed flash/replay: ${NOT_A_TURN}`));
	assertWhole(out);
});

test("a real session: once the plan has finished a request gets the error, and the session's own next prompt is told that the replay already ran", async () => {
	const out = await flashing({
		more: (answers) => [at("agent_end", 1, async (ctx) => void answers.push(await llm(ctx), await vlm(ctx)))],
		drive: async (session) => {
			await session.prompt("Solve the task.");
			await session.prompt("Go on.");
		},
	});
	assert.deepEqual(out.answers, [`failed: ${NOT_A_TURN}`, `error: ${NOT_A_TURN}`]);
	assertWhole(out);
	assert.equal(out.said.at(-1), "Flash replay already ran in this session.");
});

test("a real session: pi's own summaries ask the session's model and fail with the error: the bug report's between two turns and once the plan has finished, and a compaction afterwards; the plan runs whole", async () => {
	let session: Session | undefined;
	const summary = () =>
		(session as Session)
			.summarizeForBugReport({ signal: new AbortController().signal })
			.catch((err: Error) => `threw: ${err.message}`);
	const out = await flashing({
		before: async (_ctx, s) => {
			session = s;
		},
		more: (answers) => [
			at("turn_end", 2, async () => void answers.push(await summary())),
			at("agent_end", 1, async () => void answers.push(await summary())),
		],
		drive: async (s, _ctx, answers) => {
			await s.prompt("Solve the task.");
			// Nothing kept as it is: the whole session is to be summarized.
			s.settingsManager.applyOverrides({ compaction: { keepRecentTokens: 0 } } as never);
			answers.push(
				await s.compact().then(
					() => "compacted",
					(err: Error) => `threw: ${err.message}`,
				),
			);
		},
	});
	assert.deepEqual(out.answers, [
		`threw: Bug report summary failed: ${NOT_A_TURN}`,
		`threw: Bug report summary failed: ${NOT_A_TURN}`,
		`threw: Turn prefix summarization failed: ${NOT_A_TURN}`,
	]);
	assertWhole(out);
});

const VECTORS: UnitsSpec["vectors"] = {
	MV_FWD: [1, 0, 0],
	MV_BACK: [-1, 0, 0],
	MV_LEFT: [0, -1, 0],
	MV_RIGHT: [0, 1, 0],
	MV_UP: [0, 0, 1],
	MV_DOWN: [0, 0, -1],
};
/** The flags of a robot in pure units mode with the finish verifier at its default (auto: on for two arms). */
const UNITS = { units: "true", "units-plugins": "recovery,auto_release,proprioception,plan,mem_text" };
/**
 * A toy in the real units mode with its real finish verifier, on `arms` (one arm without names); its plan is a units
 * sequence given by --flash-plan, the toy has no plan directory. `reached` gets every unit an arm ran.
 */
const unitsToy = (arms: string[]) => (reached: string[]) => (pi: ExtensionAPI) => {
	const pos: Record<string, number[]> = Object.fromEntries(arms.map((a, i) => [a, [0.5, i ? -0.2 : 0.2, 0.2]]));
	defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 2,
		start: async () => ["finish"],
		result: () => ({}),
		finish: FINISH,
		flash: {
			load: () => {
				throw new Error("the toy has no plan directory");
			},
			start: async () => ({ localized: 0, rewrite: (e) => ({ name: e.action, arguments: e.arguments }) }),
			over: () => false,
			solved: () => true,
		},
		units: {
			vectors: VECTORS,
			stepM: 0.02,
			...(arms.length > 1 ? { arms } : {}),
			apply: async (m) => {
				const arm = (m as { arm?: string }).arm ?? arms[0];
				reached.push(`${arm} ${m.delta.join(",")}`);
				for (let i = 0; i < 3; i++) pos[arm][i] += m.delta[i];
				return {
					content: [
						{ type: "text", text: "{}" },
						{ type: "image", data: PNG, mimeType: "image/png" },
					],
					details: {},
				};
			},
			state: async (arm) => ({ eef_xyz: [...pos[arm ?? arms[0]]], gripper_width: 0.08, table_z: 0 }),
			instruction: () => "put the cube in the bowl",
			emptyWidthM: 0.005,
		},
	});
};
/** Replay a one-unit sequence (MV_LEFT, on the first arm) on `arms` with UNITS and `o`'s flags. */
async function unitsRun(arms: string[], o: Parameters<typeof flashing>[0] = {}) {
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "flash-units-")));
	try {
		const plan = join(dir, "cell_units.json");
		const unit = { unit: "MV_LEFT", ...(arms.length > 1 ? { arm: arms[0] } : {}) };
		writeFileSync(plan, JSON.stringify({ kind: "units", units: [unit] }));
		return await flashing({ ...o, robot: unitsToy(arms), flags: { ...UNITS, "flash-plan": plan, ...o.flags } });
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
}

test("a real session: the units verifier, on two arms with the default flags or on one with --units-verify=true, asks the session's model at a replayed finish: the finish is refused and the episode has no claim; with a model named for it (--aux-model, the deployment's aux.verify) it is checked, and with --units-verify=false, or on one arm by default, nothing is asked", async () => {
	const refusal = `finish refused: the verifier is unavailable (${NOT_A_TURN}), so the task could not be checked. Call \`finish\` again to retry the check. (This is not the verifier's NOT complete verdict.)`;
	for (const [arms, flags] of [
		[["left", "right"], {}],
		[["arm"], { "units-verify": "true" }],
	] as const) {
		const out = await unitsRun([...arms], { flags });
		assert.equal(out.reached[0], `${arms[0]} 0,-0.02,0`);
		assert.equal(out.results.at(-1), `finish ERROR: ${refusal}`);
		// Flash plans its finish once: the session's next turn is its closing line.
		assert.equal(out.said.at(-1), "Flash replay already ran in this session.");
		assert.equal(out.row?.claimed, null);
		assert.equal(out.row?.verifier_error, NOT_A_TURN);
		assert.equal(out.row?.finish_verified, undefined);
		assert.deepEqual(
			out.entries("units_verify").map((e) => [e.refused, e.verifier_error]),
			[[true, NOT_A_TURN]],
		);
	}
	const complete = '{"complete": true, "reason": "the cube is in the bowl"}';
	for (const named of [
		await unitsRun(["left", "right"], { flags: { "aux-model": "faux/faux-1" }, aux: [complete] }),
		await unitsRun(["left", "right"], { deployment: { aux: { verify: "faux/faux-1" } }, aux: [complete] }),
	]) {
		assert.equal(named.results.at(-1), "finish: success");
		assert.equal(named.row?.claimed, "success");
		assert.equal(named.row?.finish_verified, true);
	}
	// Switched off, and on one arm by default, the verifier asks nothing: the finish is the replayed one.
	for (const off of [
		await unitsRun(["left", "right"], { flags: { "units-verify": "false" } }),
		await unitsRun(["arm"]),
	]) {
		assert.equal(off.results.at(-1), "finish: success");
		assert.equal(off.row?.claimed, "success");
		assert.equal(off.row?.verifier_error, undefined);
		assert.deepEqual(off.entries("units_verify"), []);
	}
});

test("a real session: a reviewed approval's question is not answered by Flash: the first motion is not approved and the replay ends as a failure that says why; with a model named for the reviewer (--approval-model, --aux-model) that approves, the plan runs whole", async () => {
	const refused = `go was not approved by the reviewer (error): the reviewer failed: ${NOT_A_TURN}. It did not run.`;
	const out = await flashing({ flags: { approval: "reviewed" } });
	// The gate looks before it asks (the toy's look); no motion reached the robot.
	assert.deepEqual(
		out.reached.filter((c) => c !== "look"),
		[],
	);
	assert.deepEqual(out.results, ['look: {"anchor":10}', `go ERROR: ${refused}`, "finish: failure"]);
	assert.equal(out.row?.claimed, "failure");
	assert.equal(out.row?.summary, `flash error: go failed: ${refused}`);
	const approve = '{"decision": "approve", "reason": "the move is safe"}';
	for (const flag of ["approval-model", "aux-model"]) {
		const named = await flashing({
			flags: { approval: "reviewed", [flag]: "faux/faux-1" },
			aux: [approve, approve, approve],
		});
		assert.deepEqual(named.calls, CALLS);
		assert.deepEqual(
			named.reached.filter((c) => c !== "look"),
			["go 11", "go 12", "go 13"],
		);
		assert.equal(named.results.at(-1), "finish: success");
		assert.equal(named.row?.claimed, "success");
	}
});

test("a real session: visual differencing's question is not answered by Flash: every planned call reaches the robot and each result it would have described says why it was not; with a model named for it (--aux-model, the deployment's aux.vdm), that model describes them", async () => {
	const seen = toy({ views: 1, spec: { vdm: { views: 1 } } });
	const out = await flashing({ robot: seen, flags: { vdm: true } });
	assertWhole(out);
	assert.deepEqual(
		out.results.slice(0, -1).map((r) => r.slice(r.indexOf("}") + 1)),
		Array(4).fill(`[visual differencing unavailable: ${NOT_A_TURN}]`),
	);
	assert.equal(out.row?.vdm_errors, 4);
	const aux = ["A cube and a bowl.", "The arm moved.", "The arm moved.", "The arm moved."];
	for (const named of [
		await flashing({ robot: seen, flags: { vdm: true, "aux-model": "faux/faux-1" }, aux }),
		await flashing({ robot: seen, flags: { vdm: true }, deployment: { aux: { vdm: "faux/faux-1" } }, aux }),
	]) {
		assertWhole(named);
		assert.deepEqual(
			named.results.slice(0, -1).map((r) => r.split("\n").at(-1)),
			aux,
		);
		assert.equal(named.row?.vdm_errors, 0);
	}
});

test("a real session behind the fallback planner (--model fallback/flash/replay --fallback-model <scripted>, default flags): a request that is not the session's turn gets Flash's error back, counts as no failure and switches nothing; the plan runs whole", async () => {
	const out = await flashing({
		argv: ["--model", "fallback/flash/replay", "--fallback-model", "faux/faux-1"],
		planner: ["fallback", "flash/replay"],
		robot: toy({ asksAt: [12] }),
		before: async (ctx, _session, answers) => void answers.push(await llm(ctx)),
		more: (answers) => [at("turn_end", 2, async (ctx) => void answers.push(await llm(ctx), await vlm(ctx)))],
	});
	assert.deepEqual(out.answers, [
		`failed: ${NOT_A_TURN}`,
		`failed: ${NOT_A_TURN}`,
		`error: ${NOT_A_TURN}`,
		`error: ${NOT_A_TURN}`,
	]);
	assertWhole(out);
	// Every turn was Flash's; the backup was never asked, and no switch was noted or written.
	assert.deepEqual(out.row?.planner_models, { primary: 5, fallback: 0 });
	assert.equal(out.scripted, 0);
	assert.deepEqual(out.entries("planner_fallback"), []);
	assert.deepEqual(
		out.stderr.filter((l) => l.startsWith("[fallback]")),
		[],
	);
});
