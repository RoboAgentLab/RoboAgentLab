// The replay extension as the planner of a real pi session (pi's agent loop, extension runner and tools), over the
// robot base with a toy robot on robosuite's manifest: what a recorded turn does when its calls end as recorded, and
// when one does not; and which calls of a session pi itself recorded reach the robot again.

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import {
	createAgentSession,
	DefaultResourceLoader,
	type ExtensionAPI,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import replay, { loadRecording } from "../src/capabilities/replay/index.ts";
import { defineRobot } from "../src/robot.ts";
import { useDeployment } from "./helpers/deployment.ts";

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

/** A recorded session of `turns` (each a batch of calls with their results), as pi writes one. */
function record(dir: string, turns: Recorded[][]) {
	const lines: unknown[] = [
		{ type: "session", version: 3, id: "s", cwd: dir },
		{ type: "message", id: "u", parentId: null, message: { role: "user", content: "Solve." } },
	];
	let leaf = "u";
	turns.forEach((calls, i) => {
		const content = calls.map((c, k) => ({
			type: "toolCall",
			id: `rec_${i}_${k}`,
			name: c.name,
			arguments: c.arguments,
		}));
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

type Session = Awaited<ReturnType<typeof createAgentSession>>["session"];

/** The toy robot; `reached` gets every call its move_to ran. */
const toy = (reached: unknown[]) => (pi: ExtensionAPI) => {
	const r = defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 1,
		manifest: "robosuite",
		vars: () => ({ arms: [] }),
		start: async () => ["move_to", "finish"],
		result: () => ({}),
		status: () => ({ step: 0 }),
		finish: FINISH,
	});
	r.primitive("move_to", async (p: { xyz: number[] }) => {
		reached.push(p);
		if (p.xyz[0] === 9) throw new Error(IK);
		return { content: [{ type: "text", text: "moved" }], details: {} };
	});
};

/**
 * A real pi session in `base` with `extensions` and a scripted model, handed to `run`; pi's stderr lines are kept
 * off the test's output, and a handler pi reports as failed fails the test.
 */
async function withSession<T>(
	base: string,
	extensions: ((pi: ExtensionAPI) => void)[],
	run: (o: { session: Session; faux: ReturnType<typeof registerFauxProvider>; runtime: ModelRuntime }) => Promise<T>,
	flags: Record<string, string> = {},
): Promise<T> {
	useDeployment({});
	const faux = registerFauxProvider();
	const log = console.error;
	let session: Session | undefined;
	try {
		const model = faux.getModel();
		const runtime = await ModelRuntime.create({ authPath: join(base, "auth.json"), modelsPath: null });
		runtime.registerProvider(model.provider, {
			baseUrl: model.baseUrl,
			apiKey: "faux-key",
			api: faux.api,
			models: faux.models.map((m) => ({
				id: m.id,
				name: m.name,
				api: m.api,
				reasoning: m.reasoning,
				input: m.input,
				cost: m.cost,
				contextWindow: m.contextWindow,
				maxTokens: m.maxTokens,
				baseUrl: m.baseUrl,
			})),
		});
		const settingsManager = SettingsManager.inMemory({ retry: { enabled: false, maxRetries: 0, baseDelayMs: 1 } });
		const resourceLoader = new DefaultResourceLoader({
			cwd: base,
			agentDir: join(base, "agent"),
			settingsManager,
			extensionFactories: extensions,
		});
		await resourceLoader.reload();
		({ session } = await createAgentSession({
			cwd: base,
			agentDir: join(base, "agent"),
			model,
			modelRuntime: runtime,
			sessionManager: SessionManager.inMemory(base),
			settingsManager,
			resourceLoader,
		}));
		console.error = () => {};
		// Flags set as pi sets the ones given on the command line.
		for (const [name, value] of Object.entries(flags)) session.extensionRunner.setFlagValue(name, value);
		// pi reports a handler that throws to its error listeners only: without one, a throw would pass unseen.
		const thrown: string[] = [];
		await session.bindExtensions({ onError: (e) => void thrown.push(`${e.event}: ${e.error}`) });
		const out = await run({ session, faux, runtime });
		assert.deepEqual(thrown, [], "no extension handler threw");
		return out;
	} finally {
		console.error = log;
		session?.dispose();
		faux.unregister();
	}
}

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
		{ replay: file },
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
