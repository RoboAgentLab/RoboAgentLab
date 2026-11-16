// The replay extension as the planner of a real pi session (pi's agent loop, extension runner and tools), over the
// robot base with a toy robot on robosuite's manifest: what a recorded turn does when its calls end as recorded, and
// when one does not.

import assert from "node:assert/strict";
import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import {
	createAgentSession,
	DefaultResourceLoader,
	type ExtensionAPI,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import replay from "../src/capabilities/replay/index.ts";
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

/** Replay `turns` with `-e robot -e replay --model replay/session --replay <file>`: what reached the robot's tool and each call's result. */
async function replayed(turns: Recorded[][]) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "replay-session-")));
	const file = record(base, turns);
	useDeployment({});
	const reached: unknown[] = [];
	const robot = (pi: ExtensionAPI) => {
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
	const faux = registerFauxProvider();
	const log = console.error;
	let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
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
		// The replay extension is loaded after the robot, as `-e robot -e replay` loads them.
		const resourceLoader = new DefaultResourceLoader({
			cwd: base,
			agentDir: join(base, "agent"),
			settingsManager,
			extensionFactories: [robot, replay],
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
		// --replay <file>, set as pi sets a flag given on the command line.
		session.extensionRunner.setFlagValue("replay", file);
		// pi reports a handler that throws to its error listeners only: without one, a throw would pass unseen.
		const thrown: string[] = [];
		await session.bindExtensions({ onError: (e) => void thrown.push(`${e.event}: ${e.error}`) });
		const planner = runtime.getModel("replay", "session");
		assert.ok(planner, "the replay provider's model is registered");
		await session.setModel(planner);
		await session.prompt("Replay the session.");
		assert.deepEqual(thrown, [], "no extension handler threw");
		const results = session.messages.flatMap((m) =>
			m.role === "toolResult"
				? [
						`${m.toolName}${m.isError ? " ERROR" : ""}: ${m.content.map((c) => (c.type === "text" ? c.text : "")).join("")}`,
					]
				: [],
		);
		return { results, reached };
	} finally {
		console.error = log;
		session?.dispose();
		faux.unregister();
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
