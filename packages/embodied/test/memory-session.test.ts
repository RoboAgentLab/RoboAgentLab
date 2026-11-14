// memory's grep and find in a real pi session (../src/capabilities/memory): pi's own getAllTools, tool registry and
// tool pipeline, which the guard's checks of the session's searches rely on.

import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import {
	createAgentSession,
	createGrepToolDefinition,
	DefaultResourceLoader,
	type ExtensionAPI,
	ModelRuntime,
	SessionManager,
	SettingsManager,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { defineRobot } from "../src/robot.ts";
import { useDeployment } from "./helpers/deployment.ts";

/** A toy robot with memory and a cell, alone or beside the SDK host's own tools, in a session whose model calls `steps`. */
async function session(customTools?: ToolDefinition[]) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "memory-session-")));
	const home = join(base, "memory");
	const run = join(base, "run");
	mkdirSync(run);
	mkdirSync(join(home, "toy", "global"), { recursive: true });
	writeFileSync(join(home, "toy", "global", "leaf.md"), "MARK a leaf\n");
	// A rule only a search that reads ignore files obeys: memory's finds the leaf, pi's default would not.
	writeFileSync(join(home, "toy", "global", ".ignore"), "leaf.md\n");
	useDeployment({});
	const robot = (pi: ExtensionAPI) => {
		const r = defineRobot(pi, {
			name: "toy",
			task: [],
			keepImages: 1,
			start: async () => ["move", "finish"],
			result: () => ({}),
			status: () => ({ step: 0 }),
			finish: {
				description: "finish",
				parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
				result: (p: { status: string }) => ({ content: [{ type: "text" as const, text: p.status }], details: p }),
			},
			memory: { home: () => home, cell: () => ({ tag: "toy_s0", reference: "toy_s0" }), published: false },
		});
		r.tool("move", "move", Type.Object({}), async () => ({
			content: [{ type: "text", text: "moved" }],
			details: {},
		}));
	};
	const faux = registerFauxProvider();
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
	const settingsManager = SettingsManager.inMemory();
	const resourceLoader = new DefaultResourceLoader({
		cwd: base,
		agentDir: join(base, "agent"),
		settingsManager,
		extensionFactories: [robot],
	});
	await resourceLoader.reload();
	const { session } = await createAgentSession({
		cwd: base,
		agentDir: join(base, "agent"),
		model,
		modelRuntime: runtime,
		// A session dir: the cell's output dir, where the guard lets the planner write its files.
		sessionManager: SessionManager.create(run, run),
		settingsManager,
		resourceLoader,
		...(customTools ? { customTools } : {}),
	});
	const stderr: string[] = [];
	const log = console.error;
	console.error = (...a: unknown[]) => stderr.push(a.join(" "));
	try {
		await session.bindExtensions({});
	} finally {
		console.error = log;
	}
	return {
		home,
		session,
		stderr,
		results: () =>
			session.messages.flatMap((m) =>
				m.role === "toolResult"
					? [`${m.toolName}: ${m.content.map((c) => (c.type === "text" ? c.text : "")).join("")}`]
					: [],
			),
		run: async (...steps: ReturnType<typeof fauxAssistantMessage>[]) => {
			faux.setResponses([...steps, fauxAssistantMessage("done")]);
			await session.prompt("Solve the task.");
		},
		close: () => {
			session.dispose();
			faux.unregister();
			process.exitCode = undefined;
			rmSync(base, { recursive: true, force: true });
		},
	};
}

/** Whether pi finds ripgrep without downloading it: in its bin dir or on PATH. */
const rg = ["", ...(process.env.PATH ?? "").split(":")].some((d) =>
	existsSync(join(d || join(process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"), "bin"), "rg")),
);

test(
	"in a real session the planner's grep is memory's, past an ignore rule, and the guard pi's pipeline consults lets it search the library and refuses a search outside it",
	{ skip: !rg && "needs ripgrep" },
	async (t) => {
		const s = await session();
		t.after(s.close);
		assert.deepEqual(s.stderr, []);
		const searches = s.session.getAllTools().filter((x) => x.name === "grep" || x.name === "find");
		assert.deepEqual(
			searches.map((x) => [x.name, x.description.includes("gitignore")]),
			[
				["grep", false],
				["find", false],
			],
		);
		assert.ok(s.session.getActiveToolNames().includes("grep"));
		await s.run(
			fauxAssistantMessage([fauxToolCall("grep", { pattern: "MARK", path: join(s.home, "toy", "global") })], {
				stopReason: "toolUse",
			}),
			// Outside the library: the guard, which pi's pipeline consults, refuses it.
			fauxAssistantMessage([fauxToolCall("grep", { pattern: "MARK", path: join(s.home, "..") })], {
				stopReason: "toolUse",
			}),
		);
		const [found, outside] = s.results();
		assert.equal(found, "grep: leaf.md:1: MARK a leaf");
		assert.match(outside ?? "", /^grep: only the memory and this cell's /);
	},
);

test("in a real session the SDK host's own grep stops the robot's start", async (t) => {
	// customTools take ToolDefinition's default type arguments; pi's own tests pass a typed tool so.
	const s = await session([createGrepToolDefinition(tmpdir()) as unknown as ToolDefinition]);
	t.after(s.close);
	assert.deepEqual(s.stderr, [
		"[toy] unavailable: this session's grep (<sdk:grep>) is not the memory guard's (pi's grep and find without ignore files, by their descriptions): load the robot first, and pass no grep or find as SDK customTools",
	]);
	assert.deepEqual(s.session.getActiveToolNames(), []);
});
