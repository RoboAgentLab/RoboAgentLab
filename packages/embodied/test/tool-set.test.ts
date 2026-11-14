// The robot's tool set (../src/infra/tool-set.ts): its grant rules, and a real pi session in which pi activates tools by itself.

import assert from "node:assert/strict";
import { existsSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { type FauxResponseFactory, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import {
	createAgentSession,
	createCodemodeExtension,
	createMcpExtension,
	createToolSearchExtension,
	DefaultResourceLoader,
	type ExtensionAPI,
	type ExtensionToolContext,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { LATEST_PROTOCOL_VERSION } from "@earendil-works/pi-mcp";
import { createInMemoryTransportPair } from "@earendil-works/pi-mcp/testing";
import { Type } from "typebox";
import { ownToolSet } from "../src/infra/tool-set.ts";
import { defineRobot } from "../src/robot.ts";
import { useDeployment } from "./helpers/deployment.ts";

/** pi's activation, reduced to its list: `self` is pi activating tools by itself. */
function fakePi(initial: string[] = []) {
	let active = [...initial];
	const pi = {
		setActiveTools: (names: string[]) => {
			active = [...new Set(names)];
		},
		getActiveTools: () => [...active],
	} as unknown as ExtensionAPI;
	return {
		pi,
		active: () => active,
		self: (...names: string[]) => {
			active = [...active, ...names];
		},
		/** pi setting the active tools itself (a reload, a person's /tools). */
		set: (names: string[]) => {
			active = [...names];
		},
	};
}

test("the grant is what the robot's code activates; what pi activates by itself is neither granted nor listed", () => {
	const f = fakePi(["read", "bash", "edit", "write"]);
	const set = ownToolSet(f.pi);
	assert.equal(ownToolSet(f.pi), set, "one grant per pi, wrapped once");
	assert.deepEqual(f.pi.getActiveTools(), [], "pi's defaults are not granted");
	f.pi.setActiveTools(["move", "finish", "read"]);
	assert.deepEqual(f.active(), ["move", "finish", "read"]);
	f.self("codemode", "bash");
	assert.deepEqual(f.pi.getActiveTools(), ["move", "finish", "read"]);
	assert.equal(set.granted("codemode"), false);
	// A module that adds its tool to the active ones grants that tool and drops pi's additions.
	f.pi.setActiveTools([...f.pi.getActiveTools(), "policy_goal"]);
	assert.deepEqual(f.active(), ["move", "finish", "read", "policy_goal"]);
	assert.equal(set.granted("policy_goal"), true);
	assert.equal(set.granted("bash"), false);
	f.self("mcp__fs__read_file");
	assert.deepEqual(set.withdraw(), ["mcp__fs__read_file"]);
	assert.deepEqual(f.active(), ["move", "finish", "read", "policy_goal"]);
	assert.deepEqual(set.withdraw(), [], "nothing left to withdraw");
	assert.deepEqual(f.active(), ["move", "finish", "read", "policy_goal"]);
	// A granted tool someone else deactivated stays off: withdrawing takes out only what is not granted.
	f.set(["finish", "read", "bash"]);
	assert.deepEqual(set.withdraw(), ["bash"]);
	assert.deepEqual(f.active(), ["finish", "read"]);
	f.pi.setActiveTools([]);
	assert.equal(set.granted("move"), false, "a failed start grants nothing");
});

const FINISH = {
	description: "finish",
	parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
	result: (p: { status: string }) => ({ content: [{ type: "text" as const, text: p.status }], details: p }),
};
const MCP_TOOL = "mcp__fs__read_file";

/**
 * A real pi session (pi's agent loop, extension runner, tools and built-in codemode, tool_search and
 * MCP extensions) running the robot base with a toy robot, a scripted model whose requests are kept
 * with every tool a declaration or removal in their transcript names, and a fake MCP server whose one
 * tool reads a file. A second extension,
 * loaded after the robot, registers a tool as an extension or an MCP server connecting late may: in its
 * before_agent_start (`register: "start"`: after the robot's withdrawal, before the first request), after
 * the first `move` (`"move"`), or at the end of a failed request, before pi retries it (`"retry"`). The
 * MCP server answers once that before_agent_start has run.
 */
async function session(o: {
	exposure?: "codemode" | "direct";
	tools?: string[];
	register?: "start" | "move" | "retry";
	/** The second extension activates its tool again in its turn_end, after the robot's, every turn. */
	reactivate?: boolean;
	/** The MCP server connects before the prompt (else once the robot's before_agent_start has run). */
	connected?: boolean;
}) {
	const base = realpathSync(mkdtempSync(join(tmpdir(), "tool-set-")));
	const secret = join(base, "secret.txt");
	writeFileSync(secret, "OUTSIDE\n");
	useDeployment({});
	const served: string[] = [];
	const ran: string[] = [];
	const robot = (pi: ExtensionAPI) => {
		pi.registerFlag("seed", { type: "string", default: "0" });
		const r = defineRobot(pi, {
			name: "toy",
			task: ["seed"],
			keepImages: 1,
			start: async () => ["move", "relay", "finish"],
			result: () => ({}),
			status: () => ({ step: 0 }),
			finish: FINISH,
		});
		r.tool("move", "move", Type.Object({ n: Type.Number() }), async () => {
			ran.push("move");
			return { content: [{ type: "text", text: "moved" }], details: {} };
		});
		// A robot tool its start leaves out, as a mode hides the robot's own tools (units, code).
		r.tool("hidden", "hidden", Type.Object({}), async () => {
			ran.push("hidden");
			return { content: [{ type: "text", text: "hidden ran" }], details: {} };
		});
		// A robot tool that calls another tool, as the persistent policy does (../src/capabilities/policy).
		pi.registerTool({
			name: "relay",
			label: "relay",
			description: "Call another tool.",
			parameters: Type.Object({ tool: Type.String(), args: Type.Record(Type.String(), Type.Unknown()) }),
			async execute(_id, p, signal, _update, ctx) {
				const out = await (ctx as ExtensionToolContext).executeTool(p.tool, p.args, { signal });
				const text = out.result.content.map((c) => (c.type === "text" ? c.text : "")).join("");
				return { content: [{ type: "text", text: `${out.isError ? "error" : "ok"}: ${text}` }], details: {} };
			},
		});
	};
	let started: () => void = () => {};
	const robotWithdrew = new Promise<void>((resolve) => {
		started = resolve;
	});
	let registered = false;
	const late = (pi: ExtensionAPI) => {
		const register = () => {
			if (registered) return;
			registered = true;
			pi.registerTool({
				name: "late_tool",
				label: "late_tool",
				description: "A tool registered late.",
				parameters: Type.Object({}),
				async execute() {
					ran.push("late_tool");
					return { content: [{ type: "text", text: "late ran" }], details: {} };
				},
			});
		};
		pi.on("before_agent_start", () => {
			started();
			if (o.register === "start") register();
		});
		pi.on("tool_execution_end", (e) => {
			if (o.register === "move" && e.toolName === "move") register();
		});
		// No before_agent_start or turn_end runs between the failed request's agent_end and pi's retry of it.
		pi.on("agent_end", () => {
			if (o.register === "retry") register();
		});
		pi.on("turn_end", () => {
			if (!o.reactivate) return;
			register();
			pi.setActiveTools([...pi.getActiveTools(), "late_tool"]);
		});
	};
	const fakeServer = () => {
		const pair = createInMemoryTransportPair();
		const respond = (request: { method: string; params?: { arguments?: { path?: string } } }) => {
			if (request.method === "initialize")
				return {
					protocolVersion: LATEST_PROTOCOL_VERSION,
					capabilities: { tools: {} },
					serverInfo: { name: "fs", version: "1.0.0" },
				};
			if (request.method === "tools/list")
				return {
					tools: [
						{
							name: "read_file",
							description: "Read a file.",
							inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
						},
					],
				};
			if (request.method === "tools/call") {
				served.push(String(request.params?.arguments?.path));
				return { content: [{ type: "text", text: "SERVED" }] };
			}
			return {};
		};
		pair.server.onMessage((message: any) => {
			if (!("id" in message) || !("method" in message)) return;
			// Unless connected before the prompt: once the robot has withdrawn what pi had activated, during the
			// MCP extension's own wait for servers with direct exposure.
			void (o.connected ? Promise.resolve() : robotWithdrew).then(() =>
				pair.server.send({ jsonrpc: "2.0", id: message.id, result: respond(message) }),
			);
		});
		void pair.server.start();
		return pair.client;
	};
	const mcp = createMcpExtension({
		loadConfig: () => ({
			servers: o.exposure
				? [{ name: "fs", config: { url: "http://unused.invalid", exposure: o.exposure }, source: "test" }]
				: [],
			errors: [],
		}),
		createTransport: fakeServer as never,
	});
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
	const settingsManager = SettingsManager.inMemory({ retry: { enabled: true, maxRetries: 1, baseDelayMs: 1 } });
	// The robot first, as pi loads an `-e` extension before its built-ins.
	const resourceLoader = new DefaultResourceLoader({
		cwd: base,
		agentDir: join(base, "agent"),
		settingsManager,
		extensionFactories: [robot, late, createCodemodeExtension(), createToolSearchExtension(), mcp],
	});
	await resourceLoader.reload();
	const { session } = await createAgentSession({
		cwd: base,
		agentDir: join(base, "agent"),
		model,
		modelRuntime: runtime,
		sessionManager: SessionManager.inMemory(base),
		settingsManager,
		resourceLoader,
		...(o.tools ? { tools: o.tools } : {}),
	});
	const errors: string[] = [];
	const log = console.error;
	console.error = (...a: unknown[]) => errors.push(a.join(" "));
	await session.bindExtensions({});
	for (let i = 0; i < 200 && o.connected && !session.getAllTools().some((t) => t.name === MCP_TOOL); i++)
		await new Promise((r) => setTimeout(r, 10));
	const declared: string[][] = [];
	const results = () =>
		session.messages.flatMap((m) =>
			m.role === "toolResult"
				? [`${m.toolName}: ${m.content.map((c) => (c.type === "text" ? c.text : "")).join("")}`]
				: [],
		);
	return {
		base,
		secret,
		served,
		ran,
		session,
		results,
		/**
		 * The tools each request named to the planner, sorted: every tool a declaration or a removal in its transcript
		 * names, as a transport that replays tool changes sends them, not only the current ones.
		 */
		declared,
		/** What the robot told the operator (it has no UI here). */
		errors,
		run: async (...steps: ReturnType<typeof fauxAssistantMessage>[]) => {
			const kept = [...steps, fauxAssistantMessage("done")].map(
				(step): FauxResponseFactory =>
					(context) => {
						const named = context.messages.flatMap((m) =>
							m.role === "system" ? [...(m.toolsAdded ?? []), ...(m.toolsRemoved ?? [])].map((t) => t.name) : [],
						);
						declared.push([...new Set(named)].sort());
						return step;
					},
			);
			faux.setResponses(kept);
			await session.prompt("Solve the task.");
		},
		close: () => {
			console.error = log;
			session.dispose();
			faux.unregister();
			rmSync(base, { recursive: true, force: true });
		},
	};
}

const call = (name: string, args: Parameters<typeof fauxToolCall>[1]) =>
	fauxAssistantMessage([fauxToolCall(name, args)], { stopReason: "toolUse" });
const OWN = ["finish", "move", "relay"];

test("a real session: codemode is withdrawn before the run; an ungranted tool called through a robot tool is refused, a granted one runs", async (t) => {
	const s = await session({ exposure: "codemode", connected: true });
	t.after(s.close);
	assert.ok(s.session.getActiveToolNames().includes("codemode"), "pi activated codemode for the MCP server");
	await s.run(
		call("codemode", { code: `return await tools.${MCP_TOOL}({ path: ${JSON.stringify(s.secret)} });` }),
		call("relay", { tool: MCP_TOOL, args: { path: s.secret } }),
		call("relay", { tool: "move", args: { n: 1 } }),
	);
	assert.deepEqual(s.results(), [
		"codemode: Tool codemode not found",
		`relay: error: ${MCP_TOOL} is not one of the tools toy gives the planner.`,
		"relay: ok: moved",
	]);
	assert.deepEqual(s.served, [], "the MCP server was never asked");
	assert.deepEqual(s.ran, ["move"]);
	assert.deepEqual(s.declared, [OWN, OWN, OWN, OWN]);
	assert.deepEqual(s.errors, ["[toy] not offered to the planner (toy did not activate them): codemode"]);
});

test("a real session: an MCP server with direct exposure that connects during pi's startup wait is declared in no request, and refused", async (t) => {
	const s = await session({ exposure: "direct" });
	t.after(s.close);
	await s.run(call(MCP_TOOL, { path: s.secret }), call(MCP_TOOL, { path: s.secret }));
	assert.deepEqual(s.results(), [
		`${MCP_TOOL}: ${MCP_TOOL} is not one of the tools toy gives the planner.`,
		`${MCP_TOOL}: Tool ${MCP_TOOL} not found`,
	]);
	assert.deepEqual(s.served, []);
	assert.deepEqual(s.declared, [OWN, OWN, OWN]);
	assert.deepEqual(s.errors, [`[toy] not offered to the planner (toy did not activate them): ${MCP_TOOL}`]);
});

test("a real session: --tools entries pi activates again on a registration are declared in no request, refused in that turn and withdrawn after it, a robot's hidden tool among them", async (t) => {
	const s = await session({ tools: ["move", "relay", "finish", "hidden", "bash"], register: "start" });
	t.after(s.close);
	const marker = join(s.base, "bash-ran");
	await s.run(
		fauxAssistantMessage(
			[fauxToolCall("bash", { command: `touch ${JSON.stringify(marker)}` }), fauxToolCall("hidden", {})],
			{
				stopReason: "toolUse",
			},
		),
		call("bash", { command: `touch ${JSON.stringify(marker)}` }),
	);
	assert.deepEqual(s.results(), [
		"bash: bash is not one of the tools toy gives the planner.",
		"hidden: hidden is not one of the tools toy gives the planner.",
		"bash: Tool bash not found",
	]);
	assert.equal(existsSync(marker), false, "bash did not run");
	assert.deepEqual(s.ran, []);
	assert.deepEqual(s.declared, [OWN, OWN, OWN]);
});

test("a real session: a tool registered during a run is withdrawn after its turn, and at the start of the next run", async (t) => {
	const s = await session({ register: "move" });
	t.after(s.close);
	await s.run(call("move", { n: 1 }), call("late_tool", {}));
	assert.deepEqual(s.results(), ["move: moved", "late_tool: Tool late_tool not found"]);
	assert.deepEqual(s.declared, [OWN, OWN, OWN]);
	// pi activates it again between runs (a reload, another extension): the next run's first request does not declare it.
	s.session.setActiveToolsByName([...s.session.getActiveToolNames(), "late_tool"]);
	s.declared.length = 0;
	await s.run(call("late_tool", {}));
	assert.deepEqual(s.results().at(-1), "late_tool: Tool late_tool not found");
	assert.deepEqual(s.declared, [OWN, OWN]);
	assert.deepEqual(s.ran, ["move"]);
});

test("a real session: a tool another extension activates after the robot's turn_end, or registers before pi retries a failed request, is declared in no request", async (t) => {
	const again = await session({ reactivate: true });
	t.after(again.close);
	await again.run(call("move", { n: 1 }), call("late_tool", {}), call("late_tool", {}));
	assert.deepEqual(again.results(), [
		"move: moved",
		"late_tool: late_tool is not one of the tools toy gives the planner.",
		"late_tool: late_tool is not one of the tools toy gives the planner.",
	]);
	assert.deepEqual(again.declared, [OWN, OWN, OWN, OWN]);
	assert.deepEqual(again.errors, ["[toy] not offered to the planner (toy did not activate them): late_tool"]);
	const retried = await session({ register: "retry" });
	t.after(retried.close);
	await retried.run(
		fauxAssistantMessage("", { stopReason: "error", errorMessage: "overloaded" }),
		call("late_tool", {}),
	);
	assert.deepEqual(retried.results(), ["late_tool: late_tool is not one of the tools toy gives the planner."]);
	assert.deepEqual(retried.declared, [OWN, OWN, OWN]);
	assert.deepEqual([...again.ran, ...retried.ran], ["move"]);
});
