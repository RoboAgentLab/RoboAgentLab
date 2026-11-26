/**
 * Real pi sessions for tests: pi's agent loop, extension runner and tools through createAgentSession, with
 * pi-ai's scripted faux provider as the model a test scripts, and the requests other code sends to the
 * session's model outside its loop (the dashboard's check, a question to a VLM, a request through the registry).
 */

import assert from "node:assert/strict";
import { join } from "node:path";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import {
	createAgentSession,
	DefaultResourceLoader,
	type ExtensionAPI,
	type ExtensionContext,
	type FileEntry,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { llmCheck } from "../../src/infra/check.ts";
import type { Deployment } from "../../src/infra/config.ts";
import { askVlm } from "../../src/modes/units/vlm.ts";
import { useDeployment } from "./deployment.ts";

export type Session = Awaited<ReturnType<typeof createAgentSession>>["session"];
export type Faux = ReturnType<typeof registerFauxProvider>;

export type SessionOptions = {
	/** Extension flags, set as pi sets the ones given on the command line. */
	flags?: Record<string, string | boolean>;
	/** process.argv while the extensions load and the session runs, after `node pi` (for modules that read --model). */
	argv?: string[];
	/** The deployment in effect (./deployment.ts); none by default. */
	deployment?: Deployment;
	/**
	 * The session's model from its start, `provider/id` of a provider an extension registers (a provider that needs to
	 * be the session's model when the session starts); else the scripted model, which `run` may replace.
	 */
	model?: string;
	/** Settings as a settings file holds them (compaction's thresholds, say); pi's own retry stays off. */
	settings?: Parameters<typeof SettingsManager.inMemory>[0];
	/**
	 * The session's file, its header and entries, as pi reads a file it opens (--continue, --session, /resume) or a
	 * fork's; else the session is a new one.
	 */
	entries?: FileEntry[];
};

/**
 * A real pi session in `base` with `extensions` and a scripted model, handed to `run`; the session's stderr lines are
 * kept off the test's output and handed over as `stderr`, the context pi gives an extension at session start as
 * `ctx()`, and a handler pi reports as failed fails the test.
 */
export async function withSession<T>(
	base: string,
	extensions: ((pi: ExtensionAPI) => void)[],
	run: (o: {
		session: Session;
		faux: Faux;
		runtime: ModelRuntime;
		stderr: string[];
		ctx: () => ExtensionContext;
	}) => Promise<T>,
	o: SessionOptions = {},
): Promise<T> {
	useDeployment(o.deployment ?? {});
	const faux = registerFauxProvider();
	const log = console.error;
	const argv = process.argv;
	const code = process.exitCode;
	const stderr: string[] = [];
	let session: Session | undefined;
	let ctx: ExtensionContext | undefined;
	try {
		if (o.argv) process.argv = ["node", "pi", ...o.argv];
		const scripted = faux.getModel();
		const runtime = await ModelRuntime.create({ authPath: join(base, "auth.json"), modelsPath: null });
		runtime.registerProvider(scripted.provider, {
			baseUrl: scripted.baseUrl,
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
		const settingsManager = SettingsManager.inMemory({
			...o.settings,
			retry: { enabled: false, maxRetries: 0, baseDelayMs: 1 },
		});
		const resourceLoader = new DefaultResourceLoader({
			cwd: base,
			agentDir: join(base, "agent"),
			settingsManager,
			extensionFactories: [
				...extensions,
				(pi) => {
					pi.on("session_start", (_event, c) => {
						ctx = c;
					});
				},
			],
		});
		await resourceLoader.reload();
		let model = scripted;
		if (o.model) {
			// The providers the extensions registered while loading, as pi registers them before it creates the session:
			// by name and config, or as a provider object (createProvider: this package's planners).
			const loaded = resourceLoader.getExtensions();
			for (const { name, config } of loaded.runtime.pendingProviderRegistrations)
				runtime.registerProvider(name, config);
			loaded.runtime.pendingProviderRegistrations = [];
			for (const { provider } of loaded.runtime.pendingNativeProviderRegistrations)
				runtime.registerNativeProvider(provider);
			loaded.runtime.pendingNativeProviderRegistrations = [];
			const i = o.model.indexOf("/");
			const found = runtime.getModel(o.model.slice(0, i), o.model.slice(i + 1));
			assert.ok(found, `${o.model} is registered`);
			model = found as typeof scripted;
		}
		({ session } = await createAgentSession({
			cwd: base,
			agentDir: join(base, "agent"),
			model,
			modelRuntime: runtime,
			sessionManager: SessionManager.inMemory(base, undefined, o.entries && structuredClone(o.entries)),
			settingsManager,
			resourceLoader,
		}));
		console.error = (...line: unknown[]) => void stderr.push(line.join(" "));
		for (const [name, value] of Object.entries(o.flags ?? {})) session.extensionRunner.setFlagValue(name, value);
		// pi reports a handler that throws to its error listeners only: without one, a throw would pass unseen.
		const thrown: string[] = [];
		await session.bindExtensions({ onError: (e) => void thrown.push(`${e.event}: ${e.error}`) });
		const out = await run({ session, faux, runtime, stderr, ctx: () => ctx as ExtensionContext });
		assert.deepEqual(thrown, [], "no extension handler threw");
		return out;
	} finally {
		console.error = log;
		process.argv = argv;
		process.exitCode = code;
		session?.dispose();
		faux.unregister();
	}
}

/** A 1x1 PNG, for a tool that returns a camera view. */
export const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

/** What a question to a VLM came back with: the reply's text, or the error askVlm raised for an error reply. */
export const said = async (ask: () => Promise<{ text: string }>) => {
	try {
		return (await ask()).text;
	} catch (err) {
		return `error: ${err instanceof Error ? err.message : String(err)}`;
	}
};
/** The dashboard's check of the session's model (POST /llm-check runs it): whether it passed, and what it reports. */
export const llm = async (ctx: ExtensionContext) => {
	const r = await llmCheck(ctx.modelRegistry, ctx.model as never);
	return `${r.ok ? "ok" : "failed"}: ${r.detail}`;
};
/** A question to the session's model, as a check asks one when no other model is named for it. */
export const vlm = (ctx: ExtensionContext) => said(() => askVlm(ctx, "", "off", "Is the cube in the bowl?", []));
/** What a request through pi's model registry came back with. */
export const reply = async (ctx: ExtensionContext, context: object, options: object = {}) => {
	const m = await ctx.modelRegistry.streamSimple(ctx.model as never, context as never, options).result();
	return m.stopReason === "error"
		? `error: ${m.errorMessage}`
		: m.content.map((c) => (c.type === "text" ? c.text : `[${c.type}]`)).join("");
};
/** A user message to ask with. */
export const ask = { role: "user", content: "Summarize the conversation.", timestamp: 0 };
/** An extension that runs `fn` inside pi's loop at the `n`-th `event` of the session. */
export const at =
	(
		event:
			| "turn_start"
			| "turn_end"
			| "context"
			| "tool_call"
			| "tool_execution_start"
			| "tool_execution_end"
			| "agent_end",
		n: number,
		fn: (ctx: ExtensionContext) => Promise<unknown>,
	) =>
	(pi: ExtensionAPI) => {
		let seen = 0;
		pi.on(event as "turn_end", async (_event, ctx) => {
			if (++seen === n) await fn(ctx);
		});
	};
/** `promise`, or a rejection once `ms` have passed without it (a session that hangs fails the test). */
export const within = <T>(ms: number, promise: Promise<T>, what: string): Promise<T> =>
	new Promise<T>((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error(`${what}: still running after ${ms} ms`)), ms);
		promise.then(
			(v) => {
				clearTimeout(timer);
				resolve(v);
			},
			(err) => {
				clearTimeout(timer);
				reject(err);
			},
		);
	});
