/**
 * Self-hosted planners /embodied-setup offers next to the models of the registry: an engine serving one
 * model on a GPU box the lab controls, reached as `selfhost/<id>` over its OpenAI-compatible API (the
 * `selfhost` provider's defaults in ../params.ts apply). Where it runs is the deployment's
 * `selfhost.<id>` (../config.ts): the URL this box reaches it at, the commands that start and stop it,
 * and the machine those commands run on when it is not this one. The agent the wizard hands over to
 * starts the server, waits for the URL, registers the model in ~/.pi/agent/models.json and checks one
 * request; without a deployment entry it sets the engine up from the quickstart skill's recipe.
 */

/** A self-hosted planner: the engine, the model it serves and what it needs. */
export type SelfhostSpec = {
	/** The model id the server answers to: `selfhost/<id>`. */
	id: string;
	engine: "vLLM" | "NInfer" | "Strata";
	/** The model and its weights, as the menu shows them. */
	model: string;
	/** The GPU class the engine builds for. */
	hardware: string;
	/** The per-request context the recipe's server is started with. */
	contextWindow: number;
};
export const SELFHOST: SelfhostSpec[] = [
	{
		id: "qwen3.8-27b-bf16",
		engine: "vLLM",
		model: "Qwen3.8-27B BF16 (unquantized)",
		hardware: "a GPU with 80 GB or more (an RTX PRO 6000 96 GB, an A800 / A100 80 GB); the weights are 55.6 GB",
		contextWindow: 65536,
	},
	{
		id: "qwen3.8-27b",
		engine: "NInfer",
		model: "Qwen3.8-27B NVFP4",
		hardware: "an RTX 5090 or a DGX Spark (NInfer builds for sm_120a / sm_121a only)",
		contextWindow: 32768,
	},
	{
		// The id Strata serves a --model-path Q8_0 pack under.
		id: "qwen3.8-flash-next-unsloth-q8_0",
		engine: "Strata",
		model: "Qwen3.8-Flash-Next Q8_0",
		hardware:
			"an NVIDIA card and about 140 GB of RAM the process may use for the experts (under a smaller memory limit, such as a 90 GB container, Strata reads experts from disk with --mmap-experts at about 1 token/s)",
		contextWindow: 65536,
	},
];
/** The planner menu's line for a self-hosted planner. */
export const selfhostLabel = (s: SelfhostSpec) => `selfhost/${s.id}: ${s.engine} · ${s.model} (started by the agent)`;
export const selfhostOf = (model: string | undefined) =>
	model?.startsWith("selfhost/") ? SELFHOST.find((s) => `selfhost/${s.id}` === model) : undefined;

/** A deployment's `selfhost.<id>`: how this box reaches the server and how it is started. */
export type SelfhostDeploy = {
	/** The OpenAI-compatible base URL as this box reaches it, e.g. http://127.0.0.1:18000/v1. */
	base_url: string;
	/** The shell command that starts the server (and any tunnel), run on `run_on`. */
	start?: string;
	stop?: string;
	/** Where `start` / `stop` run when it is not this box (it cannot reach the GPU box), e.g. "the Mac". */
	run_on?: string;
};

/** The models.json entry the agent writes for a self-hosted planner (provider `selfhost`). */
export function modelsJsonEntry(s: SelfhostSpec, baseUrl: string) {
	return {
		id: s.id,
		name: `${s.model} (${s.engine})`,
		baseUrl,
		input: ["text", "image"],
		reasoning: true,
		contextWindow: s.contextWindow,
		maxTokens: 8192,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	};
}
/** The provider fields the entry sits under: an OpenAI-compatible server with Qwen's thinking switch. */
export const SELFHOST_PROVIDER = {
	api: "openai-completions",
	apiKey: "none",
	compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens", thinkingFormat: "qwen-chat-template" },
};

/** The handoff lines that bring a self-hosted planner up before the install ends. */
export function selfhostHandoff(s: SelfhostSpec, d: SelfhostDeploy | undefined, deploymentName: string): string[] {
	const entry = JSON.stringify(modelsJsonEntry(s, d?.base_url ?? "<base URL>"));
	const register = `register it in ~/.pi/agent/models.json under providers.selfhost (create the provider with ${JSON.stringify(SELFHOST_PROVIDER)} if missing; replace a model with the same id): ${entry}`;
	const check = `check it before ending: GET ${d?.base_url ?? "<base URL>"}/models lists \`${s.id}\`, and one chat request with a camera-sized image and one tool answers (an image the model describes, a tool call it makes)`;
	if (!d)
		return [
			`- planner server: ${s.engine} · ${s.model}; deployment ${deploymentName || "default"} has no selfhost.${s.id} in ~/.pi/agent/embodied.json. Set it up on ${s.hardware} following the skill's "Self-hosted planners" section; ask me which box first. Then add selfhost.${s.id} (base_url as this box reaches it, start, stop, run_on) to the deployment, ${register}, and ${check}`,
		];
	const where = d.run_on
		? ` on ${d.run_on} (this box cannot run it: show me the command, I run it there, then you continue)`
		: "";
	return [
		`- planner server: ${s.engine} · ${s.model} at ${d.base_url}; if ${d.base_url}/models does not answer, start it${where}: \`${d.start ?? "(no start command in the deployment: ask me)"}\`${d.stop ? `; it stops with \`${d.stop}\` (one engine per card: stop the other self-hosted planner first when they share one)` : ""}`,
		`- planner registration: ${register}`,
		`- planner check: ${check}`,
	];
}

/** Whether a base URL answers /models within a few seconds (the run's preflight of a self-hosted planner). */
export async function selfhostUp(baseUrl: string, timeoutMs = 4000): Promise<boolean> {
	try {
		const r = await fetch(`${baseUrl.replace(/\/$/, "")}/models`, { signal: AbortSignal.timeout(timeoutMs) });
		return r.ok;
	} catch {
		return false;
	}
}
