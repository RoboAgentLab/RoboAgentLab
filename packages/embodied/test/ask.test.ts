import assert from "node:assert/strict";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ASK_TOOL, type AskInput, answerText, askErrors, DONE, registerAsk, TYPE_OWN } from "../src/infra/setup/ask.ts";

type Tool = { execute: (...args: unknown[]) => Promise<{ content: { text: string }[]; details: unknown }> };

/** ask_user registered on a faux pi; the UI answers from `select` and `input` in order. */
function fake(o: {
	episode?: boolean;
	select?: (string | undefined)[];
	input?: (string | undefined)[];
	hasUI?: boolean;
}) {
	const tools = new Map<string, Tool>();
	const handlers: Record<string, (() => void)[]> = {};
	let active = ["read", ASK_TOOL];
	const offered: Record<string, string[][]> = {};
	const pi = {
		registerTool: (t: Tool & { name: string }) => tools.set(t.name, t),
		on: (name: string, fn: () => void) => {
			handlers[name] = [...(handlers[name] ?? []), fn];
		},
		getActiveTools: () => active,
		setActiveTools: (t: string[]) => {
			active = t;
		},
	} as unknown as ExtensionAPI;
	registerAsk(pi, () => o.episode === true);
	const select = [...(o.select ?? [])];
	const input = [...(o.input ?? [])];
	const ctx = {
		hasUI: o.hasUI ?? true,
		ui: {
			select: async (title: string, options: string[]) => {
				offered[title] = [...(offered[title] ?? []), options];
				const want = select.shift();
				return want === undefined ? undefined : options.find((x) => x.startsWith(want));
			},
			input: async () => input.shift(),
		},
	};
	const ask = async (params: AskInput) => {
		const r = await tools.get(ASK_TOOL)!.execute("id", params, undefined, undefined, ctx);
		return { text: r.content[0].text, details: r.details };
	};
	return {
		ask,
		offered,
		start: () => {
			for (const h of handlers.session_start ?? []) h();
		},
		active: () => active,
	};
}

const PLANNER: AskInput["questions"][number] = {
	header: "Planner",
	question: "Which model plans each step?",
	options: [
		{ label: "openai/gpt-6 (Recommended)", description: "this session's model, sees images" },
		{ label: "selfhost/qwen3.8-27b-bf16", description: "vLLM on the PRO 6000" },
	],
};

test("ask_user: invalid calls come back as errors the model can fix", () => {
	assert.deepEqual(askErrors({ questions: [] }), ["ask 1-4 questions, not 0"]);
	const bad = askErrors({
		questions: [
			{ header: "A very long header", question: "Which?", options: [{ label: "x" }] },
			{
				header: "GPU",
				question: "Which GPU?",
				options: [{ label: "GPU 0" }, { label: "GPU 5 (Recommended)" }, { label: "Other" }],
			},
		],
	});
	assert.deepEqual(bad, [
		"A very long header: the header must be 1-12 characters",
		"A very long header: give 2-4 options, not 1",
		`GPU: drop the Other option; "${TYPE_OWN}" is always added`,
		"GPU: put the (Recommended) option first",
	]);
	assert.deepEqual(askErrors({ questions: [PLANNER] }), []);
});

test("ask_user: each question is a menu with its descriptions and a typed answer; the answers go back by header", async () => {
	const f = fake({ select: ["openai/gpt-6", TYPE_OWN], input: ["2"] });
	const r = await f.ask({
		questions: [
			PLANNER,
			{ header: "GPU", question: "Which GPU?", options: [{ label: "GPU 5 (Recommended)" }, { label: "GPU 0" }] },
		],
	});
	assert.deepEqual(f.offered["Planner: Which model plans each step?"]?.[0], [
		"openai/gpt-6 (Recommended) — this session's model, sees images",
		"selfhost/qwen3.8-27b-bf16 — vLLM on the PRO 6000",
		TYPE_OWN,
	]);
	assert.equal(r.text, "Planner: openai/gpt-6 (Recommended)\nGPU: typed: 2");
	// A wrong call is not shown to the user.
	const g = fake({});
	const e = await g.ask({ questions: [{ header: "X", question: "?", options: [{ label: "a" }] }] });
	assert.match(e.text, /^Not asked; fix the call: X: give 2-4 options, not 1$/);
	assert.deepEqual(g.offered, {});
});

test("ask_user: multiSelect picks until Done; cancelling stops the asking", async () => {
	const panels = {
		header: "Panels",
		question: "Which panels open?",
		multiSelect: true,
		options: [{ label: "Dashboard (Recommended)" }, { label: "Viser 3D view" }, { label: "Camera grid" }],
	};
	let f = fake({ select: ["Dashboard", "Camera", DONE] });
	let r = await f.ask({ questions: [panels] });
	assert.equal(r.text, "Panels: Dashboard (Recommended); Camera grid");
	assert.deepEqual(f.offered["Panels: Which panels open?"]?.[2], [
		`${DONE} (Dashboard (Recommended), Camera grid)`,
		"Viser 3D view",
		TYPE_OWN,
	]);
	f = fake({ select: [undefined] });
	r = await f.ask({ questions: [PLANNER, panels] });
	assert.match(r.text, /^The user cancelled/);
	assert.equal(answerText([{ header: "X", selected: [] }]), "X: (no answer)");
});

test("ask_user: not in an episode, not without a UI", async () => {
	const e = fake({ episode: true });
	e.start();
	assert.deepEqual(e.active(), ["read"], "an episode's planner does not get the tool");
	assert.match((await e.ask({ questions: [PLANNER] })).text, /not available in an episode/);
	const s = fake({});
	s.start();
	assert.deepEqual(s.active(), ["read", ASK_TOOL]);
	assert.match((await fake({ hasUI: false }).ask({ questions: [PLANNER] })).text, /^No UI to ask in/);
});
