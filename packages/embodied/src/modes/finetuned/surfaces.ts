/**
 * The action surfaces a fine-tuned student can speak besides action units (`--ft-surface`, ./index.ts):
 *
 * - `tools` (A2, tool orchestration): the student answers one JSON tool call `{"tool", "arguments"}`
 *   drawn from the robot's active tools, whose names, descriptions and parameter schemas the prompt
 *   lists (./templates/tools_v1.txt). The call is checked against the tool's schema exactly as the
 *   agent's would be (pi-ai's validateToolArguments); anything else falls back to the observe tool
 *   (`--ft-observe`, a read), never to a mutating call.
 * - `code` (A3, code synthesis): the student answers one fenced Python program
 *   (./templates/code_v1.txt), run through `run_code` under the same `code.api` whitelist and tier as
 *   ../code; the prompt lists the tier's primitives with ../code's own renderer, so a program sees
 *   the same declaration a planner model does. A reply that is not exactly one fenced block falls
 *   back to NOOP_PROGRAM (no primitive call: it only returns the next observation); the single word
 *   DONE ends the episode.
 *
 * Both share the units surface's request contract: images first, then the prompt, one stateless
 * request per step, temperature 0, thinking off. The training side (./sessions_to_sft.ts) renders
 * the same templates with the same functions, so a training prompt is byte for byte the inference one.
 */

import { readFileSync } from "node:fs";
import { type ToolCall, validateToolArguments } from "@earendil-works/pi-ai";
import { type TSchema, Type } from "typebox";
import { StringEnum } from "../../string-enum.ts";

export const SURFACES = ["units", "tools", "code"] as const;
export type Surface = (typeof SURFACES)[number];
export const isSurface = (s: string): s is Surface => (SURFACES as readonly string[]).includes(s);

/** The prompt of each non-units surface (ours; no Show-Harness template exists for them). */
export const SURFACE_PROMPTS: Record<Exclude<Surface, "units">, string> = {
	tools: readFileSync(new URL("./templates/tools_v1.txt", import.meta.url), "utf8").trim(),
	code: readFileSync(new URL("./templates/code_v1.txt", import.meta.url), "utf8").trim(),
};
/** Reply budget per surface: a bare unit token, a JSON call, a short program. */
export const SURFACE_MAX_TOKENS: Record<Surface, number> = { units: 24, tools: 256, code: 1024 };
/** The code surface's fallback: a program without a primitive call, so the step only observes. */
export const NOOP_PROGRAM = '# no program parsed from the reply: observe only\nRESULT = "observe"\n';
/** How many recent calls the tools prompt lists, newest first (the units prompt's window). */
export const RECENT_CALLS_MAX = 5;
/** The longest "last result" / "last report" excerpt a prompt carries. */
export const EXCERPT_MAX = 600;

/** A tool as the prompt shows it and the parser checks it (../../robot.ts PrimitivesHandle.tools). */
export type ToolSpec = { name: string; description: string; parameters: TSchema };
export type SurfaceCall = { name: string; arguments: Record<string, unknown> };
/** The robot's finish as the tools prompt offers it: the claim every robot's finish takes (one schema at training and inference). */
export const FINISH_TOOL: ToolSpec = {
	name: "finish",
	description: "End the episode with your claim about the task.",
	parameters: Type.Object({
		status: StringEnum(["success", "failure"] as const, { description: "Whether the task is done" }),
		summary: Type.String({ description: "What happened, in one or two sentences" }),
	}),
};

type Schema = {
	type?: string;
	enum?: unknown[];
	items?: Schema;
	minItems?: number;
	maxItems?: number;
	minimum?: number;
	maximum?: number;
	description?: string;
	properties?: Record<string, Schema>;
	required?: string[];
};

/** One parameter's type as the prompt names it: enum values, a fixed-length array, a scalar. */
function typeName(s: Schema): string {
	if (Array.isArray(s.enum)) return s.enum.map((v) => JSON.stringify(v)).join(" | ");
	if (s.type === "array") {
		const item = s.items ? typeName(s.items) : "any";
		if (s.minItems !== undefined && s.minItems === s.maxItems) return `[${Array(s.minItems).fill(item).join(", ")}]`;
		return `${item}[]`;
	}
	const range =
		s.minimum !== undefined || s.maximum !== undefined
			? ` (${s.minimum !== undefined ? `>= ${s.minimum}` : ""}${s.minimum !== undefined && s.maximum !== undefined ? ", " : ""}${s.maximum !== undefined ? `<= ${s.maximum}` : ""})`
			: "";
	return `${s.type ?? "any"}${range}`;
}

/**
 * The tools as the prompt lists them: one `- name: description` line, then one indented line per
 * parameter (`name: type, required: description`). Deterministic in the schema, so the training
 * renderer (./sessions_to_sft.ts) and the provider agree.
 */
export function renderTools(tools: readonly ToolSpec[]): string {
	return tools
		.map((t) => {
			const s = t.parameters as unknown as Schema;
			const required = new Set(s.required ?? []);
			const params = Object.entries(s.properties ?? {}).map(
				([k, p]) =>
					`    ${k}: ${typeName(p)}${required.has(k) ? ", required" : ""}${p.description ? `: ${p.description}` : ""}`,
			);
			return [`- ${t.name}${t.description ? `: ${t.description}` : ""}`, ...params].join("\n");
		})
		.join("\n");
}

/** A call as the recent-calls history shows it: `name({"k": v})`, newest first; "none" when empty. */
export const recentCallsText = (calls: readonly SurfaceCall[]) =>
	calls.length ? calls.map((c) => `${c.name}(${JSON.stringify(c.arguments)})`).join(", ") : "none";

/** A result text for the prompt: whitespace collapsed, cut at EXCERPT_MAX; "none" when empty. */
export function excerpt(text: string): string {
	const s = text.replace(/\s+/g, " ").trim();
	if (!s) return "none";
	return s.length > EXCERPT_MAX ? `${s.slice(0, EXCERPT_MAX)}...` : s;
}

const stripReasoning = (s: string) =>
	s
		.replace(/<think>[\s\S]*?<\/think>/gi, "")
		.replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, "")
		.trim();
const FENCE = /```[a-zA-Z0-9_-]*[ \t]*\r?\n([\s\S]*?)```/g;

/** The JSON object a reply carries: the whole reply, else a fenced block, else its outermost braces. */
function objectIn(s: string): unknown {
	const candidates = [s];
	for (const m of s.matchAll(FENCE)) candidates.push(m[1].trim());
	const open = s.indexOf("{");
	const close = s.lastIndexOf("}");
	if (open >= 0 && close > open) candidates.push(s.slice(open, close + 1));
	for (const c of candidates) {
		try {
			const v = JSON.parse(c);
			if (v && typeof v === "object" && !Array.isArray(v)) return v;
		} catch {}
	}
	throw new Error("no JSON object in the reply");
}

/**
 * The tool call a reply makes, checked: a JSON object naming one of `tools` (`tool` or `name`) with
 * `arguments` (default `{}`) valid against that tool's schema; the arguments come back as the
 * validator parses them. Throws on anything else (the caller falls back to the observe tool).
 */
export function parseToolCall(raw: string, tools: readonly ToolSpec[]): SurfaceCall {
	const v = objectIn(stripReasoning(raw)) as Record<string, unknown>;
	const name = v.tool ?? v.name;
	if (typeof name !== "string") throw new Error("the reply names no tool");
	const tool = tools.find((t) => t.name === name);
	if (!tool) throw new Error(`${name} is not one of the offered tools (${tools.map((t) => t.name).join(", ")})`);
	const args = v.arguments ?? v.args ?? v.parameters ?? {};
	if (!args || typeof args !== "object" || Array.isArray(args))
		throw new Error(`${name}: arguments must be an object`);
	// pi-ai's own check: what the agent loop would refuse, the student's call is refused too.
	const parsed = validateToolArguments(tool, {
		type: "toolCall",
		id: "ft",
		name,
		// Wire input: what the student answered, checked against the schema right here.
		arguments: args as ToolCall["arguments"],
	}) as Record<string, unknown>;
	return { name, arguments: parsed };
}

/** A code reply: exactly one fenced block (the program), or the single word DONE. Throws otherwise. */
export function parseProgram(raw: string): { kind: "program"; code: string } | { kind: "done" } {
	const s = stripReasoning(raw);
	if (/^DONE[.!]?$/i.test(s)) return { kind: "done" };
	const blocks = [...s.matchAll(FENCE)].map((m) => m[1].replace(/\s+$/, ""));
	if (blocks.length !== 1) throw new Error(`expected exactly one fenced code block, found ${blocks.length}`);
	if (!blocks[0].trim()) throw new Error("the fenced code block is empty");
	// Nothing but whitespace may surround the block: prose with a program in it is not a program.
	const rest = s.replace(FENCE, "").trim();
	if (rest) throw new Error("text outside the fenced code block");
	return { kind: "program", code: `${blocks[0]}\n` };
}

/** A program as the training output and the student's reply show it. */
export const fenced = (code: string) => `\`\`\`python\n${code.replace(/\s+$/, "")}\n\`\`\``;
