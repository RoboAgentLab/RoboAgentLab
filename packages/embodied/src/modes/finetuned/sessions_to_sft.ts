/**
 * Teacher sessions -> SFT pairs for a student speaking the tools or code surface (./surfaces.ts).
 *
 *   node --experimental-strip-types packages/embodied/src/modes/finetuned/sessions_to_sft.ts \
 *     --surface tools|code --out <dir> [--robot R] [--vars arms=left,right,...] [--tools a,b] \
 *     [--code-api high|low|...] [--task T] [--include-failures] [--dry-run] <session.jsonl | dir>...
 *
 * Reads pi session files (`.jsonl`, or directories searched for them) of runs a planner model made
 * on a robot, keeps the successful ones (the `robot_result` entry claims success and reports no
 * env or planner error; `--include-failures` keeps every run) and writes one sample per teacher
 * step in LLaMA-Factory's Alpaca layout, as Show-Harness's rollouts_to_alpaca.py writes the units
 * samples: `<out>/sft.json` with `{instruction: "<image>" * k + prompt, input: "", output, images}`
 * and `<out>/images/<session>/<step>_<i>.png` (the observation's images as recorded).
 *
 * - tools: a step is an assistant turn that calls exactly one of the robot's tools (or finish)
 *   right after a tool result with images; the prompt is the tools template over the robot's
 *   manifest tools (`--tools` restricts them, `--vars` fills the manifest's `{{name}}` variables
 *   as the robot would, e.g. `arms=`), the recent calls and the last result, exactly as the
 *   provider renders it; the output is the call as JSON.
 * - code: a step is a `run_code` call whose result ran (status ran, no error), after a result with
 *   images; the prompt is the code template over the tier's primitives (`--code-api`, else the
 *   session's recorded `code_api`; the `code_api` entry's available set bounds them), rendered as
 *   code mode renders them; the output is the program in a fenced block.
 *
 * `--dry-run` prints the counts per session and in all, and writes nothing.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { parseSessionEntries, type SessionEntry } from "@earendil-works/pi-coding-agent";
import {
	loadManifest,
	type Manifest,
	tools as manifestTools,
	toolDescription,
	toolSchema,
	type Vars,
} from "../../primitives/manifest.ts";
import { CODE_API_ENTRY, type CodeApiTier, codePrimitives } from "../../primitives/registry.ts";
import { RESULT_ENTRY, TASK_ENTRY } from "../../robot.ts";
import { codeTier, renderPrimitives } from "../code/index.ts";
import { formatPrompt } from "./index.ts";
import {
	excerpt,
	FINISH_TOOL,
	fenced,
	RECENT_CALLS_MAX,
	recentCallsText,
	renderTools,
	SURFACE_PROMPTS,
	type SurfaceCall,
	type ToolSpec,
} from "./surfaces.ts";

type Json = Record<string, unknown>;
export type Sample = { instruction: string; input: string; output: string; images: string[] };
type Part = { type: string; text?: string; data?: string };
type Message = {
	role: string;
	content: Part[] | string;
	toolCallId?: string;
	toolName?: string;
	details?: Json;
	isError?: boolean;
};

/** Every `.jsonl` under `root` (itself when it is one), sorted. */
export function findSessions(root: string): string[] {
	if (statSync(root).isFile()) return [root];
	return readdirSync(root)
		.map((d) => join(root, d))
		.sort()
		.flatMap((p) => (statSync(p).isDirectory() ? findSessions(p) : p.endsWith(".jsonl") ? [p] : []));
}

/** The session's custom entries of `type`, oldest first. */
const custom = (entries: SessionEntry[], type: string): Json[] =>
	entries.flatMap((e) => (e.type === "custom" && e.customType === type ? [(e.data ?? {}) as Json] : []));

/** Whether the run succeeded: its result claims success with no env or planner error. */
export function succeeded(entries: SessionEntry[]): boolean {
	const r = custom(entries, RESULT_ENTRY).pop();
	return !!r && r.claimed === "success" && r.env_error !== true && !r.planner_error;
}

/** The task text as the provider words it: the robot_task entry's fields, `key value, ...`. */
export function taskText(entries: SessionEntry[]): { robot: string; task: string } {
	const t = custom(entries, TASK_ENTRY).pop() ?? {};
	const robot = String(t.robot ?? "");
	const task = Object.entries(t)
		.filter(([k]) => k !== "robot")
		.map(([k, v]) => `${k} ${v}`)
		.join(", ");
	return { robot, task };
}

/** `arms=left,right,max_move=0.2` as manifest variables (a value with commas is a list; a number a number). */
export function parseVars(spec: string): Vars {
	const vars: Vars = {};
	for (const pair of spec.split(/[;]/).filter(Boolean)) {
		const i = pair.indexOf("=");
		if (i < 0) throw new Error(`bad --vars ${pair}: use name=value or name=a,b,c (pairs separated by ;)`);
		const [k, v] = [pair.slice(0, i).trim(), pair.slice(i + 1).trim()];
		vars[k] = v.includes(",")
			? v
					.split(",")
					.map((s) => s.trim())
					.filter(Boolean)
			: v === ""
				? []
				: Number.isFinite(Number(v)) && v !== ""
					? Number(v)
					: v;
	}
	return vars;
}

/**
 * The manifest's tools as the provider offers them (schema and description from the manifest), plus
 * finish. A list variable an enum names (`"values": "{{cameras}}"`) must come from `--vars`: the robot
 * fills it at run time from its flags, which a session file does not hold; every missing one is named.
 */
export function offeredTools(m: Manifest, vars: Vars, only?: readonly string[]): ToolSpec[] {
	const entries = manifestTools(m).filter((e) => !e.module && (!only || only.includes(e.name)));
	const missing = new Set<string>();
	for (const e of entries)
		for (const p of Object.values(e.params ?? {})) {
			const v = typeof p.values === "string" ? /^\{\{(\w+)\}\}$/.exec(p.values)?.[1] : undefined;
			if (v && !Array.isArray(vars[v])) missing.add(v);
		}
	if (missing.size)
		throw new Error(
			`the ${m.robot} manifest's tools take the list variables ${[...missing].join(", ")}: pass --vars "${[...missing].map((v) => `${v}=a,b`).join(";")}" (an empty value drops the parameter, as the robot does)`,
		);
	const list = entries.map((e) => ({
		name: e.name,
		description: toolDescription(e, vars),
		parameters: toolSchema(e, vars),
	}));
	return [...list, FINISH_TOOL];
}

/** The messages of a session in order (pi's message entries). */
const messages = (entries: SessionEntry[]): Message[] =>
	entries.flatMap((e) => (e.type === "message" ? [e.message as unknown as Message] : []));
const images = (m: Message) =>
	typeof m.content === "string" ? [] : m.content.flatMap((c) => (c.type === "image" && c.data ? [c.data] : []));
const texts = (m: Message) =>
	typeof m.content === "string" ? [m.content] : m.content.flatMap((c) => (c.type === "text" ? [c.text ?? ""] : []));
const calls = (m: Message) =>
	typeof m.content === "string"
		? []
		: (m.content as (Part & { name?: string; arguments?: Json })[]).filter((c) => c.type === "toolCall");

/**
 * The tools-surface samples of one session: for each assistant turn with exactly one call to an
 * offered tool after an observation with images, the prompt the provider would have rendered there
 * and the call as JSON. `image` stores a PNG and returns its path.
 */
export function toolSamples(
	entries: SessionEntry[],
	tools: ToolSpec[],
	task: string,
	image: (step: number, index: number, base64: string) => string,
): Sample[] {
	const names = new Set(tools.map((t) => t.name));
	const out: Sample[] = [];
	let recent: SurfaceCall[] = [];
	let last: Message | undefined;
	let step = 0;
	for (const m of messages(entries)) {
		if (m.role === "toolResult") {
			if (images(m).length) last = m;
			continue;
		}
		if (m.role !== "assistant") continue;
		const made = calls(m);
		if (made.length !== 1 || !last) continue;
		const c = { name: String(made[0].name), arguments: (made[0].arguments ?? {}) as Json };
		if (!names.has(c.name)) continue;
		const prompt = formatPrompt(SURFACE_PROMPTS.tools, {
			task,
			recent_calls: recentCallsText(recent),
			last_result: excerpt(texts(last)[0] ?? ""),
			tools: renderTools(tools),
		});
		const views = images(last).map((data, i) => image(step, i, data));
		out.push({
			instruction: `${"<image>".repeat(views.length)}${prompt}`,
			input: "",
			output: JSON.stringify({ tool: c.name, arguments: c.arguments }),
			images: views,
		});
		recent = [c, ...recent].slice(0, RECENT_CALLS_MAX);
		last = undefined;
		step++;
	}
	return out;
}

/**
 * The code-surface samples of one session: for each `run_code` call whose result ran, after an
 * observation with images, the code prompt over `api` and the program fenced.
 */
export function codeSamples(
	entries: SessionEntry[],
	api: { tier: string; api: string },
	task: string,
	image: (step: number, index: number, base64: string) => string,
): Sample[] {
	const out: Sample[] = [];
	const ms = messages(entries);
	const resultOf = (id: string) => ms.find((m) => m.role === "toolResult" && m.toolCallId === id);
	let last: Message | undefined;
	let step = 0;
	for (const m of ms) {
		if (m.role === "toolResult") {
			if (images(m).length) last = m;
			continue;
		}
		if (m.role !== "assistant") continue;
		const made = calls(m);
		if (made.length !== 1 || made[0].name !== "run_code" || !last) continue;
		const code = String(made[0].arguments?.code ?? "");
		const r = resultOf(String((made[0] as { id?: string }).id ?? ""));
		if (!code.trim() || !r || r.isError || r.details?.status !== "ran") continue;
		const prompt = formatPrompt(SURFACE_PROMPTS.code, {
			task,
			last_report: excerpt(texts(last)[0] ?? ""),
			tier: api.tier,
			api: api.api,
		});
		const views = images(last).map((data, i) => image(step, i, data));
		out.push({
			instruction: `${"<image>".repeat(views.length)}${prompt}`,
			input: "",
			output: fenced(code),
			images: views,
		});
		last = undefined;
		step++;
	}
	return out;
}

/** The code API the session's programs saw: `--code-api` (else the result's `code_api`) over the manifest, bounded by the `code_api` entry's available set. */
export function sessionCodeApi(entries: SessionEntry[], m: Manifest, vars: Vars, explicit: string) {
	const result = custom(entries, RESULT_ENTRY).pop() ?? {};
	const recorded = custom(entries, CODE_API_ENTRY).pop();
	const tier = explicit || String(result.code_api ?? "") || codeTier("", m, false);
	const available = Array.isArray(recorded?.available) ? new Set(recorded.available as string[]) : undefined;
	const prims = codePrimitives(m, tier as CodeApiTier, () => true, vars).filter(
		(p) => !available || available.has(p.name),
	);
	return { tier, api: renderPrimitives(prims) || "(none)" };
}

export function convert(o: {
	surface: "tools" | "code";
	sessions: string[];
	out: string;
	robot?: string;
	vars?: string;
	tools?: string;
	"code-api"?: string;
	task?: string;
	"include-failures"?: boolean;
	"dry-run"?: boolean;
}) {
	const vars = parseVars(o.vars ?? "");
	const only = o.tools
		?.split(",")
		.map((s) => s.trim())
		.filter(Boolean);
	const samples: Sample[] = [];
	const report: { session: string; samples: number; skipped?: string }[] = [];
	const dry = o["dry-run"] === true;
	for (const file of o.sessions.flatMap(findSessions)) {
		const entries = parseSessionEntries(readFileSync(file, "utf8")) as SessionEntry[];
		const who = taskText(entries);
		const robot = o.robot ?? who.robot;
		const name = basename(file, ".jsonl");
		if (!robot) {
			report.push({ session: file, samples: 0, skipped: "no robot_task entry (pass --robot)" });
			continue;
		}
		if (!o["include-failures"] && !succeeded(entries)) {
			report.push({ session: file, samples: 0, skipped: "not a successful run" });
			continue;
		}
		const m = loadManifest(robot);
		const task = o.task ?? who.task;
		const image = (step: number, i: number, data: string) => {
			const rel = join("images", name, `${String(step).padStart(4, "0")}_${i}.png`);
			if (!dry) {
				mkdirSync(join(o.out, "images", name), { recursive: true });
				writeFileSync(join(o.out, rel), Buffer.from(data, "base64"));
			}
			return rel;
		};
		const mine =
			o.surface === "tools"
				? toolSamples(entries, offeredTools(m, vars, only), task, image)
				: codeSamples(entries, sessionCodeApi(entries, m, vars, o["code-api"] ?? ""), task, image);
		report.push({ session: file, samples: mine.length });
		samples.push(...mine);
	}
	if (!dry) {
		mkdirSync(o.out, { recursive: true });
		writeFileSync(join(o.out, "sft.json"), `${JSON.stringify(samples, null, 1)}\n`);
	}
	return {
		out: dry ? null : resolve(o.out, "sft.json"),
		surface: o.surface,
		samples: samples.length,
		sessions: report,
	};
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) {
	const { values, positionals } = parseArgs({
		allowPositionals: true,
		options: {
			surface: { type: "string" },
			out: { type: "string" },
			robot: { type: "string" },
			vars: { type: "string" },
			tools: { type: "string" },
			"code-api": { type: "string" },
			task: { type: "string" },
			"include-failures": { type: "boolean" },
			"dry-run": { type: "boolean" },
		},
	});
	const surface = values.surface;
	if ((surface !== "tools" && surface !== "code") || !positionals.length || (!values.out && !values["dry-run"])) {
		console.error(
			"usage: sessions_to_sft.ts --surface tools|code --out <dir> [--robot R] [--vars k=v;k=a,b] [--tools a,b] [--code-api tier] [--task T] [--include-failures] [--dry-run] <session.jsonl | dir>...",
		);
		process.exit(2);
	}
	console.log(
		JSON.stringify(
			convert({ ...values, surface, out: values.out ?? "", sessions: positionals.map((p) => resolve(p)) }),
			null,
			2,
		),
	);
	if (!existsSync(values.out ?? "") && !values["dry-run"]) process.exitCode = 1;
}
