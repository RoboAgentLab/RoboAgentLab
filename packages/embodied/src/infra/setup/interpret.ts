/**
 * A free-text `/embodied-setup` or `/embodied-run` request read by the session's model instead of by
 * keywords: the model gets the request and the catalog of what the robots run (./catalog.ts, generated
 * from the code at call time) and answers with one JSON object; the code checks it against the same
 * catalog and ../tiers.ts (`choose`, `conflicts`, `unsupported`), sends a failed answer back with the
 * reasons (at most two more rounds), and turns an answer that passes into what the wizard already takes
 * (./request.ts `Request`, plus the episode or batch of a run). Then the usual menus ask for the rest,
 * and the confirmation shows it all before anything is written, installed or started.
 *
 * The keyword reading (./request.ts `parseRequest`) stays: it is used when no model answers (none
 * configured, offline, an answer that never passes) and in the tests that need a deterministic reading.
 */

import type { Api, Model } from "@earendil-works/pi-ai";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";
import { parseRoleJson } from "../../modes/motormind/models.ts";
import { AS, type Cell, cellKey, OS, PAPERS, type Paper, paperFlag, supported } from "./axes.ts";
import { robotCaps, settingsFor, tierProblems } from "./caps.ts";
import {
	type CatalogExperiment,
	type CatalogRobot,
	evalUsage,
	flashFor,
	knownFlags,
	runCatalog,
	setupCatalog,
	surfaceFor,
	THINKING,
} from "./catalog.ts";
import { FINETUNED_REF, FLASH_REF, type Kind, type Request } from "./request.ts";

/** One round with the model: the system prompt and one user message; the reply's text. */
export type Ask = (system: string, user: string, signal?: AbortSignal) => Promise<string>;

/** An `Ask` over pi's model registry with `model`, the way the agent's own requests go (provider, key, headers). */
export function registryAsk(registry: Pick<ModelRegistry, "streamSimple">, model: Model<Api>): Ask {
	return async (system, user, signal) => {
		const reply = await registry
			.streamSimple(
				model,
				{ systemPrompt: system, messages: [{ role: "user", content: user, timestamp: Date.now() }] },
				{ maxTokens: 4000, signal: signal ?? AbortSignal.timeout(120_000) },
			)
			.result();
		if (reply.stopReason === "error" || reply.stopReason === "aborted")
			throw new Error(reply.errorMessage ?? reply.stopReason);
		return reply.content.map((c) => (c.type === "text" ? c.text : "")).join("");
	};
}

/** A checked answer: the value, or why it fails. */
export type Checked<T> = { value: T } | { errors: string[] };
/** The rounds with the model: the first answer, then up to `rounds - 1` corrections. */
export type Reading<T> = { value?: T; errors: string[]; rounds: number };

/** Ask, check, and send a failed answer back with the reasons until one passes or the rounds are spent. */
export async function interpret<T>(
	ask: Ask,
	system: string,
	user: string,
	check: (json: Record<string, unknown>) => Checked<T>,
	rounds = 3,
): Promise<Reading<T>> {
	let message = user;
	let errors: string[] = [];
	for (let n = 1; n <= rounds; n++) {
		const reply = await ask(system, message);
		const json = parseRoleJson(reply);
		const checked: Checked<T> =
			json && typeof json === "object" && !Array.isArray(json)
				? check(json as Record<string, unknown>)
				: { errors: ["the reply holds no JSON object"] };
		if ("value" in checked) return { value: checked.value, errors: [], rounds: n };
		errors = checked.errors;
		message = `${user}\n\nYour previous answer:\n${reply.trim()}\n\nIt failed these checks:\n${errors.map((e) => `- ${e}`).join("\n")}\nAnswer again with one corrected JSON object.`;
	}
	return { errors, rounds };
}

const RULES = `Rules:
- Use only ids, cells, setting ids, flags, planners and values that appear in the catalog. Never invent a flag or an id.
- When the request asks for something the catalog does not offer (a robot, a work that is not ported, a planner, a flag), leave that field null and say in "unsupported" what is missing, one short sentence each.
- Leave a field null when the request does not settle it; the user is asked for it afterwards.
- Reply with one JSON object and nothing else.`;

/** The setup reader's instructions: the answer's fields and what each may hold. */
export const SETUP_SYSTEM = `You read a request to set up a pi-embodied robot experiment and answer with the setup it asks for, using only the catalog you are given.
A robot is one of catalog.robots (by id). Its cells (observation O1-O3 x execution A1C/A1M/A2/A3) are the ones it lists; a setting (a ported work's preset or CaP-X tier) runs in its own cell, on the robots it lists. "VLA only", "no agent" or a VLA run without a planner is vla_alone, never the A1M cell (A1M is an agent driving the robot's VLA); only robots whose vla_alone.mounted is true run it.
Answer:
{"robots": [robot ids: one when the request settles it, several when it names a family such as "franka", [] when it names none],
 "kind": "sim" | "real" | null,
 "vla_alone": true | false,
 "cell": "O1/A2" (a cell of the robot) | null,
 "setting": a setting id from catalog.settings | null,
 "turns": "single" | "multi" | null (single-turn = one program, CaP-X S tiers; multi-turn = the M tiers; only when no setting is named),
 "planner": a catalog.planners.registry ref | "finetuned" | "flash/replay" | null,
 "thinking": one of catalog.thinking_levels | null,
 "where": "here" | "remote" | null,
 "host": "user@host" or an ssh host for remote services | null,
 "unsupported": [strings],
 "summary": one sentence of what you understood}
${RULES}`;

/** The run reader's instructions. */
export const RUN_SYSTEM = `You read a request to run episodes of an installed pi-embodied experiment and answer with the run it asks for, using only the catalog you are given.
An experiment is one of catalog.experiments (by dir); its setting and planner are recorded. A run is one episode (mode "here": in this terminal, "headless": in the background by the agent, "print": the command is shown) or a batch over tasks and seeds with the robot's eval script (mode "batch"), whose positional arguments and environment variables are in the experiment's eval.usage and eval.env; the first argument is the output directory, "runs/<a short name>".
Answer:
{"experiment": an experiment dir | null,
 "mode": "here" | "headless" | "print" | "batch" | null,
 "task_flags": the episode's task flags in the format of the experiment's task_flags | null,
 "batch": {"args": [positional arguments of eval.sh, the first "runs/<name>"], "env": {VAR: value, only names in eval.env}, "flags": "extra pi flags or empty"} | null,
 "gpu": a catalog.gpus index | null,
 "planner": a catalog.planners ref, only to replace the recorded planner | null,
 "thinking": one of catalog.thinking_levels | null,
 "max_turns": integer | null,
 "time_limit": seconds | null,
 "unsupported": [strings],
 "summary": one sentence of what you understood}
${RULES}`;

const str = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
/** Shell-safe text: no quotes, separators, substitutions or redirections. */
const SAFE = /^[A-Za-z0-9 _./:,=+@%~^-]*$/;

/** The flags among `text` this robot does not take, and text that is not shell-safe. */
function flagProblems(robot: CatalogRobot, text: string, what: string): string[] {
	if (!SAFE.test(text)) return [`${what} "${text}" has characters a command line cannot carry safely`];
	const known = knownFlags(robot);
	return text
		.split(/\s+/)
		.filter((t) => t.startsWith("--"))
		.map((t) => t.slice(2).split("=")[0])
		.filter((n) => !known.has(n))
		.map((n) => `${what}: --${n} is not a flag ${robot.id} takes`);
}

/** What the setup reader's answer is checked against, besides the catalog. */
export type SetupContext<R extends CatalogRobot & { kind: Kind }> = {
	robots: R[];
	planners: string[];
};
/** The setup answer checked and turned into the wizard's `Request`. */
export function checkSetup<R extends CatalogRobot & { kind: Kind }>(
	json: Record<string, unknown>,
	x: SetupContext<R>,
): Checked<Request<R>> {
	const errors: string[] = [];
	const r: Request<R> = { settled: [], unknown: [] };
	const ids = strs(json.robots);
	const robots = ids.map((id) => x.robots.find((b) => b.id === id));
	ids.forEach((id, i) => {
		if (!robots[i]) errors.push(`robots: ${id} is not a robot id of the catalog`);
	});
	const found = robots.filter((b): b is R => b !== undefined);
	if (found.length) {
		r.robots = found;
		r.settled.push(
			found.length === 1 ? `robot ${found[0].label}` : `robot: one of ${found.map((b) => b.label).join(" / ")}`,
		);
	}
	const one = found.length === 1 ? found[0] : undefined;
	const kind = json.kind;
	if (kind === "sim" || kind === "real") r.kind = kind;
	else if (kind != null) errors.push(`kind: ${String(kind)} is not sim or real`);
	if (!r.kind && found.length && found.every((b) => b.kind === found[0].kind)) r.kind = found[0].kind;

	if (json.vla_alone === true) {
		r.vlaAlone = true;
		for (const b of found) {
			if (!robotCaps(b).vlaAlone)
				errors.push(`vla_alone: ${b.id} does not run a VLA alone (vla_alone.mounted is false)`);
			else if (!b.vlaAlone)
				errors.push(
					`vla_alone: ${b.id} runs a VLA alone, but /embodied-setup has no flags for it yet; say so in unsupported`,
				);
		}
		r.settled.push("the VLA alone, no agent (the VLA-alone baseline)");
	}

	const setting = str(json.setting);
	let paper: Paper | undefined;
	if (setting) {
		paper = PAPERS.find((p) => p.id === setting);
		if (!paper) errors.push(`setting: ${setting} is not a setting id of the catalog`);
		else if (r.vlaAlone) errors.push(`setting: ${setting} plans with an agent; the VLA alone has no setting`);
		else {
			for (const b of one ? [one] : []) {
				if (!settingsFor(b, paper.cell).includes(paper)) {
					const why = tierProblems(b, paperFlag(paper, b.id));
					errors.push(
						why.length
							? `setting ${paper.id} on ${b.id}: ${why.join("; ")}`
							: `setting: ${paper.id} is not offered on ${b.id} (its robots: ${paper.robots?.join(", ") ?? "any with the cell"})`,
					);
				}
			}
			r.paper = paper;
			r.settled.push(`${paper.name}'s setting (${paper.cell.o} × ${paper.cell.a}, ${paper.flag})`);
		}
	}
	const cellText = str(json.cell);
	let cell: Cell | undefined;
	if (cellText) {
		const [o, a] = cellText.split("/") as [Cell["o"], Cell["a"]];
		if (!OS.includes(o) || !AS.includes(a)) errors.push(`cell: ${cellText} is not one of O1-O3 / A1C, A1M, A2, A3`);
		else {
			cell = { o, a };
			if (r.vlaAlone) errors.push("cell: the VLA alone has no cell");
			if (paper && cellKey(paper.cell) !== cellKey(cell))
				errors.push(
					`cell: ${cellText} is not ${paper.id}'s cell (${cellKey(paper.cell)}); leave cell null with a setting`,
				);
			if (one && !supported(one.axes, cell)) errors.push(`cell: ${one.id} has no ${cellText}`);
			if (!paper) {
				r.o = o;
				r.a = a;
				r.settled.push(`${o} × ${a}`);
			}
		}
	}
	const turns = json.turns;
	if (turns === "single" || turns === "multi") {
		r.turns = turns;
		r.settled.push(`${turns}-turn`);
	} else if (turns != null) errors.push(`turns: ${String(turns)} is not single or multi`);

	const planner = str(json.planner);
	if (planner) {
		const where = cell ?? paper?.cell;
		if (planner === FINETUNED_REF) {
			if (one && where && !surfaceFor(one, where, paper))
				errors.push(`planner: the fine-tuned student does not plan ${paper?.id ?? cellKey(where)} on ${one.id}`);
		} else if (planner === FLASH_REF) {
			if (one && !flashFor(one, paper))
				errors.push(`planner: ${one.id} cannot replay a plan with flash/replay here`);
		} else if (!x.planners.includes(planner)) errors.push(`planner: ${planner} is not a planner of the catalog`);
		if (r.vlaAlone) errors.push("planner: the VLA alone has no planner");
		r.model = planner;
		r.settled.push(planner === FINETUNED_REF ? "planner: the fine-tuned student" : `planner ${planner}`);
	}
	const thinking = str(json.thinking);
	if (thinking) {
		if (!THINKING.includes(thinking)) errors.push(`thinking: ${thinking} is not one of ${THINKING.join(", ")}`);
		r.thinking = thinking;
		r.settled.push(`thinking ${thinking}`);
	}
	const where = json.where;
	const host = str(json.host);
	if (host && !/^[\w.@-]+$/.test(host)) errors.push(`host: ${host} is not an ssh host`);
	if (where === "remote" || host) {
		r.where = "remote";
		if (host) r.host = host;
		r.settled.push(host ? `services on ${host}` : "services on a remote box");
	} else if (where === "here") {
		r.where = "here";
		r.settled.push("services here");
	} else if (where != null) errors.push(`where: ${String(where)} is not here or remote`);
	r.unsupported = strs(json.unsupported);
	return errors.length ? { errors } : { value: r };
}

/** A run the reader settled: the experiment, the episode's task flags or a batch, and the overrides. */
export type RunPlan = {
	dir?: string;
	how?: "here" | "headless" | "print" | "batch";
	taskFlags?: string;
	batch?: { args: string[]; env: Record<string, string>; flags: string };
	gpu?: number;
	model?: string;
	thinking?: string;
	maxTurns?: number;
	timeLimit?: number;
	unsupported: string[];
	settled: string[];
};
export type RunContext = {
	experiments: CatalogExperiment[];
	planners: string[];
	gpus: number[];
};
const posInt = (v: unknown) => (typeof v === "number" && Number.isInteger(v) && v > 0 ? v : undefined);

/** The run answer checked: the experiment, flags the robot takes, the setting's preset or tier still valid, the batch script's arguments. */
export function checkRun(json: Record<string, unknown>, x: RunContext): Checked<RunPlan> {
	const errors: string[] = [];
	const plan: RunPlan = { unsupported: strs(json.unsupported), settled: [] };
	const dir = str(json.experiment);
	let exp: CatalogExperiment | undefined;
	if (dir) {
		exp = x.experiments.find((e) => e.dir === dir);
		if (!exp)
			errors.push(
				`experiment: ${dir} is not an experiment of the catalog (${x.experiments.map((e) => e.dir).join(", ")})`,
			);
		else {
			plan.dir = dir;
			plan.settled.push(`experiment ${dir}`);
		}
	} else if (x.experiments.length === 1) exp = x.experiments[0];
	const how = json.mode;
	if (how === "here" || how === "headless" || how === "print" || how === "batch") plan.how = how;
	else if (how != null) errors.push(`mode: ${String(how)} is not here, headless, print or batch`);
	const needExp = (what: string) => {
		if (!exp) errors.push(`${what} needs the experiment: name one`);
		return exp;
	};
	const task = str(json.task_flags);
	if (task) {
		const e = needExp("task_flags");
		if (e) errors.push(...flagProblems(e.robot, task, "task_flags"));
		plan.taskFlags = task;
		plan.settled.push(`task ${task}`);
	}
	const batch = json.batch;
	let batchFlags = "";
	if (batch != null) {
		const b = batch as { args?: unknown; env?: unknown; flags?: unknown };
		const e = needExp("batch");
		const ev = e ? evalUsage(e.robot) : undefined;
		if (e && !ev) errors.push(`batch: ${e.robot.id} has no eval.sh`);
		if (plan.how !== "batch") errors.push('batch: set mode "batch" with a batch');
		const args = strs(b.args);
		if (!args.length || !/^runs\/[A-Za-z0-9._-]+(\/[A-Za-z0-9._-]+)*$/.test(args[0]) || args[0].includes(".."))
			errors.push(
				`batch.args: the first argument must be the output directory runs/<name>, got ${args[0] ?? "nothing"}`,
			);
		for (const a of args.slice(1))
			if (!/^[A-Za-z0-9._,:+=-]+$/.test(a)) errors.push(`batch.args: "${a}" is not a plain argument`);
		const env: Record<string, string> = {};
		for (const [k, v] of Object.entries((b.env ?? {}) as Record<string, unknown>)) {
			if (ev && !ev.env.includes(k))
				errors.push(`batch.env: ${k} is not a variable ${ev.script} reads (${ev.env.join(", ") || "none"})`);
			const val = String(v);
			if (!/^[A-Za-z0-9._,:+=-]+$/.test(val)) errors.push(`batch.env: ${k}="${val}" is not a plain value`);
			env[k] = val;
		}
		batchFlags = str(b.flags) ?? "";
		if (e && batchFlags) errors.push(...flagProblems(e.robot, batchFlags, "batch.flags"));
		plan.batch = { args, env, flags: batchFlags };
		plan.settled.push(`batch ${args.join(" ")}`);
	} else if (plan.how === "batch") errors.push('mode "batch" needs a batch');
	if (exp && (task || batchFlags))
		errors.push(...tierProblems(exp.robot, [exp.record?.flags ?? "", task ?? "", batchFlags].join(" ")));
	const gpu = json.gpu;
	if (typeof gpu === "number") {
		if (x.gpus.length && !x.gpus.includes(gpu)) errors.push(`gpu: ${gpu} is not one of ${x.gpus.join(", ")}`);
		plan.gpu = gpu;
		plan.settled.push(`GPU ${gpu}`);
	} else if (gpu != null) errors.push(`gpu: ${String(gpu)} is not a GPU index`);
	const planner = str(json.planner);
	if (planner) {
		if (!x.planners.includes(planner)) errors.push(`planner: ${planner} is not a planner of the catalog`);
		plan.model = planner;
		plan.settled.push(`planner ${planner}`);
	}
	const thinking = str(json.thinking);
	if (thinking) {
		if (!THINKING.includes(thinking)) errors.push(`thinking: ${thinking} is not one of ${THINKING.join(", ")}`);
		plan.thinking = thinking;
		plan.settled.push(`thinking ${thinking}`);
	}
	for (const [k, field] of [
		["max_turns", "maxTurns"],
		["time_limit", "timeLimit"],
	] as const) {
		if (json[k] == null) continue;
		const n = posInt(json[k]);
		if (n === undefined) errors.push(`${k}: ${String(json[k])} is not a positive integer`);
		else {
			plan[field] = n;
			plan.settled.push(`${k.replace("_", " ")} ${n}`);
		}
	}
	return errors.length ? { errors } : { value: plan };
}

/** The user message: the catalog as JSON, then the request. */
const userMessage = (catalog: unknown, text: string) => `Catalog:\n${JSON.stringify(catalog)}\n\nRequest:\n${text}`;

/** Read a setup request with the model; undefined when the reading never passed (the caller falls back to keywords). */
export async function readSetup<R extends CatalogRobot & { kind: Kind }>(
	ask: Ask,
	text: string,
	x: SetupContext<R> & { refs: { ref: string; name?: string }[] },
): Promise<Reading<Request<R>>> {
	return interpret(ask, SETUP_SYSTEM, userMessage(setupCatalog(x.robots, x.refs), text), (j) => checkSetup(j, x));
}
/** Read a run request with the model. */
export async function readRun(
	ask: Ask,
	text: string,
	x: RunContext & {
		refs: { ref: string; name?: string }[];
		gpuLabels: { index: number; label: string }[];
		budget: string;
	},
): Promise<Reading<RunPlan>> {
	return interpret(ask, RUN_SYSTEM, userMessage(runCatalog(x.experiments, x.refs, x.gpuLabels, x.budget), text), (j) =>
		checkRun(j, x),
	);
}
