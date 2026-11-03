/**
 * Flash: replay a recorded plan with live grounding and no LLM, for any robot with a Flash hook.
 *
 *   pi -p -e src/<robot> --model flash/replay <the robot's task and Flash flags> "Solve the task."
 *
 * Flash is the planner, and in pi the planner is the model: this module registers a `flash/replay`
 * provider whose every turn is the plan's next tool call (with zero usage; an abort ends the
 * replay). The robot's tools execute it, so the session, `finish`, and the `robot_result` row are
 * exactly those of an LLM run. It is mounted by `defineRobot` for a robot whose spec has `flash`.
 *
 * What a plan holds, where it lives, and how it meets the live scene are the robot's (its
 * `FlashHook`). Flash loads the robot's program, lets the robot re-localize it (`start`), then walks
 * the plan in order: the robot rewrites each planned call against what it localized (or skips it,
 * or stops the replay), the call runs, and the robot sees its result (`after`). A robot that names
 * its picks gets grasp retries: a pick the robot judges failed is retried in place after replaying
 * its approach (the motion since the last pick or release), and a pick that never takes hold skips
 * its carry up to and including the next release.
 *
 * Verbatim replay (../replay, `replay/session`) is Flash without re-localization: a recording's
 * calls sent unchanged. The two are separate providers for now.
 *
 * The replay is independent of the action surface: a plan file may hold the robot's tool-call plan
 * (A2, above: re-anchored by the robot's hook), an action-unit sequence (A1: `act` calls replayed
 * as recorded, nothing to re-anchor) or a program (A3: run once through `run_code`); ./kinds.ts
 * reads the latter two by content, the robot's hook finds them next to its own plans
 * (`<name>_units.json`, `<name>_program.py`), and `--flash-plan <file>` names any kind directly.
 * ./generate.ts writes all three from a solved session.
 */

import {
	type Api,
	type AssistantMessage,
	createAssistantMessageEventStream,
	createProvider,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type ToolCall,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { type KindProgram, readKind, replayProgram, replayUnits } from "./kinds.ts";

type Json = Record<string, unknown>;
export type FlashCall = { name: string; arguments: Json };
/**
 * One tool result: its JSON (the text result parsed: the whole text, else the first text part that
 * is JSON, as a `run_code` or `act` result puts its observation after a report line), its images,
 * the result's `details`, and its error text when it failed.
 */
export type FlashReply = { json: Json; images: string[]; details?: Json; error?: string };
/** One planned call; a robot's plans may carry more per entry (e.g. the anchor a waypoint was written against). */
export type FlashEntry = { action: string; arguments: Json };
export type FlashProgram<E extends FlashEntry = FlashEntry> = { name: string; plan: E[] };

/** The robot as the replay sees it: tool calls go out as model turns and resolve with their results. */
export type FlashRobot = {
	/** Run `calls` as one turn; their replies, in order. */
	act: (calls: FlashCall[]) => Promise<FlashReply[]>;
	/** Run one motion call; its reply becomes the latest observation. A failed call ends the replay. */
	move: (call: FlashCall) => Promise<FlashReply>;
	/** The latest motion reply (empty before the first). */
	latest: () => FlashReply;
	/** A line of the next turn's text. */
	note: (line: string) => void;
};

/** How a robot's picks are judged and retried. */
export type FlashPicks = {
	/** A grasp that can be judged and retried. */
	isPick: (name: string) => boolean;
	succeeded: (reply: FlashReply) => boolean;
	attempts: number;
	/** Calls that make up a pick's approach, replayed before a retry (the last `keep`). */
	approach: readonly string[];
	keep: number;
	/** Calls besides picks after which the approach starts over. */
	boundary: readonly string[];
	/** The call that ends a failed pick's carry; it is skipped with the carry. */
	release: string;
};

export type Rewritten = FlashCall | FlashCall[] | "skip" | "stop";

/** A replay underway: how the plan meets the live scene. */
export type FlashReplay<E extends FlashEntry = FlashEntry> = {
	/** Anchors located on the live scene, for the summary. */
	localized: number;
	/**
	 * The call to send for `entry`, with its arguments rewritten (several when one planned move is split
	 * into the robot's per-call steps); "skip" to leave it out; "stop" to end the replay.
	 */
	rewrite(entry: E): Rewritten | Promise<Rewritten>;
	/** Called after each sent call (after its retries). */
	after?: (call: FlashCall, reply: FlashReply) => void | Promise<void>;
	/** Retry picks that did not take hold; without it every call runs once. */
	picks?: FlashPicks;
};

/**
 * A robot's Flash: its programs, and how it replays one. `start` and `rewrite` are methods, so a hook
 * over the robot's own program type is a `FlashHook` (what RobotSpec takes).
 */
export type FlashHook<P extends FlashProgram = FlashProgram> = {
	/**
	 * This episode's program; `cwd` resolves relative plan directories. Throws when there is none. A
	 * units or program plan found next to the robot's own (./kinds.ts `kindFiles`) is returned as is:
	 * the replay plays it without the hook's `start`.
	 */
	load: (cwd: string) => P | KindProgram | Promise<P | KindProgram>;
	/** The robot's own plan read from one file (`--flash-plan`); without it only units and program files may be named. */
	loadFile?: (path: string) => P | Promise<P>;
	/** Re-localize the program on the live scene (the robot may move and observe) and return the replay. */
	start(program: P, robot: FlashRobot): Promise<FlashReplay<P["plan"][number]>>;
	/** The episode is over: no further calls are sent. */
	over: (latest: FlashReply) => boolean;
	solved: (latest: FlashReply) => boolean;
	/** A tool's text result that is not JSON but still reports the episode (e.g. it has ended), as JSON. */
	textResult?: (text: string) => Json | undefined;
};

/**
 * Register `--flash-reanchor on|off`. A string flag on purpose: pi sets a boolean extension flag to
 * true whatever follows it (`--flash-reanchor=false` was true), so the off switch needs a value pi
 * keeps. Both Flash hooks (../flash/recipe.ts, ../../robots/libero/flash.ts) register it.
 */
export function registerReanchorFlag(pi: ExtensionAPI) {
	pi.registerFlag("flash-reanchor", {
		type: "string",
		default: "on",
		description:
			"Flash re-anchors each point with Molmo (services.molmo): on (default) | off (anchors replay at their recorded positions)",
	});
}

/** Whether Flash re-anchors (`--flash-reanchor on|off`); another value throws, and the replay is invalid. */
export function reanchor(pi: ExtensionAPI): boolean {
	const v = String(pi.getFlag("flash-reanchor") ?? "on")
		.trim()
		.toLowerCase();
	if (v === "on" || v === "off") return v === "on";
	throw new Error(`--flash-reanchor takes on or off, not ${JSON.stringify(v)}`);
}

/** Walk `program`: rewrite each call through the robot, run it, retry picks. */
export async function runFlash<P extends FlashProgram>(hook: FlashHook<P>, program: P, robot: FlashRobot) {
	const replay = await hook.start(program, robot);
	const { picks } = replay;
	const over = () => hook.over(robot.latest());
	let recent: FlashCall[] = [];
	let skipCarry = false;
	for (const entry of program.plan) {
		if (over()) break;
		const name = entry.action;
		if (skipCarry && picks) {
			// A pick that never took hold must not fall through into its carry.
			if (name === picks.release) {
				skipCarry = false;
				recent = [];
				continue;
			}
			if (!picks.isPick(name)) continue;
			skipCarry = false;
		}
		const rewritten = await replay.rewrite(entry);
		if (rewritten === "stop") break;
		if (rewritten === "skip") continue;
		for (const call of Array.isArray(rewritten) ? rewritten : [rewritten]) {
			if (over()) break;
			let reply = await robot.move(call);
			if (picks?.isPick(call.name) && !picks.succeeded(reply)) {
				for (let attempt = 1; attempt < picks.attempts && !over(); attempt++) {
					for (const again of recent) await robot.move({ name: again.name, arguments: { ...again.arguments } });
					reply = await robot.move({ name: call.name, arguments: { ...call.arguments } });
					if (picks.succeeded(reply)) break;
				}
				if (!picks.succeeded(reply)) {
					robot.note("pick unconfirmed, skipping its carry");
					skipCarry = true;
				}
			}
			await replay.after?.(call, reply);
			if (!picks) continue;
			if (picks.isPick(call.name) || picks.boundary.includes(call.name)) recent = [];
			else if (picks.approach.includes(call.name)) recent = [...recent, call].slice(-picks.keep);
		}
	}
	return { done: hook.solved(robot.latest()), anchors: replay.localized, plan: program.plan.length };
}

/** Parse the tool results for `ids` out of the transcript the model turn received. */
function repliesFor(messages: Message[], ids: string[], textResult?: (text: string) => Json | undefined): FlashReply[] {
	const byId = new Map<string, Message>();
	for (const m of messages) if (m.role === "toolResult") byId.set(m.toolCallId, m);
	return ids.map((id) => {
		const m = byId.get(id);
		if (!m || m.role !== "toolResult") return { json: {}, images: [], error: "no tool result" };
		const texts = m.content.flatMap((c) => (c.type === "text" ? [c.text] : []));
		const text = texts.join("\n");
		const images = m.content.flatMap((c) => (c.type === "image" ? [c.data] : []));
		const details = m.details as Json | undefined;
		// The whole text (a robot's toolResult), else a part (an act or run_code result puts its observation
		// after a report part), else a line (a result that joined the two).
		const json = jsonIn([text, ...texts, ...text.split("\n")]);
		if (json === undefined) {
			const reported = textResult?.(text);
			return reported ? { json: reported, images, details } : { json: {}, images, details, error: text };
		}
		return m.isError ? { json, images, details, error: text } : { json, images, details };
	});
}

/** The first of `candidates` that parses as a JSON object. */
function jsonIn(candidates: string[]): Json | undefined {
	for (const c of candidates) {
		try {
			const v = JSON.parse(c);
			if (v && typeof v === "object" && !Array.isArray(v)) return v as Json;
		} catch {}
	}
	return undefined;
}

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
/** Flash runs no model, so every turn reports zero usage. */
const USAGE: Usage = { ...ZERO_COST, totalTokens: 0, cost: { ...ZERO_COST, total: 0 } };
const MODEL: Model<"flash"> = {
	id: "replay",
	name: "Flash replay",
	api: "flash",
	provider: "flash",
	baseUrl: "",
	reasoning: false,
	input: ["text", "image"],
	cost: ZERO_COST,
	contextWindow: 100_000_000,
	maxTokens: 16_384,
};

/**
 * Register the `flash/replay` model, replaying the programs of the robot's `hook`. `invalid` marks the
 * episode invalid (../../robot.ts: an env_error): a replay with no program to load is a configuration
 * error, not a failed episode, so evaluations must not count it.
 */
export function flash<P extends FlashProgram>(
	pi: ExtensionAPI,
	hook: FlashHook<P>,
	o: { invalid?: (why: string) => void } = {},
) {
	pi.registerFlag("flash-plan", {
		type: "string",
		default: "",
		description:
			"Replay this plan file instead of searching the robot's plan directories: a units sequence (<name>_units.json, or a JSONL of act lines), a program (.py, or JSON with code), or the robot's own tool-call plan",
	});
	type Turn = { text: string; calls: ToolCall[] };
	let toModel: { resolve: (turn: Turn) => void; reject: (err: Error) => void } | undefined;
	let toReplay: { resolve: (messages: Message[]) => void; reject: (err: Error) => void } | undefined;
	let pending: string[] = [];
	let notes: string[] = [];
	let latest: FlashReply = { json: {}, images: [] };
	let started = false;
	let over = false;
	let stopped: Error | undefined;
	let calls = 0;
	let cwd = process.cwd();

	const say = (turn: Turn) => {
		const f = toModel;
		toModel = undefined;
		f?.resolve(turn);
	};
	const flush = () => {
		const text = notes.join("\n");
		notes = [];
		return text;
	};
	const toolCall = (c: FlashCall): ToolCall => ({
		type: "toolCall",
		id: `flash_${++calls}`,
		name: c.name,
		arguments: c.arguments as ToolCall["arguments"],
	});

	/** An aborted turn ends the replay: the waiting turn fails, and the plan stops at its next call. */
	function stop() {
		stopped ??= new Error("Flash replay aborted");
		over = true;
		toModel?.reject(stopped);
		toReplay?.reject(stopped);
		toModel = toReplay = undefined;
	}

	async function act(batch: FlashCall[]): Promise<FlashReply[]> {
		if (stopped) throw stopped;
		const toolCalls = batch.map(toolCall);
		pending = toolCalls.map((c) => c.id);
		const results = new Promise<Message[]>((resolve, reject) => {
			toReplay = { resolve, reject };
		});
		say({ text: flush(), calls: toolCalls });
		return repliesFor(await results, pending, hook.textResult);
	}

	const robot: FlashRobot = {
		act,
		async move(call) {
			const [reply] = await act([call]);
			if (reply.error !== undefined) throw new Error(`${call.name} failed: ${reply.error}`);
			latest = reply;
			return reply;
		},
		latest: () => latest,
		note: (line) => notes.push(line),
	};

	/** `--flash-plan`: the named file as the kind its content says, else as the robot's own plan. */
	async function loadNamed(file: string): Promise<P | KindProgram> {
		const name = file.replace(/^.*[\\/]/, "").replace(/\.(json|jsonl|py)$/, "");
		const kind = readKind(file, name);
		if (kind) return kind;
		if (!hook.loadFile)
			throw new Error(
				`${file} is neither a units sequence nor a program, and this robot's Flash reads its own plans from its plan directories only`,
			);
		return hook.loadFile(file);
	}

	async function run() {
		const t0 = Date.now();
		let status: "success" | "failure" = "failure";
		let summary: string;
		let program: P | KindProgram;
		try {
			const named = String(pi.getFlag("flash-plan") ?? "").trim();
			program = named ? await loadNamed(named) : await hook.load(cwd);
		} catch (err) {
			if (stopped) return;
			summary = `flash error: no program to replay: ${err instanceof Error ? err.message : String(err)}`;
			o.invalid?.(summary);
			over = true;
			say({ text: flush(), calls: [toolCall({ name: "finish", arguments: { status, summary } })] });
			return;
		}
		try {
			const seconds = () => `${((Date.now() - t0) / 1000).toFixed(1)} s`;
			if ("kind" in program && program.kind === "units") {
				notes.push(`replaying the ${program.name} unit sequence`);
				const out = await replayUnits(program, robot, hook.over);
				status = hook.solved(robot.latest()) ? "success" : "failure";
				summary = `replayed the ${program.name} unit sequence: ${out.sent} of ${program.units.length} units sent (pure replay, nothing re-anchored), ${seconds()}`;
			} else if ("kind" in program && program.kind === "program") {
				notes.push(`replaying the ${program.name} program through run_code`);
				const out = await replayProgram(program, robot);
				status = hook.solved(robot.latest()) ? "success" : "failure";
				summary = `ran the ${program.name} program once through run_code (${out.status}), ${seconds()}`;
			} else {
				notes.push(`replaying the ${program.name} program`);
				const out = await runFlash(hook, program, robot);
				status = out.done ? "success" : "failure";
				summary = `replayed the ${program.name} program: ${out.plan} actions, ${out.anchors} anchors re-localized, ${seconds()}`;
			}
		} catch (err) {
			if (stopped) return;
			summary = `flash error: ${err instanceof Error ? err.message : String(err)}`;
		}
		over = true;
		say({ text: flush(), calls: [toolCall({ name: "finish", arguments: { status, summary } })] });
	}

	/** The next turn of the replay: the plan's next tool calls, or `finish` once it is done. */
	function next(messages: Message[], signal?: AbortSignal): Promise<Turn> {
		if (signal?.aborted) stop();
		if (over) return Promise.resolve({ text: "Flash replay already ran in this session.", calls: [] });
		const turn = new Promise<Turn>((resolve, reject) => {
			toModel = { resolve, reject };
		});
		signal?.addEventListener("abort", stop, { once: true });
		if (!started) {
			started = true;
			void run();
		} else {
			const f = toReplay;
			toReplay = undefined;
			f?.resolve(messages);
		}
		return turn;
	}

	function streamSimple(model: Model<Api>, context: TranscriptContext, options?: SimpleStreamOptions) {
		const stream = createAssistantMessageEventStream();
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: { ...USAGE, cost: { ...USAGE.cost } },
			stopReason: "stop",
			timestamp: Date.now(),
		};
		next(context.messages as Message[], options?.signal).then(
			(turn) => {
				stream.push({ type: "start", partial: message });
				if (turn.text) {
					message.content.push({ type: "text", text: turn.text });
					stream.push({ type: "text_start", contentIndex: 0, partial: message });
					stream.push({ type: "text_delta", contentIndex: 0, delta: turn.text, partial: message });
					stream.push({ type: "text_end", contentIndex: 0, content: turn.text, partial: message });
				}
				for (const call of turn.calls) {
					const contentIndex = message.content.push(call) - 1;
					stream.push({ type: "toolcall_start", contentIndex, partial: message });
					stream.push({ type: "toolcall_end", contentIndex, toolCall: call, partial: message });
				}
				message.stopReason = turn.calls.length ? "toolUse" : "stop";
				stream.push({ type: "done", reason: message.stopReason, message });
				stream.end();
			},
			(err: Error) => {
				message.stopReason = options?.signal?.aborted ? "aborted" : "error";
				message.errorMessage = err.message;
				stream.push({ type: "error", reason: message.stopReason, error: message });
				stream.end();
			},
		);
		return stream;
	}

	pi.registerProvider(
		createProvider({
			id: "flash",
			name: "Flash",
			auth: { apiKey: { name: "Flash", resolve: async () => ({ auth: {} }) } },
			models: [MODEL],
			api: { stream: streamSimple, streamSimple },
		}),
	);

	pi.on("session_start", (_event, ctx) => {
		cwd = ctx.cwd;
		started = over = false;
		stopped = toModel = toReplay = undefined;
		pending = [];
		notes = [];
		latest = { json: {}, images: [] };
	});
}
