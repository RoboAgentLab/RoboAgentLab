/**
 * Layered memory as a pi extension helper. A robot calls `memory(pi, { cell, primitives, ... })`:
 * the agent reads memory with pi's built-in read/ls/grep/find and writes with write, a tool_call guard
 * enforces the memory access boundary, the published corpus is synced from Hugging Face, and the
 * solved recipe is rebuilt from the session branch. Loaded on its own it only adds the command:
 *
 *   pi -e packages/embodied/src/capabilities/memory -p "/memory validate"   (local corpus: dirs.memory of the deployment config)
 */

import { createHash } from "node:crypto";
import {
	existsSync,
	lstatSync,
	mkdirSync,
	readFileSync,
	readlinkSync,
	realpathSync,
	type Stats,
	writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, parse, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI, ExtensionContext, SessionEntry } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { dir as cfgDir } from "../../infra/config.ts";
import { branchProviders, plannerOf } from "../../planner/kind.ts";
import { hasFiles, mergeMemory, rebuildIndex, str, validateMemory } from "./corpus.ts";
import { syncMemory } from "./sync.ts";

const READABLE = ["global", "suite", "task_only", "results"];
/** The default memory section (see MemoryOptions.prompt). */
const SECTION = readFileSync(new URL("./section.md", import.meta.url), "utf8").trim();
/** What closes a memory section that names files, the robot's own too: reading memory is required and reported. */
const REQUIRED = readFileSync(new URL("./required.md", import.meta.url), "utf8").trim();
const ACCESS: Record<string, Access> = {
	read: "read",
	ls: "read",
	grep: "search",
	find: "search",
	write: "write",
	edit: "write",
};

type Access = "read" | "search" | "write";
type Details = { terminated?: unknown; error?: unknown; result?: { error?: unknown }; camera?: unknown } | undefined;
/**
 * Canonical roots: the memory corpus, every robot's memory home, the run's output dir and the cell tag, plus read-only
 * dirs and, when exploring, the cell's inbox and its attempt archive (`<output>/attempts/<tag>`, see ../explore.ts).
 */
export type Guard = {
	root: string;
	home: string;
	output: string;
	tag: string;
	inbox?: string;
	attempts?: string;
	readable?: string[];
};
export type MemoryOptions = {
	/** Corpus name under the memory home, also the Hugging Face subdirectory (default "libero"). */
	robot?: string;
	/** Directory holding every robot's corpus (default $PI_EMBODIED_MEMORY, else ~/.pi/embodied/memory). */
	home?: () => string;
	/** The current cell; without it there is no guard, sync, recipe or prompt (maintenance only). */
	cell?: () => { tag: string; reference: string } | undefined;
	/** State-advancing tools that belong in a recipe (plus successful `segment` calls). */
	primitives?: readonly string[];
	/**
	 * The robot's memory section, as a template (`{{memory_dir}}` ...); default ./section.md: read the
	 * corpus, then the cell's reference and recipe. ../../robot.ts puts it in every evaluation prompt: at
	 * the robot prompt's `{{memory}}`, else at its end, and after the mode's prompt when a mode replaces
	 * the robot's (--units=true, --code=true). ./required.md closes it either way. A default section whose
	 * library holds nothing at session start is replaced by a notice (`section()`).
	 */
	prompt?: (profile: "hf" | "local") => string;
	/**
	 * The robot's own prompt already carries its memory instructions, with no `{{memory}}` slot (LIBERO's
	 * RPent WORKFLOW step, its compact memory text): the section is added only where a mode replaces that prompt.
	 */
	inPrompt?: boolean;
	/** Exploration run: local profile, the cell's inbox becomes writable. */
	explore?: () => boolean;
	/** Directories the agent may also read and search, e.g. the robot's saved state images. */
	readable?: () => string[];
	/**
	 * The robot's prompt has a memory-free form, so `--memory-profile off` may turn memory off: no
	 * memory section, no file tools, no sync, no recipe export. Without it `off` refuses the start.
	 * A function decides at the start (e.g. only in a mode that replaces the robot's prompt).
	 */
	optional?: boolean | (() => boolean);
	/**
	 * Whether the robot has a published corpus on Hugging Face (default true). Without one, runs
	 * default to the local profile, and a local corpus exploration has not written yet is empty, not an error.
	 */
	published?: boolean;
	/**
	 * The cell's audit (`<output>/<tag>.json`, the task_only pair with the recipe) is written through the
	 * `write_audit` tool, never free-written JSON: the model fills AUDIT_PARAMETERS, `facts` adds what the
	 * runtime knows (the cell's identity, the final state, the environment's own solved flag).
	 */
	audit?: { facts: () => Record<string, unknown> };
};

/** What the model states in an audit; everything else in it is the robot's `audit.facts`. */
export const AUDIT_PARAMETERS = Type.Object({
	terminated: Type.Boolean({
		description: "true only when the latest tool result reports terminated: true (the task is solved)",
	}),
	strategy_notes: Type.String({
		description:
			"How you localized, then what you did step by step in trace order with the parameters used; when unsolved, what you tried and where it stalled",
	}),
	memory_files_read: Type.Array(Type.String(), {
		description: "The exact memory file names you read; empty when no matching task memory was found",
	}),
	pick_result: Type.Optional(
		Type.Record(Type.String(), Type.Unknown(), {
			description: "Per pick: keys name the recipe step it came from (e.g. bowl_pi0_pick_step3)",
		}),
	),
	attempts: Type.Optional(Type.Integer({ minimum: 1, description: "Exploration: the attempts this cell took" })),
});

const defaultHome = () => process.env.PI_EMBODIED_MEMORY || join(homedir(), ".pi", "embodied", "memory");

// ---------------------------------------------------------------------------
// Access boundary and recipes (write_recipe_from_states)
// ---------------------------------------------------------------------------

/** The spaces pi's tools turn into a plain one in a path they are given (coding-agent/src/utils/paths.ts). */
const UNICODE_SPACES = /[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g;

/** `input` as pi's tools read it (normalizePath, coding-agent/src/utils/paths.ts) against `cwd`: absolute, its links not looked at. */
function written(input: string | undefined, cwd: string): string {
	let p = (input || ".").replace(UNICODE_SPACES, " ");
	if (p.startsWith("@")) p = p.slice(1);
	if (p === "~") p = homedir();
	else if (p.startsWith("~/") || (sep === "\\" && p.startsWith("~\\"))) p = join(homedir(), p.slice(2));
	else if (p.startsWith("file://")) p = fileURLToPath(p);
	return resolve(cwd, p);
}

/**
 * Whether pi's read looks for `path`, when it is missing, under another spelling: a narrow no-break space
 * before AM/PM, the NFD form, a curly apostrophe (resolveReadPath, coding-agent/src/core/tools/path-utils.ts).
 */
export const respelled = (path: string) => /'| (AM|PM)\./i.test(path) || path.normalize("NFD") !== path;

/**
 * Canonical path: pi's tool path normalization, then the path the system would open. Links are resolved
 * one component at a time, as the system does: a ".." applies to where the links before it lead
 * (fs.realpathSync folds it into the text first), and a dangling link is followed to its target (a write
 * through it lands there). From the first component that does not exist the rest is taken as written; what
 * exists is spelled as the system's realpath reports it (on macOS the name the volume stores, which is one
 * file under several spellings). Throws where the system would open nothing: a link loop, a name below a
 * file, a ".." behind a component that does not exist, and any component that cannot be looked at. The
 * error names the path as written and carries the system's `code` and the `component` it came from.
 */
export function canonicalPath(input: string | undefined, cwd: string): string {
	const path = written(input, cwd);
	const lost = (code: string | undefined, component: string, what = "path does not resolve") =>
		Object.assign(new Error(`${what}: ${path}`), { code, component });
	/** The system's realpath of `at`, which the walk found to exist (a change since is no answer either). */
	const real = (at: string) => {
		try {
			return realpathSync.native(at);
		} catch (e) {
			throw lost((e as NodeJS.ErrnoException).code, at);
		}
	};
	// A link's target is split as the system reads it (on Windows either separator).
	const parts = (p: string) => p.split(sep === "/" ? sep : /[\\/]/);
	let at = parse(path).root; // what is resolved so far: it exists and no link is left in it
	let dir = true; // whether that is a directory
	const todo = path.slice(at.length).split(sep);
	for (let hops = 0; todo.length; ) {
		const name = todo.shift() as string;
		if (!dir) throw lost("ENOTDIR", at);
		if (!name || name === ".") continue;
		if (name === "..") {
			at = dirname(at);
			continue;
		}
		const next = join(at, name);
		let entry: Stats;
		try {
			entry = lstatSync(next);
		} catch (e) {
			const code = (e as NodeJS.ErrnoException).code;
			const rest = todo.filter((part) => part && part !== ".");
			if (code !== "ENOENT" || rest.includes("..")) throw lost(code, next);
			return join(real(at), [name, ...rest].join(sep));
		}
		if (!entry.isSymbolicLink()) {
			at = next;
			dir = entry.isDirectory();
		} else if (++hops > 40) throw lost("ELOOP", next, "too many symbolic links");
		else {
			let target: string;
			try {
				target = readlinkSync(next);
			} catch (e) {
				throw lost((e as NodeJS.ErrnoException).code, next);
			}
			if (!target) throw lost("ENOENT", next);
			// An absolute target starts again at its root (on Windows the root of its drive).
			const root = parse(target).root;
			if (root) at = parse(resolve(at, target)).root;
			todo.unshift(...parts(target.slice(root.length)));
		}
	}
	return real(at);
}

/**
 * Why `path` (canonical) may not be accessed, or undefined. Deny by default: only the published memory
 * (read), the cell's inbox and attempt archive (exploration) and the cell's files in the output dir (`<tag>*`) are reachable.
 * The reason names `shown`, the path as it was asked: a refusal does not tell where a link leads or how a name is stored.
 */
export function denied(path: string, access: Access, g: Guard, shown = path): string | undefined {
	const inside = (base: string) => path === base || path.startsWith(base + sep);
	if (access !== "write" && g.readable?.some(inside)) return undefined;
	if (g.attempts && inside(g.attempts)) return undefined;
	if (!inside(g.root)) {
		if (inside(g.home)) return `access to another robot's memory is denied: ${shown}`;
		if (path === g.output && access === "read") return undefined;
		const name = basename(path);
		const own = name === g.tag || name.startsWith(`${g.tag}.`) || name.startsWith(`${g.tag}_`);
		if (dirname(path) === g.output && own) return undefined;
		return `only the memory and this cell's ${g.tag}.* / ${g.tag}_* files in ${g.output} are accessible: ${shown}`;
	}
	const parts = relative(g.root, path).split(sep).filter(Boolean);
	const ownInbox = g.inbox !== undefined && parts[0] === "_internal" && parts[1] === "inbox" && parts[2] === g.inbox;
	if (access === "write") return ownInbox ? undefined : `writing to memory is denied in this mode: ${shown}`;
	if (!parts.length)
		return access === "read"
			? undefined
			: `search global/, suite/ or task_only/ rather than the memory root: ${shown}`;
	if (READABLE.includes(parts[0]) || (parts.length === 1 && parts[0].endsWith(".md")) || ownInbox) return undefined;
	return `reading this memory path is denied: ${shown}`;
}

export type Call = { name: string; args: Record<string, unknown>; details: Details; isError: boolean };

/** The session's tool calls in order, each with its arguments and its result's details (../flash/generate.ts reads them too). */
export function toolCalls(entries: SessionEntry[]): Call[] {
	const args = new Map<string, Record<string, unknown>>();
	const calls: Call[] = [];
	for (const entry of entries) {
		if (entry.type !== "message") continue;
		const m = entry.message;
		if (m.role === "assistant")
			for (const part of m.content) if (part.type === "toolCall") args.set(part.id, part.arguments);
		if (m.role === "toolResult")
			calls.push({
				name: m.toolName,
				args: args.get(m.toolCallId) ?? {},
				details: m.details as unknown as Details,
				isError: m.isError,
			});
	}
	return calls;
}

/**
 * The command sequence after the last successful `reset` if it reached `terminated`, else undefined.
 * Contract: an executed primitive's result carries `details.terminated` (boolean); a failed call is
 * `isError` or has `details.error` / `details.result.error` and is dropped. Lines are
 * `{"action": name, ...args}`, plus successful `segment` calls as `{action, prompt | point, camera}`.
 */
export function recipe(
	entries: SessionEntry[],
	primitives: ReadonlySet<string>,
): Record<string, unknown>[] | undefined {
	const calls = toolCalls(entries);
	const ok = (c: Call) => !c.isError && !c.details?.error && !c.details?.result?.error;
	let start = 0;
	calls.forEach((c, i) => {
		if (c.name === "reset" && ok(c)) start = i + 1;
	});
	const episode = calls.slice(start);
	if (!episode.some((c) => c.details?.terminated === true)) return undefined;
	return episode
		.filter(
			(c) =>
				ok(c) && (c.name === "segment" || (primitives.has(c.name) && typeof c.details?.terminated === "boolean")),
		)
		.map((c) => {
			if (c.name !== "segment") return { action: c.name, ...c.args };
			const camera = c.details?.camera ?? c.args.camera ?? "agentview";
			const prompt = str(c.args.prompt).trim();
			return prompt ? { action: "segment", prompt, camera } : { action: "segment", point: c.args.point, camera };
		});
}

// ---------------------------------------------------------------------------
// Extension wiring
// ---------------------------------------------------------------------------

function say(ctx: ExtensionContext, text: string, level: "info" | "warning" | "error" = "info") {
	if (ctx.hasUI) ctx.ui.notify(text, level);
	else console.error(`[memory] ${text}`);
}

export function memory(pi: ExtensionAPI, opts: MemoryOptions = {}) {
	pi.registerFlag("memory-profile", {
		type: "string",
		description: "hf (evaluation default) | local | off (no memory: no section, file tools, sync or recipe)",
	});
	pi.registerFlag("auto-merge-memory", {
		type: "boolean",
		default: false,
		description: "Merge the inbox after exploration",
	});
	let root = "";
	let home = "";
	let outputDir = "";
	let profile: "hf" | "local" | "off" = "hf";
	let cell: { tag: string; reference: string } | undefined;
	let guard: Guard | undefined;
	let emptySection = false;
	/** The memory files the agent read this episode: path under the corpus -> SHA-256 when read (../context-version.ts). */
	let loaded: Record<string, string> = {};

	function locate(ctx: ExtensionContext, hf = false) {
		// dirs.memory is the local corpus; the hf profile always reads the synced published one.
		const dir = hf ? "" : cfgDir(pi, "memory");
		const out = cfgDir(pi, "memory_out") || ctx.sessionManager.getSessionDir();
		const robot = opts.robot ?? "libero";
		// Nothing of an earlier session stays behind. Then each place as written: when one does not resolve
		// (thrown below), a session that goes on without memory still names them.
		home = root = outputDir = "";
		outputDir = out ? written(out, ctx.cwd) : "";
		home = written(opts.home?.() || defaultHome(), ctx.cwd);
		root = written(dir || join(home, robot), ctx.cwd);
		outputDir = out ? canonicalPath(out, ctx.cwd) : "";
		home = canonicalPath(home, ctx.cwd);
		root = canonicalPath(dir || join(home, robot), ctx.cwd);
	}

	/**
	 * The directory the published corpus is synced into, which the sync names the corpus by: the root as it
	 * resolved; or, where the volume stores this very directory under another spelling of the robot's name, the
	 * robot's spelling (the same entry, not a link to it).
	 */
	function corpusDir(): string {
		const alias = join(dirname(root), opts.robot ?? "libero");
		try {
			const a = lstatSync(alias, { bigint: true });
			const r = lstatSync(root, { bigint: true });
			if (alias !== root && a.isDirectory() && a.dev === r.dev && a.ino === r.ino) return alias;
		} catch {}
		return root;
	}

	/** Why this session cannot have memory (robot.ts refuses to start), else undefined. */
	let problem: string | undefined;

	// A failure here is a fail-closed start, not an extension error: pi would log a throwing
	// handler and go on, and the robot's own session_start would run the episode memory-less
	// (verify3 bug 38). The problem is recorded and robot.ts reads it before the robot boots.
	pi.on("session_start", async (_event, ctx) => {
		guard = undefined;
		loaded = {};
		problem = undefined;
		emptySection = false;
		try {
			await establish(ctx);
			// Snapshot what the default section points to; a robot's custom template may name other files.
			// A tier counts only when it holds a file: a merge creates global/, suite/ and task_only/ even
			// when it publishes nothing. A cell without a reference (the real robots) has no task pair to
			// name, so any file under task_only/ counts for it.
			if (cell && profile !== "off" && !opts.prompt) {
				const ref = cell.reference;
				const task = ref
					? [`${ref}.json`, `${ref}_recipe.jsonl`].some((f) => existsSync(join(root, "task_only", f)))
					: hasFiles(join(root, "task_only"));
				emptySection =
					!existsSync(join(root, "MEMORY.md")) &&
					!["global", "suite"].some((s) => hasFiles(join(root, s))) &&
					!task;
			}
		} catch (err) {
			guard = undefined;
			problem = err instanceof Error ? err.message : String(err);
		}
	});

	async function establish(ctx: ExtensionContext) {
		cell = opts.cell?.();
		const explore = opts.explore?.() ?? false;
		const explicit = str(pi.getFlag("memory-profile"));
		const requested = explicit || (explore || opts.published === false ? "local" : "hf");
		// This session's places, as its profile reads them (the hf profile has the synced corpus, never
		// dirs.memory). A start refused below names its own, never an earlier session's. One that does not
		// resolve stops a session that will use memory, not one without a cell or with memory off (the /memory
		// command looks again and reports it).
		let unresolved: (Error & { code?: string; component?: string }) | undefined;
		try {
			locate(ctx, cell !== undefined && requested === "hf");
		} catch (e) {
			unresolved = e as Error;
		}
		if (!cell) return;
		if (!/^[\w.-]+$/.test(cell.tag) || /^\.+$/.test(cell.tag))
			throw new Error(`invalid memory cell tag: ${cell.tag}`);
		if (requested !== "hf" && requested !== "local" && requested !== "off")
			throw new Error(`unknown --memory-profile ${requested}`);
		if (explore && requested === "hf") throw new Error("--explore cannot be used with --memory-profile hf");
		if (requested === "off") {
			if (explore) throw new Error("--explore writes memory: it cannot run with --memory-profile off");
			if (!(typeof opts.optional === "function" ? opts.optional() : opts.optional))
				throw new Error(
					`--memory-profile off: ${opts.robot ?? "libero"}'s prompt reads memory and has no form without it`,
				);
			// No guard: a file tool is refused (none is active), and nothing is synced or exported.
			profile = "off";
			return;
		}
		if (unresolved)
			throw new Error(
				unresolved.component
					? `${unresolved.message} (${unresolved.code} at ${unresolved.component})`
					: unresolved.message,
			);
		profile = requested;
		guard = {
			root,
			home,
			output: outputDir,
			tag: cell.tag,
			inbox: explore ? cell.tag : undefined,
			attempts: explore && outputDir ? join(outputDir, "attempts", cell.tag) : undefined,
		};
		if (explore) return;
		if (profile === "hf") await syncMemory(corpusDir(), (m) => say(ctx, m, "warning"));
		// A run that asks for memory (a published corpus, --memory-profile, or dirs.memory) refuses to start
		// without one: it would otherwise go on as a run without memory (the notice of `section()`, or a
		// robot's own section naming files that do not exist) although memory was asked for.
		else if (
			(explicit || opts.published !== false || cfgDir(pi, "memory")) &&
			!existsSync(join(root, "MEMORY.md")) &&
			!["global", "suite", "task_only"].some((s) => hasFiles(join(root, s)))
		)
			throw new Error(`local memory corpus not found at ${root}; run exploration first or use --memory-profile hf`);
	}

	// A robot's file tools fail closed: until session_start has established every root, nothing is reachable.
	pi.on("tool_call", (event, ctx) => {
		const access = ACCESS[event.toolName];
		if (!access || !opts.cell) return undefined;
		const g = guard;
		if (!g || ![g.root, g.home, g.output].every(isAbsolute) || !g.tag)
			return { block: true, reason: "file access is disabled: the memory guard has no memory, output dir or cell" };
		try {
			// A read-only dir that does not resolve is not readable; it does not close everything else.
			const readable = (opts.readable?.() ?? []).filter(Boolean).flatMap((d) => {
				try {
					return [canonicalPath(d, ctx.cwd)];
				} catch {
					return [];
				}
			});
			const rules = { ...g, readable };
			const input = event.input as unknown as { path?: unknown };
			const asked = written(str(input.path), ctx.cwd);
			let path: string;
			try {
				path = canonicalPath(str(input.path), ctx.cwd);
			} catch (e) {
				// Nothing opens there. Where the place as asked is refused anyway, or the walk stopped in a refused
				// place (a link led there), the answer is that refusal, as for any other name there: it does not
				// tell what exists outside the boundary.
				const stopped = (e as { component?: string }).component;
				return {
					block: true,
					reason:
						denied(asked, access, rules) ??
						(stopped === undefined ? undefined : denied(stopped, access, rules, asked)) ??
						`file access denied: ${(e as Error).message}`,
				};
			}
			// The decision is the opened file's. A refusal gives the rule of the place as asked (by its spelling)
			// where that refuses too, so that a link at a refused place does not tell where it leads.
			const reason = denied(path, access, rules, asked);
			if (reason) return { block: true, reason: denied(asked, access, rules) ?? reason };
			// The tool gets the path that was judged, so that it opens that file and no other: its own reading
			// of the argument ("~", "@", a relative path, links) is done. Three things would still move it or
			// hang it. The tools turn the unicode spaces of a path into plain ones; a pipe or a device waits for
			// a peer; and `read` looks for a missing file under other spellings, which may be a link elsewhere.
			if (path.search(UNICODE_SPACES) !== -1)
				return { block: true, reason: `file access denied: ${asked} cannot be named` };
			let entry: Stats | undefined;
			try {
				entry = lstatSync(path);
			} catch {}
			if (entry && !entry.isFile() && !entry.isDirectory())
				return { block: true, reason: `file access denied: ${asked} is not a file or a directory` };
			if (event.toolName === "read" && !entry && respelled(path))
				return { block: true, reason: `no such file: ${asked}` };
			input.path = path;
			return undefined;
		} catch (e) {
			return { block: true, reason: `file access denied: ${(e as Error).message}` };
		}
	});

	pi.on("tool_result", (event, ctx) => {
		if (event.toolName !== "read" || event.isError || !root) return;
		try {
			const path = canonicalPath(str((event.input as { path?: unknown }).path), ctx.cwd);
			const rel = relative(root, path);
			if (!rel || rel.startsWith("..") || isAbsolute(rel)) return;
			loaded[rel.split(sep).join("/")] = createHash("sha256").update(readFileSync(path)).digest("hex");
		} catch {}
	});

	// Once the run has settled (continuations such as DISTIL included): export the recipe, merge drafts.
	pi.on("agent_settled", async (_event, ctx) => {
		if (!cell || !outputDir || profile === "off") return;
		const branch = ctx.sessionManager.getBranch();
		const commands = recipe(branch, new Set(opts.primitives ?? []));
		const path = join(outputDir, `${cell.tag}_recipe.jsonl`);
		if (commands) {
			mkdirSync(outputDir, { recursive: true });
			writeFileSync(path, commands.map((c) => `${JSON.stringify(c)}\n`).join(""));
			// Who solved it (../planner.ts): a person's or a replayed solve is not a model's recipe.
			writeFileSync(
				join(outputDir, `${cell.tag}_recipe.meta.json`),
				`${JSON.stringify({ planner: plannerOf(branchProviders(branch)) })}\n`,
			);
		}
		say(ctx, commands ? `recipe: ${path}` : "recipe: not written (cell unsolved)");
		let failed = false;
		for (const e of branch)
			if (e.type === "message" && e.message.role === "assistant") failed = e.message.stopReason === "error";
		// Exploration memory is the model's own: a run a person planned (human/operator) is not merged.
		const human = plannerOf(branchProviders(branch)) === "human";
		if (human && guard?.inbox) say(ctx, "memory not merged: a person planned this run (human/operator)");
		if (guard?.inbox && pi.getFlag("auto-merge-memory") === true && !failed && !human)
			say(ctx, `memory merged: ${JSON.stringify(await mergeMemory(root, cell.tag, outputDir, !!commands))}`);
	});

	if (opts.audit) {
		const audit = opts.audit;
		pi.registerTool({
			name: "write_audit",
			label: "write_audit",
			description:
				"Write this cell's audit (the task_only record paired with the exported recipe). The runtime adds the suite, task, seed, final state and its own solved flag; call it again to replace the audit. Call it before `finish`.",
			parameters: AUDIT_PARAMETERS,
			async execute(_id, params) {
				if (!cell || !outputDir) throw new Error("write_audit: no cell or output dir this session");
				const path = join(outputDir, `${cell.tag}.json`);
				// The runtime's facts win over anything the model states under the same key.
				const facts = audit.facts();
				const doc = { ...facts, ...params, ...facts };
				mkdirSync(outputDir, { recursive: true });
				writeFileSync(path, `${JSON.stringify(doc, null, 2)}\n`);
				return { content: [{ type: "text", text: `audit written: ${path}` }], details: { path, audit: doc } };
			},
		});
	}

	const usage = "usage: /memory sync | validate | index | merge <cell> [output-dir] [--solved]";
	pi.registerCommand("memory", {
		description: "Memory corpus maintenance: sync | validate | index | merge <cell> [output-dir] [--solved]",
		handler: async (args, ctx) => {
			// Without a cell nothing was established: look again, and report a place that does not resolve.
			if (!cell || !root) locate(ctx);
			const [sub, ...rest] = args.trim().split(/\s+/);
			if (sub === "sync") {
				await syncMemory(corpusDir(), (m) => say(ctx, m, "warning"));
				say(ctx, `memory: ${root}`);
			} else if (sub === "validate") {
				const problems = validateMemory(root);
				if (problems.length) process.exitCode = 1;
				say(
					ctx,
					problems.length ? problems.join("\n") : "local memory is valid",
					problems.length ? "error" : "info",
				);
			} else if (sub === "index") {
				say(ctx, rebuildIndex(root) ?? "no indexable memory leaves; MEMORY.md not regenerated");
			} else if (sub === "merge" && rest[0] && rest[0] !== "--solved") {
				const [tag, dir] = rest.filter((a) => a !== "--solved");
				const result = await mergeMemory(
					root,
					tag,
					dir ? canonicalPath(dir, ctx.cwd) : outputDir,
					rest.includes("--solved"),
				);
				say(ctx, JSON.stringify(result, null, 2));
			} else say(ctx, usage, "error");
		},
	});

	return {
		/** Built-in tools the agent uses on memory; add them to setActiveTools (none with --memory-profile off). */
		get tools(): string[] {
			return profile === "off"
				? []
				: ["read", "ls", "grep", "find", "write", ...(opts.audit ? ["write_audit"] : [])];
		},
		/** Why memory could not be established this session (no corpus, a bad cell tag or profile); robot.ts fails the start with it. */
		configError: () => problem,
		/** The memory files the agent read this episode, with their SHA-256 at the time. */
		loaded: () => ({ ...loaded }),
		get profile() {
			return profile;
		},
		/**
		 * The memory section of an evaluation prompt: the robot's template or ./section.md, closed by
		 * ./required.md. When the default section's library held nothing at session start (no index, no file
		 * in global/ or suite/, no task reference; for a cell without a reference, no file under task_only/),
		 * a notice replaces it: the planner would spend its first calls on reads that can only fail.
		 * Undefined without a cell (nothing to read) or with memory off.
		 */
		section(): string | undefined {
			if (!cell || profile === "off") return undefined;
			if (emptySection)
				return this.render(
					"# Memory\nCell `{{recipe_tag}}`. No memory is available for this cell at session start. " +
						"Skip memory reads and searches; proceed with the task. " +
						"You write only to `{{output_dir}}/`, and you read nothing else on disk. " +
						"In your `finish` summary, say that none matched.",
				);
			return this.render(`${(opts.prompt?.(profile) ?? SECTION).trim()}\n\n${REQUIRED}`, { task: "" });
		},
		/** Fill {{memory_dir}}, {{memory_inbox}}, {{memory_profile}}, {{output_dir}}, {{recipe_tag}}, {{reference_tag}} and `extra`. */
		render(text: string, extra: Record<string, string | number> = {}): string {
			const vars: Record<string, string | number> = {
				memory_dir: root,
				memory_inbox: join(root, "_internal", "inbox", cell?.tag ?? ""),
				memory_profile: profile,
				output_dir: outputDir,
				recipe_tag: cell?.tag ?? "",
				reference_tag: cell?.reference ?? "",
				...extra,
			};
			return text.replace(/\{\{(\w+)\}\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m));
		},
	};
}

export default function (pi: ExtensionAPI) {
	memory(pi);
}
