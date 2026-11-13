import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fsModule, {
	chmodSync,
	existsSync,
	lstatSync,
	mkdirSync,
	mkdtempSync,
	type PathLike,
	readdirSync,
	readFileSync,
	readlinkSync,
	realpathSync,
	rmSync,
	statSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import fsPromises from "node:fs/promises";
import { createServer } from "node:http";
import { syncBuiltinESMExports } from "node:module";
import type { AddressInfo } from "node:net";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";
import { mergeMemory, renderLeaf, splitFrontmatter, validateMemory } from "../src/capabilities/memory/corpus.ts";
import { canonicalPath, denied, memory, recipe, respelled } from "../src/capabilities/memory/index.ts";
import { deployFlags } from "./helpers/deployment.ts";

const leaf = (meta: string, body: string) => `---\n${meta}\n---\n${body}`;
const globalDraft = (cells: string, body: string) =>
	leaf(
		`id: lift-first\nscope: global\nkind: primitive\ntitle: Lift before carrying\napplies_when: "after pi0_pick: carry"\nconfidence: single-shot\nevidence:\n  cells: [${cells}]\n  attempts: 2`,
		body,
	);

test("frontmatter round-trips through renderLeaf", () => {
	const meta = { id: "x", title: "a: b", n: 3, yes: "yes", evidence: { cells: ["10_t2_s0", "on"], attempts: 1 } };
	const text = renderLeaf(meta, "\n# Body\n");
	assert.deepEqual(splitFrontmatter(text), { meta, body: "\n# Body\n" });
});

test("merge publishes drafts, merges evidence, archives conflicts and task pairs", async () => {
	const root = mkdtempSync(join(tmpdir(), "memory-"));
	const run = mkdtempSync(join(tmpdir(), "run-"));
	const inbox = (cell: string) => join(root, "_internal", "inbox", cell);

	mkdirSync(join(inbox("10_t2_s0"), "wip"), { recursive: true });
	writeFileSync(
		join(inbox("10_t2_s0"), "new_global_primitive_lift-first_draft.md"),
		globalDraft("10_t2_s0", "\nLift.\n"),
	);
	writeFileSync(
		join(inbox("10_t2_s0"), "suite_10_t2_s0_draft.md"),
		leaf(
			"scope: suite\nsuite: libero10\nregime: task\ntask_id: 2\ntask_language: put the bowl on the plate\nconfidence: single-shot\nevidence:\n  cells: [10_t2_s0]",
			"\n## Applicable pattern\n",
		),
	);
	writeFileSync(join(inbox("10_t2_s0"), "broken.md"), "no frontmatter");
	writeFileSync(join(run, "10_t2_s0.json"), "{}");
	writeFileSync(join(run, "10_t2_s0_recipe.jsonl"), "{}\n");

	const first = await mergeMemory(root, "10_t2_s0", run, true);
	assert.deepEqual([first.global, first.suite, first.task, first.evidence, first.conflicts], [1, 1, 1, 0, 0]);
	assert.match(first.skipped[0], /^broken\.md: missing YAML frontmatter/);
	assert.ok(existsSync(join(root, "global", "lift-first.md")));
	assert.ok(existsSync(join(root, "suite", "suite_libero10_task_t2.md")));
	assert.ok(existsSync(join(root, "_internal", "merged", "10_t2_s0", "wip")));
	assert.ok(!existsSync(inbox("10_t2_s0")));
	assert.ok(existsSync(join(root, "task_only", "10_t2_s0_recipe.jsonl")));
	assert.match(
		readFileSync(join(root, "MEMORY.md"), "utf8"),
		/- \[Lift before carrying\]\(global\/lift-first\.md\) — after pi0_pick: carry/,
	);
	assert.deepEqual(validateMemory(root), []);

	mkdirSync(inbox("goal_t1_s0"), { recursive: true });
	writeFileSync(
		join(inbox("goal_t1_s0"), "new_global_primitive_lift-first.md"),
		globalDraft("goal_t1_s0", "\nOther.\n"),
	);
	const second = await mergeMemory(root, "goal_t1_s0", run, false);
	assert.deepEqual([second.global, second.evidence, second.conflicts, second.task], [0, 1, 1, 0]);
	const merged = splitFrontmatter(readFileSync(join(root, "global", "lift-first.md"), "utf8"));
	assert.deepEqual(merged.meta.evidence, { cells: ["10_t2_s0", "goal_t1_s0"], attempts: 4 });
	assert.equal(merged.meta.confidence, "probable");
	assert.equal(merged.body, "\n\nLift.\n");
	assert.ok(existsSync(join(root, "_internal", "conflicts", "lift-first__from_goal_t1_s0.md")));
});

test("a LIBERO-plus cell publishes its own suite note; the plain name is refused either way", async () => {
	const root = mkdtempSync(join(tmpdir(), "memory-"));
	const run = mkdtempSync(join(tmpdir(), "run-"));
	const suite = (cell: string, family: string) => {
		const inbox = join(root, "_internal", "inbox", cell);
		mkdirSync(inbox, { recursive: true });
		writeFileSync(
			join(inbox, `suite_${cell}_draft.md`),
			leaf(
				`scope: suite\nsuite: ${family}\nregime: task\ntask_id: 2\ntask_language: put the bowl on the plate\nconfidence: single-shot\nevidence:\n  cells: [${cell}]`,
				"\n## Applicable pattern\n",
			),
		);
	};
	suite("10_task_plus_t2_s0", "libero10");
	const plain = await mergeMemory(root, "10_task_plus_t2_s0", run, false);
	assert.equal(plain.suite, 0);
	assert.match(plain.skipped[0], /suite 'libero10' must end with _plus for cell 10_task_plus_t2_s0/);
	suite("10_task_plus_t2_s0", "libero10_plus");
	assert.equal((await mergeMemory(root, "10_task_plus_t2_s0", run, false)).suite, 1);
	assert.ok(existsSync(join(root, "suite", "suite_libero10_plus_task_t2.md")));
	assert.ok(!existsSync(join(root, "suite", "suite_libero10_task_t2.md")));
	suite("10_task_t2_s0", "libero10_plus");
	const pro = await mergeMemory(root, "10_task_t2_s0", run, false);
	assert.match(pro.skipped[0], /suite 'libero10_plus' must not carry _plus for cell 10_task_t2_s0/);
	suite("10_task_t2_s0", "libero10");
	assert.equal((await mergeMemory(root, "10_task_t2_s0", run, false)).suite, 1);
	assert.deepEqual(validateMemory(root), []);
});

test("guard enforces the memory boundary", () => {
	const g = {
		root: "/m/libero",
		home: "/m",
		output: "/tmp/run",
		tag: "10_t2_s0",
		inbox: "10_t2_s0",
		attempts: "/tmp/run/attempts",
	};
	const ro = { ...g, inbox: undefined, attempts: undefined };
	assert.equal(denied("/m/libero/MEMORY.md", "read", ro), undefined);
	assert.equal(denied("/m/libero/global/a.md", "read", ro), undefined);
	assert.equal(denied("/m/libero", "read", ro), undefined);
	assert.match(denied("/m/libero", "search", ro) ?? "", /memory root/);
	assert.match(denied("/m/libero/_internal/inbox/10_t2_s0/x.md", "read", ro) ?? "", /reading/);
	assert.match(denied("/m/libero/global/a.md", "write", ro) ?? "", /writing/);
	assert.match(denied("/m/robotwin/MEMORY.md", "read", ro) ?? "", /another robot/);
	assert.equal(denied("/tmp/run/10_t2_s0.json", "write", ro), undefined);
	assert.equal(denied("/tmp/run/10_t2_s0.json", "read", ro), undefined);
	assert.equal(denied("/tmp/run", "read", ro), undefined);
	assert.match(denied("/tmp/run", "search", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/10_t2_s01.json", "write", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/sub/10_t2_s0.json", "write", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/other.jsonl", "read", ro) ?? "", /only the memory/);
	assert.match(denied("/etc/passwd", "read", ro) ?? "", /only the memory/);
	assert.match(denied("/", "search", ro) ?? "", /only the memory/);
	assert.equal(denied("/m/libero/_internal/inbox/10_t2_s0/wip/notes.md", "write", g), undefined);
	assert.equal(denied("/m/libero/_internal/inbox/10_t2_s0/wip/notes.md", "read", g), undefined);
	assert.match(denied("/m/libero/_internal/inbox/other/x.md", "write", g) ?? "", /writing/);
	assert.equal(denied("/tmp/run/attempts/attempt_1_failed.json", "write", g), undefined);
	assert.equal(denied("/tmp/run/attempts", "search", g), undefined);
	assert.match(denied("/tmp/run/attempts/attempt_1_failed.json", "write", ro) ?? "", /only the memory/);
	assert.match(denied("/tmp/run/attempts2/x.json", "write", g) ?? "", /only the memory/);
});

test("recipe keeps successful primitives and segments after the last reset", () => {
	let id = 0;
	const entries: unknown[] = [];
	const call = (name: string, args: Record<string, unknown>, details: unknown, isError = false) => {
		const toolCallId = `c${id++}`;
		const base = { id: toolCallId, parentId: null, timestamp: "" };
		entries.push(
			{
				...base,
				type: "message",
				message: { role: "assistant", content: [{ type: "toolCall", id: toolCallId, name, arguments: args }] },
			},
			{
				...base,
				type: "message",
				message: { role: "toolResult", toolCallId, toolName: name, content: [], details, isError, timestamp: 0 },
			},
		);
	};
	const moved = { result: {}, terminated: false };
	call("move_to", { xyz: [0, 0, 1] }, moved);
	call("reset", {}, moved);
	call("view_env_state", {}, moved);
	call("segment", { prompt: " the black bowl " }, { found: true, camera: "agentview" });
	call("segment", { point: [1, 2] }, { found: false, error: "no mask" });
	call("rotate_wrist", {}, undefined, true);
	call("move_to", { xyz: [0.1, 0, 1] }, { result: { name: "move_to" }, terminated: false });
	call("release", {}, { result: {}, terminated: true });
	const primitives = new Set(["move_to", "release", "rotate_wrist", "reset"]);
	assert.deepEqual(recipe(entries as SessionEntry[], primitives), [
		{ action: "segment", prompt: "the black bowl", camera: "agentview" },
		{ action: "move_to", xyz: [0.1, 0, 1] },
		{ action: "release" },
	]);
	assert.equal(recipe(entries.slice(0, 4) as SessionEntry[], primitives), undefined);
});

/** memory() wired to a stub pi: returns the tool_call hook after session_start (unless `start` is false). */
async function guarded(
	opts: {
		cell?: string;
		output?: string;
		home?: string;
		dir?: string;
		profile?: string;
		optional?: boolean;
		readable?: () => string[];
		start?: boolean;
		explore?: boolean;
		audit?: { facts: () => Record<string, unknown> };
	} = {},
) {
	// The stored spelling: the guard names what exists as the system's realpath does.
	const base = realpathSync.native(mkdtempSync(join(tmpdir(), "guard-")));
	const home = opts.home ?? join(base, "memory");
	const run = join(base, "run");
	if (!opts.home) {
		mkdirSync(join(home, "libero", "global"), { recursive: true });
		mkdirSync(join(home, "robotwin"), { recursive: true });
		writeFileSync(join(home, "libero", "MEMORY.md"), "# memory\n");
	}
	mkdirSync(run);
	writeFileSync(join(base, "secret.txt"), "secret\n");
	const hooks = new Map<string, (event: unknown, ctx: unknown) => unknown>();
	const flags: Record<string, unknown> = deployFlags({
		"output-dir": opts.output ?? run,
		...(opts.dir ? { "memory-dir": opts.dir } : {}),
		...(opts.profile ? { "memory-profile": opts.profile } : {}),
	});
	const tools = new Map<string, any>();
	const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
	const pi = {
		on: (name: string, fn: (event: unknown, ctx: unknown) => unknown) => hooks.set(name, fn),
		registerFlag: () => {},
		registerCommand: (name: string, spec: { handler: (args: string, ctx: unknown) => Promise<void> }) =>
			commands.set(name, spec),
		registerTool: (t: any) => tools.set(t.name, t),
		getFlag: (name: string) => flags[name],
	} as unknown as ExtensionAPI;
	const cell = opts.cell;
	const told = { home };
	const mem = memory(pi, {
		home: () => told.home,
		cell: () => (cell === undefined ? undefined : { tag: cell, reference: cell }),
		explore: () => opts.explore ?? true,
		...(opts.optional ? { optional: true } : {}),
		...(opts.readable ? { readable: opts.readable } : {}),
		...(opts.audit ? { audit: opts.audit } : {}),
	});
	const ctx = { cwd: run, hasUI: false, sessionManager: { getSessionDir: () => "", getBranch: () => [] } };
	if (opts.start !== false) await hooks.get("session_start")?.({}, ctx);
	const call = (toolName: string, path?: string, input: { path?: unknown } = { path }) =>
		hooks.get("tool_call")?.({ type: "tool_call", toolName, toolCallId: "t", input }, ctx) as
			| { block: true; reason: string }
			| undefined;
	/** The path the tool is handed for an allowed call (the hook rewrites its input), else the refusal. */
	const handed = (toolName: string, path?: string) => {
		const input: { path?: unknown } = { path };
		return call(toolName, path, input)?.reason ?? input.path;
	};
	const command = (name: string, args: string) => commands.get(name)?.handler(args, ctx);
	/** A later session on the same instance, told another memory home. */
	const again = async (otherHome: string) => {
		told.home = otherHome;
		await hooks.get("session_start")?.({}, ctx);
	};
	return { base, home, run, call, handed, tools, mem, command, again };
}

const FILE_TOOLS = ["read", "ls", "grep", "find", "write", "edit"];
/** Take every permission off `dir`; false when that does not keep this user out (root, or one with the capability). */
function lockOut(dir: string): boolean {
	chmodSync(dir, 0o000);
	try {
		readdirSync(dir);
	} catch {
		return true;
	}
	chmodSync(dir, 0o700);
	return false;
}
/** pi's own path rules, which the guard copies (written, the unicode spaces, respelled): the tests ask the original. */
const PI_PATH_UTILS = new URL("../../coding-agent/src/core/tools/path-utils.ts", import.meta.url);

test("guard denies file tools outside memory and the cell's output files", async () => {
	const { base, home, run, call } = await guarded({ cell: "10_t2_s0" });
	for (const tool of FILE_TOOLS) {
		assert.ok(call(tool, join(base, "secret.txt"))?.block, `${tool} absolute`);
		assert.ok(call(tool, "../secret.txt")?.block, `${tool} ..`);
		assert.ok(call(tool, join(home, "libero", "..", "..", "secret.txt"))?.block, `${tool} memory/..`);
		assert.ok(call(tool, join(home, "robotwin"))?.block, `${tool} other robot`);
		assert.ok(call(tool, "/")?.block, `${tool} /`);
	}
	assert.ok(call("grep")?.block, "default path is cwd = output dir root");
	assert.ok(call("find")?.block, "find is a search: not the output dir root");
	assert.ok(call("find", join(home, "libero"))?.block, "find is a search: not the memory root");
	assert.ok(call("edit", join(home, "libero", "MEMORY.md"))?.block, "edit is a write: not the published memory");
	assert.ok(call("write", "notes.md")?.block, "relative write in cwd");

	// Symlink escapes: an existing link, a link directory, and a dangling link whose write lands outside.
	symlinkSync(join(base, "secret.txt"), join(run, "10_t2_s0.json"));
	symlinkSync(base, join(home, "libero", "global", "up"));
	symlinkSync(join(base, "planted.txt"), join(run, "10_t2_s0_audit.json"));
	assert.equal(canonicalPath(join(run, "10_t2_s0_audit.json"), run), join(base, "planted.txt"));
	assert.match(call("read", "10_t2_s0.json")?.reason ?? "", /only the memory/);
	assert.ok(call("read", join(home, "libero", "global", "up", "secret.txt"))?.block);
	assert.ok(call("write", join(run, "10_t2_s0_audit.json"))?.block);
});

test("a link loop is refused, not followed for ever", (t) => {
	// In a child with a time limit: a resolver that keeps following the links does not return, and nothing in
	// this process could stop it. The hook turns the refusal into a block, as for any path that does not resolve.
	const base = realpathSync.native(mkdtempSync(join(tmpdir(), "guard-loops-")));
	t.after(() => rmSync(base, { recursive: true, force: true }));
	const at = (name: string) => join(base, name);
	symlinkSync("self", at("self"));
	symlinkSync("pong", at("ping"));
	symlinkSync("ping", at("pong"));
	symlinkSync("inside/x", at("inside"));
	symlinkSync(at("abs-self"), at("abs-self"));
	symlinkSync(at("abs-pong"), at("abs-ping"));
	symlinkSync(at("abs-ping"), at("abs-pong"));
	mkdirSync(at("dir"));
	symlinkSync("../dir/round", at("dir/round"));
	// Not loops: a row of links ends in a file. The system gives up after 32 (macOS) or 40 (Linux) of them.
	writeFileSync(at("c0"), "x");
	for (let i = 1; i <= 45; i++) symlinkSync(`c${i - 1}`, at(`c${i}`));
	// Not a loop either, but one for a resolver that folds ".." into the text first: the target then names
	// the link itself, while the system goes through `hop` and finds the file beside it.
	mkdirSync(at("away/sub"), { recursive: true });
	writeFileSync(at("away/note.md"), "x");
	symlinkSync(at("away/sub"), at("dir/hop"));
	symlinkSync("hop/../note.md", at("dir/note.md"));
	const asks = ["self", "ping", "inside", "dir/round", "self/below", "c40", "c41", "dir/note.md"].map(at);
	asks.push(at("abs-self"), at("abs-ping"));
	const resolve =
		"import(process.argv[1]).then((m) => console.log(JSON.stringify(process.argv.slice(2).map((p) => { try { return m.canonicalPath(p, '/'); } catch (e) { return 'threw ' + e.message + ' (' + e.code + ' at ' + e.component + ')'; } }))))";
	const module = new URL("../src/capabilities/memory/index.ts", import.meta.url).href;
	// The child answers in under a second; the limit is for a resolver that never does, and for a loaded machine.
	const r = spawnSync(process.execPath, ["--experimental-strip-types", "-e", resolve, module, ...asks], {
		encoding: "utf8",
		timeout: 120_000,
		killSignal: "SIGKILL",
	});
	const got: string[] = JSON.parse(r.stdout || "[]");
	assert.equal(got.length, asks.length, r.error?.message ?? r.stderr);
	for (const i of [1, 2, 3, 4, 6, 8, 9])
		assert.match(got[i], /^threw too many symbolic links: .* \(ELOOP at /s, asks[i]);
	assert.equal(got[0], `threw too many symbolic links: ${at("self")} (ELOOP at ${at("self")})`);
	// 40 links are followed and no more.
	assert.equal(got[5], at("c0"));
	assert.equal(got[7], at("away/note.md"));
});

test("guard allows the memory, the exploration inbox and the audit write", async () => {
	const { home, run, call } = await guarded({ cell: "10_t2_s0" });
	assert.equal(call("write", join(run, "10_t2_s0.json")), undefined);
	assert.equal(call("write", "10_t2_s0.json"), undefined);
	writeFileSync(join(run, "10_t2_s0.json"), "{}");
	assert.equal(call("read", "10_t2_s0.json"), undefined);
	assert.equal(call("ls"), undefined);
	assert.equal(call("read", join(home, "libero", "MEMORY.md")), undefined);
	assert.equal(call("grep", join(home, "libero", "global")), undefined);
	assert.equal(call("write", join(home, "libero", "_internal", "inbox", "10_t2_s0", "x.md")), undefined);
	assert.ok(call("write", join(home, "libero", "_internal", "inbox", "10_t2_s1", "x.md"))?.block);
	assert.ok(call("write", join(home, "libero", "global", "x.md"))?.block);
	assert.equal(call("bash"), undefined, "tools other than the file tools are not the guard's business");
});

test("the guard judges the file the system opens, and hands every tool that path", async () => {
	const { base, home, run, call, handed } = await guarded({ cell: "10_t2_s0" });
	const lib = join(home, "libero");
	mkdirSync(join(lib, "suite"));
	mkdirSync(join(base, "elsewhere", "deep"), { recursive: true });
	writeFileSync(join(lib, "suite", "real.md"), "a leaf\n");
	writeFileSync(join(base, "elsewhere", "secret.md"), "outside\n");
	// A link whose target goes through another link and then "..": the system follows `hop` before it goes up
	// and lands outside. Folded as text the target is global/secret.md, a name in the tier, there or not.
	symlinkSync(join(base, "elsewhere", "deep"), join(lib, "global", "hop"));
	symlinkSync("hop/../secret.md", join(lib, "global", "leak.md"));
	assert.equal(readFileSync(join(lib, "global", "leak.md"), "utf8"), "outside\n");
	assert.equal(canonicalPath(join(lib, "global", "leak.md"), run), join(base, "elsewhere", "secret.md"));
	for (const decoy of [false, true]) {
		if (decoy) writeFileSync(join(lib, "global", "secret.md"), "a file where the folded path points\n");
		for (const tool of FILE_TOOLS)
			assert.match(
				call(tool, join(lib, "global", "leak.md"))?.reason ?? "",
				// A write in a tier is refused by its own rule, wherever the link leads.
				tool === "write" || tool === "edit" ? /^writing to memory is denied/ : /^only the memory/,
				`${tool}, decoy ${decoy}`,
			);
	}
	// The same in the output dir, where the folded text is a file of the cell's own: a write or an edit through
	// the link would land outside.
	symlinkSync(join(base, "elsewhere", "deep"), join(run, "10_t2_s0_hop"));
	symlinkSync("10_t2_s0_hop/../10_t2_s0_x.md", join(run, "10_t2_s0_leak.md"));
	for (const tool of FILE_TOOLS)
		assert.equal(
			call(tool, "10_t2_s0_leak.md")?.reason,
			`only the memory and this cell's 10_t2_s0.* / 10_t2_s0_* files in ${run} are accessible: ${join(run, "10_t2_s0_leak.md")}`,
			tool,
		);
	// The other way round: folded as text the target leaves the library, the system stays in it.
	// (The target is written out: join() would fold "back/.." away.)
	symlinkSync(join(lib, "suite"), join(base, "elsewhere", "back"));
	symlinkSync(`${join(base, "elsewhere", "back")}/../suite/real.md`, join(lib, "global", "alias.md"));
	for (const tool of ["read", "ls", "grep", "find"])
		assert.equal(handed(tool, join(lib, "global", "alias.md")), join(lib, "suite", "real.md"), tool);
	// Every tool is handed the judged path: absolute, without links, however the planner wrote it.
	writeFileSync(join(run, "10_t2_s0.json"), "{}");
	symlinkSync("10_t2_s0.json", join(run, "10_t2_s0_same.json"));
	for (const tool of FILE_TOOLS) {
		assert.equal(handed(tool, "10_t2_s0_same.json"), join(run, "10_t2_s0.json"), tool);
		assert.equal(handed(tool, `@${join(run, "10_t2_s0.json")}`), join(run, "10_t2_s0.json"), tool);
	}
	assert.equal(handed("ls"), run);
	assert.equal(handed("write", "./10_t2_s0_new.md"), join(run, "10_t2_s0_new.md"));
	// A write through a dangling link is handed the place where it would land.
	symlinkSync("10_t2_s0_target.md", join(run, "10_t2_s0_link.md"));
	assert.equal(handed("write", "10_t2_s0_link.md"), join(run, "10_t2_s0_target.md"));
	// Where the system opens nothing the call is refused: a ".." behind a name that does not exist, a name
	// below a file (the text of both folds to a real file).
	symlinkSync("10_t2_s0_missing/../10_t2_s0.json", join(run, "10_t2_s0_bad.md"));
	symlinkSync("10_t2_s0_missing/deeper/../../10_t2_s0.json", join(run, "10_t2_s0_bad2.md"));
	symlinkSync("10_t2_s0.json/../10_t2_s0.json", join(run, "10_t2_s0_below.md"));
	symlinkSync("10_t2_s0.json/", join(run, "10_t2_s0_slash.md"));
	symlinkSync("10_t2_s0.json/.", join(run, "10_t2_s0_dot.md"));
	for (const name of ["bad", "bad2", "below", "slash", "dot"])
		for (const tool of FILE_TOOLS)
			assert.match(call(tool, `10_t2_s0_${name}.md`)?.reason ?? "", /does not resolve/, `${tool} ${name}`);
	assert.throws(() => canonicalPath("10_t2_s0_below.md", run), {
		code: "ENOTDIR",
		component: join(run, "10_t2_s0.json"),
	});
	// A link to nothing at all (macOS makes one, Linux does not).
	try {
		symlinkSync("", join(run, "10_t2_s0_empty.md"));
	} catch {}
	if (lstatSync(join(run, "10_t2_s0_empty.md"), { throwIfNoEntry: false }))
		assert.match(call("read", "10_t2_s0_empty.md")?.reason ?? "", /does not resolve/);
	// The tools would turn a no-break space into a plain one and open another name.
	symlinkSync("10_t2_s0_a\u00a0b.md", join(run, "10_t2_s0_nbsp.md"));
	for (const tool of FILE_TOOLS)
		assert.equal(
			call(tool, "10_t2_s0_nbsp.md")?.reason,
			`file access denied: ${join(run, "10_t2_s0_nbsp.md")} cannot be named`,
			tool,
		);
	// The space can be in a directory's name: the tools would turn it there too, and open below a twin that
	// leads outside.
	mkdirSync(join(lib, "global", "a\u00a0b"));
	writeFileSync(join(lib, "global", "a\u00a0b", "x.md"), "x");
	symlinkSync(join(base, "elsewhere"), join(lib, "global", "a b"));
	writeFileSync(join(base, "elsewhere", "x.md"), "outside\n");
	symlinkSync("a\u00a0b", join(lib, "global", "spaced"));
	for (const tool of ["read", "ls", "grep", "find"])
		assert.equal(
			call(tool, join(lib, "global", "spaced", "x.md"))?.reason,
			`file access denied: ${join(lib, "global", "spaced", "x.md")} cannot be named`,
			tool,
		);
	// Where "/" separates names a backslash is a character of a name, in a link's target too.
	if (sep === "/") {
		symlinkSync("10_t2_s0_a\\..\\b.md", join(run, "10_t2_s0_bs.md"));
		assert.equal(handed("write", "10_t2_s0_bs.md"), join(run, "10_t2_s0_a\\..\\b.md"));
	}
	// The argument is read as pi's tools read it: "~" and "~/" are the home directory, "~name" is a name, and
	// a file:// URL is its path.
	assert.ok(call("read", "~/nothing.md")?.reason.endsWith(`: ${resolve(run, join(homedir(), "nothing.md"))}`));
	assert.ok(call("read", "~nothing.md")?.reason.endsWith(`: ${join(run, "~nothing.md")}`));
	assert.equal(handed("read", pathToFileURL(join(lib, "suite", "real.md")).href), join(lib, "suite", "real.md"));
	// An error inside the hook is a refusal: a call whose input cannot be rewritten, read-only dirs that throw.
	const frozen = Object.freeze({ path: "10_t2_s0.json" });
	assert.match(call("write", undefined, frozen)?.reason ?? "", /^file access denied: /);
	const broken = await guarded({
		cell: "10_t2_s0",
		readable: () => {
			throw new Error("no state dir");
		},
	});
	assert.match(broken.call("read", join(broken.run, "10_t2_s0.json"))?.reason ?? "", /^file access denied: /);
});

test("the memory home, the output dir and a read-only dir are resolved however the session is told them", async () => {
	const real = await guarded({ cell: "10_t2_s0" });
	const states = join(real.base, "states");
	mkdirSync(states);
	const at = (name: string, ...rest: string[]) => join(real.base, name, ...rest);
	symlinkSync(real.home, at("memory-link"));
	symlinkSync(real.run, at("run-link"));
	symlinkSync(states, at("states-link"));
	const told = await guarded({
		cell: "10_t2_s0",
		home: at("memory-link"),
		output: at("run-link"),
		readable: () => ["", at("states-link")],
	});
	assert.equal(told.mem.configError(), undefined);
	assert.equal(told.mem.render("{{memory_dir}} {{output_dir}}"), `${join(real.home, "libero")} ${real.run}`);
	assert.equal(told.handed("read", at("memory-link", "libero", "MEMORY.md")), join(real.home, "libero", "MEMORY.md"));
	assert.equal(told.handed("write", at("run-link", "10_t2_s0.json")), join(real.run, "10_t2_s0.json"));
	assert.equal(told.handed("ls", at("states-link")), states);
	// The rules know the memory home as it resolves: another robot's memory, as the prompt spells it.
	assert.match(told.call("read", join(real.home, "robotwin"))?.reason ?? "", /^access to another robot's memory/);
	// The exploration archive is under the output dir as it resolves.
	const archive = at("run-link", "attempts", "10_t2_s0", "a1.json");
	assert.equal(told.handed("write", archive), join(real.run, "attempts", "10_t2_s0", "a1.json"));
	// A session without a cell names the places as they resolve too.
	const plain = await guarded({ home: at("memory-link") });
	assert.equal(plain.mem.render("{{memory_dir}}"), join(real.home, "libero"));
	// A read-only dir is read-only, and an empty entry among them is no directory (it would be the cwd).
	assert.match(told.call("write", join(states, "x.md"))?.reason ?? "", /^only the memory/);
	writeFileSync(join(told.run, "other.json"), "{}");
	assert.match(told.call("read", join(told.run, "other.json"))?.reason ?? "", /^only the memory/);
});

test("a refusal names the path as it was asked and does not tell what is there", async () => {
	const readable: string[] = [];
	const { base, home, run, call, handed } = await guarded({ cell: "10_t2_s0", readable: () => readable });
	const shared = join(base, "shared");
	const lib = join(home, "libero");
	mkdirSync(shared);
	mkdirSync(join(lib, "_internal", "merged"), { recursive: true });
	writeFileSync(join(lib, "_internal", "merged", "z.md"), "x");
	writeFileSync(join(lib, "global", "leaf.md"), "x");
	writeFileSync(join(home, "robotwin", "held.json"), "{}");
	// Outside the boundary a file, a pipe, a name below either and a name that is not there are refused alike.
	const outside: [string, string, RegExp][] = [
		[join(base, "secret.txt"), join(base, "nothing.txt"), /^only the memory/],
		[join(home, "robotwin", "held.json"), join(home, "robotwin", "nope.json"), /^access to another robot's memory/],
	];
	// So is a link there, whatever zone it leads to.
	symlinkSync(join(home, "robotwin", "held.json"), join(base, "zone"));
	outside.push([join(base, "zone"), join(base, "nothing.md"), /^only the memory/]);
	if (spawnSync("mkfifo", [join(base, "pipe")]).status === 0)
		outside.push([join(base, "pipe"), join(base, "nothing"), /^only the memory/]);
	for (const [there, gone, rule] of outside)
		for (const below of ["", "x"]) {
			const reason = call("read", join(there, below))?.reason ?? "";
			assert.match(reason, rule);
			assert.equal(reason.replaceAll(there, gone), call("read", join(gone, below))?.reason);
		}
	// A link is refused under the name it was asked by, whatever the rule: never under where it leads.
	const links: [string, string, string, RegExp][] = [
		["read", join(lib, "global", "out.md"), join(base, "secret.txt"), /^only the memory/],
		[
			"read",
			join(lib, "global", "other.md"),
			join(home, "robotwin", "held.json"),
			/^access to another robot's memory is denied/,
		],
		[
			"read",
			join(lib, "global", "inner.md"),
			join(lib, "_internal", "merged", "z.md"),
			/^reading this memory path is denied/,
		],
		[
			"write",
			join(run, "10_t2_s0_w.md"),
			join(lib, "global", "leaf.md"),
			/^writing to memory is denied in this mode/,
		],
		["grep", join(run, "10_t2_s0_root"), lib, /^search global\/, suite\/ or task_only\//],
	];
	for (const [tool, link, target, rule] of links) {
		symlinkSync(target, link);
		const reason = call(tool, link)?.reason ?? "";
		assert.match(reason, rule);
		assert.ok(reason.endsWith(`: ${link}`), reason);
		assert.ok(!reason.includes(target), reason);
	}
	// Behind a link too, what is at the refused place does not show: a file, a name below a file and a name
	// that is not there give the same refusal.
	const behind = [join(base, "secret.txt"), join(base, "secret.txt", "below"), join(base, "nothing", "x")].map(
		(target, i) => {
			const link = join(lib, "global", `behind${i}.md`);
			symlinkSync(target, link);
			return (call("read", link)?.reason ?? "").replace(link, "<link>");
		},
	);
	assert.match(behind[0], /^only the memory.*: <link>$/s);
	assert.deepEqual(behind, [behind[0], behind[0], behind[0]]);
	// That holds for every tool, and whatever stopped the walk there: a name below a file, more links than are
	// followed, a name that is not there. (A write in a tier is refused by its own rule first. A row of links,
	// not a loop: a resolver without a limit still returns; loops are a test of their own.)
	for (let i = 1; i <= 41; i++) symlinkSync(i === 1 ? "secret.txt" : `row${i - 1}`, join(base, `row${i}`));
	for (const tool of FILE_TOOLS) {
		const told = [join(base, "secret.txt", "below"), join(base, "row41"), join(base, "nothing", "x")].map(
			(target, i) => {
				const link = join(lib, "global", `stop-${tool}-${i}.md`);
				symlinkSync(target, link);
				return (call(tool, link)?.reason ?? "").replace(link, "<link>");
			},
		);
		const rule = tool === "write" || tool === "edit" ? /^writing to memory is denied/ : /^only the memory/;
		assert.match(told[0], new RegExp(`${rule.source}.*: <link>$`, "s"), tool);
		assert.deepEqual(told, [told[0], told[0], told[0]], tool);
	}
	// Where the place as asked is allowed and the walk stops in a refused place, that place's rule answers, for
	// the tool asked, under the name asked (here written relative to the cwd).
	symlinkSync(join(lib, "global", "leaf.md", "below"), join(run, "10_t2_s0_below.md"));
	assert.equal(
		call("write", "10_t2_s0_below.md")?.reason,
		`writing to memory is denied in this mode: ${join(run, "10_t2_s0_below.md")}`,
	);
	symlinkSync(join(home, "robotwin", "held.json", "x"), join(run, "10_t2_s0_other"));
	assert.equal(
		call("grep", "10_t2_s0_other")?.reason,
		`access to another robot's memory is denied: ${join(run, "10_t2_s0_other")}`,
	);
	// A link that itself sits in a refused place is answered by that place's rule, wherever outside it leads
	// (the guard judges what opens: one that leads into the library is read there).
	const fromRefused = [join(base, "secret.txt", "below"), join(base, "nothing", "x"), join(base, "secret.txt")];
	const answers = fromRefused.map((target, i) => {
		const link = join(home, "robotwin", `link${i}.md`);
		symlinkSync(target, link);
		return (call("read", link)?.reason ?? "").replace(link, "<link>");
	});
	assert.match(answers[0], /^access to another robot's memory is denied: <link>$/s);
	assert.deepEqual(answers, [answers[0], answers[0], answers[0]]);
	symlinkSync(join(lib, "global", "leaf.md"), join(home, "robotwin", "home.md"));
	assert.equal(handed("read", join(home, "robotwin", "home.md")), join(lib, "global", "leaf.md"));
	// More links than are followed, in an allowed place: the answer says so, under the name asked (a row, not
	// a loop: a resolver without a limit still returns).
	writeFileSync(join(run, "10_t2_s0_c0"), "x");
	for (let i = 1; i <= 41; i++) symlinkSync(`10_t2_s0_c${i - 1}`, join(run, `10_t2_s0_c${i}`));
	assert.equal(
		call("read", "10_t2_s0_c41")?.reason,
		`file access denied: too many symbolic links: ${join(run, "10_t2_s0_c41")}`,
	);
	// Nor does a refused place tell that it is a link to something inside that opens nothing.
	symlinkSync(join(lib, "global", "leaf.md", "below"), join(base, "into"));
	assert.equal(
		call("read", join(base, "into"))?.reason,
		call("read", join(base, "nothing"))?.reason?.replace(join(base, "nothing"), join(base, "into")),
	);
	// The rules come before anything is looked at: a device, or a name with a no-break space, outside the
	// boundary is refused like any other path there.
	if (existsSync("/dev/null")) assert.match(call("read", "/dev/null")?.reason ?? "", /^only the memory/);
	symlinkSync("a\u00a0b", join(base, "nbsp"));
	assert.match(call("read", join(base, "nbsp"))?.reason ?? "", /^only the memory/);
	// A read-only dir that does not resolve (here a name below a file) is not readable; the others still are,
	// and so is the memory.
	const nowhere = join(base, "secret.txt", "states");
	readable.push(nowhere, shared);
	assert.equal(handed("ls", shared), shared);
	assert.equal(handed("read", join(lib, "MEMORY.md")), join(lib, "MEMORY.md"));
	assert.match(call("ls", nowhere)?.reason ?? "", /^only the memory/);
});

test("what the system answers differently on a second look is refused without its own words", async () => {
	// The walk found the file; the system's realpath of it then fails (a race with someone else's change). The
	// answer is the walk's, not the system's message, which names where it looked.
	const { run, call } = await guarded({ cell: "10_t2_s0" });
	writeFileSync(join(run, "10_t2_s0.json"), "{}");
	const fs = realpathSync as unknown as { native: (path: PathLike) => string };
	const native = fs.native;
	fs.native = (path) => {
		throw Object.assign(new Error(`ENOENT: no such file or directory, realpath '${path}'`), { code: "ENOENT" });
	};
	try {
		assert.equal(
			call("read", "10_t2_s0.json")?.reason,
			`file access denied: path does not resolve: ${join(run, "10_t2_s0.json")}`,
		);
		// The same where the walk ends at a name that is not there (the realpath of the directory before it).
		assert.equal(
			call("read", "10_t2_s0_new.json")?.reason,
			`file access denied: path does not resolve: ${join(run, "10_t2_s0_new.json")}`,
		);
	} finally {
		fs.native = native;
	}
	// A link that is gone when its target is read.
	symlinkSync("10_t2_s0.json", join(run, "10_t2_s0_link.json"));
	const readlink = fsModule.readlinkSync;
	fsModule.readlinkSync = ((path: PathLike) => {
		throw Object.assign(new Error(`ENOENT: no such file or directory, readlink '${path}'`), { code: "ENOENT" });
	}) as typeof fsModule.readlinkSync;
	syncBuiltinESMExports();
	try {
		assert.equal(
			call("read", "10_t2_s0_link.json")?.reason,
			`file access denied: path does not resolve: ${join(run, "10_t2_s0_link.json")}`,
		);
	} finally {
		fsModule.readlinkSync = readlink;
		syncBuiltinESMExports();
	}
});

test("a later session keeps nothing of the one before", async () => {
	const s = await guarded({ cell: "10_t2_s0", explore: false, profile: "local" });
	assert.equal(s.mem.render("{{memory_dir}}"), join(s.home, "libero"));
	// A home that cannot even be read as a path: nothing of the first session is named or open.
	await s.again("file://elsewhere/memory");
	assert.match(s.mem.configError() ?? "", /^File URL host must be [^()]*$/);
	assert.equal(s.mem.render("{{memory_dir}}"), "");
	assert.match(s.call("read", join(s.home, "libero", "MEMORY.md"))?.reason ?? "", /^file access is disabled/);
	// A start refused before its places matter (a bad cell tag) names its own places, not the earlier ones.
	const b = await guarded({ cell: "10_t2_s0", explore: false, profile: "local" });
	const other = join(b.base, "other-memory");
	mkdirSync(join(other, "libero"), { recursive: true });
	const tagged = await guarded({ cell: "bad tag", explore: false, profile: "local", home: other });
	assert.match(tagged.mem.configError() ?? "", /^invalid memory cell tag/);
	assert.equal(tagged.mem.render("{{memory_dir}}"), join(other, "libero"));
	await b.again(other);
	assert.equal(b.mem.render("{{memory_dir}}"), join(other, "libero"));
});

test("what exists is named as the system stores it", async () => {
	const { base, home, run, call, handed } = await guarded({ cell: "10_t2_s0" });
	const lib = join(home, "libero");
	writeFileSync(join(lib, "global", "leaf.md"), "a leaf\n");
	// On a volume that does not tell case apart, another spelling is the same file: the tool is handed the
	// stored name and the rules judge that one, for a file that is there and for one to be made. On other
	// volumes the other spelling is another name, and a missing one.
	if (existsSync(join(lib, "GLOBAL", "LEAF.md"))) {
		assert.equal(handed("read", join(lib, "GLOBAL", "LEAF.md")), join(lib, "global", "leaf.md"));
		assert.equal(handed("write", join(base, "RUN", "10_t2_s0_new.md")), join(run, "10_t2_s0_new.md"));
	} else {
		assert.match(call("read", join(lib, "GLOBAL", "LEAF.md"))?.reason ?? "", /reading this memory path is denied/);
		assert.match(call("write", join(base, "RUN", "10_t2_s0_new.md"))?.reason ?? "", /only the memory/);
	}
});

test("a component that cannot be looked at is no answer: the call is refused, not judged by its text", async (t) => {
	const readable: string[] = [];
	const { base, home, run, call } = await guarded({ cell: "10_t2_s0", readable: () => readable });
	const lib = join(home, "libero");
	// The text names a place in a tier, where a search and a read are allowed.
	const locked = join(lib, "global", "locked");
	mkdirSync(join(locked, "sub"), { recursive: true });
	if (!lockOut(locked)) return t.skip("this user looks into any directory");
	t.after(() => chmodSync(locked, 0o700));
	// A link in an allowed place whose walk stops at such a directory outside is refused as outside.
	const away = join(base, "away");
	mkdirSync(join(away, "in"), { recursive: true });
	symlinkSync(join(away, "in", "x.md"), join(lib, "global", "to-away.md"));
	chmodSync(away, 0o000);
	try {
		assert.equal(
			call("read", join(lib, "global", "to-away.md"))?.reason,
			`only the memory and this cell's 10_t2_s0.* / 10_t2_s0_* files in ${run} are accessible: ${join(lib, "global", "to-away.md")}`,
		);
	} finally {
		chmodSync(away, 0o700);
	}
	// A read-only dir behind such a directory is not readable, and is refused like any place outside.
	const closed = join(base, "closed");
	mkdirSync(join(closed, "states"), { recursive: true });
	readable.push(join(closed, "states"));
	chmodSync(closed, 0o000);
	try {
		assert.match(call("ls", join(closed, "states"))?.reason ?? "", /^only the memory/);
	} finally {
		chmodSync(closed, 0o700);
	}
	try {
		assert.throws(() => canonicalPath(join(locked, "sub", "x.md"), run), { code: "EACCES" });
		assert.match(call("grep", join(locked, "sub"))?.reason ?? "", /^file access denied: path does not resolve: /);
		assert.match(
			call("read", join(locked, "sub", "x.md"))?.reason ?? "",
			/^file access denied: path does not resolve: /,
		);
	} finally {
		chmodSync(locked, 0o700);
	}
});

test("a memory home that cannot be reached stops a session that uses memory, and no other", async (t) => {
	const base = realpathSync.native(mkdtempSync(join(tmpdir(), "guard-home-")));
	const locked = join(base, "locked");
	const home = join(locked, "deep", "memory");
	mkdirSync(home, { recursive: true });
	if (!lockOut(locked)) return t.skip("this user looks into any directory");
	t.after(() => chmodSync(locked, 0o700));
	try {
		// (The local profile: a resolver that guessed the home would otherwise go on to sync the published one.)
		const uses = await guarded({ cell: "10_t2_s0", home, explore: false, profile: "local" });
		assert.equal(uses.mem.configError(), `path does not resolve: ${home} (EACCES at ${join(locked, "deep")})`);
		assert.match(uses.call("read", join(uses.run, "10_t2_s0.json"))?.reason ?? "", /^file access is disabled/);
		// Exploration uses memory too.
		assert.match((await guarded({ cell: "10_t2_s0", home })).mem.configError() ?? "", / \(EACCES at /);
		const off = await guarded({ cell: "10_t2_s0", home, explore: false, profile: "off", optional: true });
		assert.equal(off.mem.configError(), undefined);
		assert.equal(off.mem.render("{{memory_dir}} {{output_dir}}"), `${join(home, "libero")} ${off.run}`);
		// The output dir does not wait for the home: a session that goes on names where it leads.
		mkdirSync(join(base, "out"));
		symlinkSync(join(base, "out"), join(base, "out-link"));
		const linked = await guarded({
			cell: "10_t2_s0",
			home,
			explore: false,
			profile: "off",
			optional: true,
			output: join(base, "out-link"),
		});
		assert.equal(linked.mem.render("{{output_dir}}"), join(base, "out"));
		const noCell = await guarded({ home });
		assert.equal(noCell.mem.configError(), undefined);
	} finally {
		chmodSync(locked, 0o700);
	}
});

test("a session resolves the places its own profile reads", async (t) => {
	const offline = process.env.HF_HUB_OFFLINE;
	process.env.HF_HUB_OFFLINE = "1";
	t.after(() => {
		if (offline === undefined) delete process.env.HF_HUB_OFFLINE;
		else process.env.HF_HUB_OFFLINE = offline;
	});
	const base = realpathSync.native(mkdtempSync(join(tmpdir(), "guard-profile-")));
	writeFileSync(join(base, "file"), "x");
	mkdirSync(join(base, "corpus"));
	writeFileSync(join(base, "corpus", "MEMORY.md"), "# local\n");
	// The published profile reads <home>/<robot>, never dirs.memory: it does not look there ...
	const looked: string[] = [];
	const fs = fsModule as unknown as { lstatSync: (...args: unknown[]) => unknown };
	const lstat = fs.lstatSync;
	fs.lstatSync = (path, ...rest) => {
		looked.push(String(path));
		return lstat(path, ...rest);
	};
	syncBuiltinESMExports();
	let hf: Awaited<ReturnType<typeof guarded>>;
	try {
		hf = await guarded({ cell: "10_t2_s0", explore: false, dir: join(base, "corpus") });
	} finally {
		fs.lstatSync = lstat;
		syncBuiltinESMExports();
	}
	assert.deepEqual(
		looked.filter((path) => path.startsWith(join(base, "corpus"))),
		[],
	);
	const index = join(hf.home, "libero", "MEMORY.md");
	assert.equal(hf.mem.configError(), undefined);
	assert.equal(hf.handed("read", index), index);
	assert.match(hf.call("read", join(base, "corpus", "MEMORY.md"))?.reason ?? "", /^only the memory/);
	// ... and one that does not resolve is not its concern.
	const other = await guarded({ cell: "10_t2_s0", explore: false, dir: join(base, "file", "corpus") });
	assert.equal(other.mem.configError(), undefined);
	assert.equal(other.handed("read", join(other.home, "libero", "MEMORY.md")), join(other.home, "libero", "MEMORY.md"));
	// It is stopped by its own root. The message names the system's error and the component it came from.
	const below = await guarded({
		cell: "10_t2_s0",
		explore: false,
		home: join(base, "file"),
		dir: join(base, "corpus"),
	});
	assert.equal(
		below.mem.configError(),
		`path does not resolve: ${join(base, "file", "libero")} (ENOTDIR at ${join(base, "file")})`,
	);
	// An error that is not the walk's own (Node's, for a file URL with a host) is reported as it stands.
	const url = await guarded({ cell: "10_t2_s0", explore: false, profile: "local", home: "file://elsewhere/memory" });
	assert.match(url.mem.configError() ?? "", /^File URL host must be [^()]*$/);
	// A session without a cell, and the /memory command on its own, name dirs.memory, the local corpus, whatever
	// profile a cell would have had.
	for (const explore of [true, false])
		assert.equal(
			(await guarded({ dir: join(base, "corpus"), explore })).mem.render("{{memory_dir}}"),
			join(base, "corpus"),
		);
	const plain = await guarded({ dir: join(base, "corpus") });
	await plain.command("memory", "validate");
	assert.equal(plain.mem.render("{{memory_dir}}"), join(base, "corpus"));
	// /memory merge resolves the directory it is given, and reports one that does not resolve.
	writeFileSync(join(plain.run, "a-file"), "x");
	await assert.rejects(async () => plain.command("memory", "merge 10_t2_s0 a-file/runs"), {
		message: `path does not resolve: ${join(plain.run, "a-file", "runs")}`,
	});
	// The local profile reads dirs.memory, and is stopped by that.
	const local = await guarded({
		cell: "10_t2_s0",
		explore: false,
		profile: "local",
		dir: join(base, "file", "corpus"),
	});
	assert.equal(
		local.mem.configError(),
		`path does not resolve: ${join(base, "file", "corpus")} (ENOTDIR at ${join(base, "file")})`,
	);
});

test("the /memory command on its own looks its corpus up again and reports one that does not resolve", async () => {
	const base = realpathSync.native(mkdtempSync(join(tmpdir(), "guard-command-")));
	writeFileSync(join(base, "file"), "x");
	const alone = await guarded({ home: join(base, "file", "memory") });
	assert.equal(alone.mem.configError(), undefined);
	await assert.rejects(async () => alone.command("memory", "validate"), {
		message: `path does not resolve: ${join(base, "file", "memory")}`,
	});
});

test("the published corpus is fetched by the name of the robot's directory, however the volume spells it", async (t) => {
	// A stand-in for the hub: the files of two robots' corpora. It records which corpus was asked for.
	const asked: string[] = [];
	const server = createServer((req, res) => {
		if (req.url?.startsWith("/api/datasets/")) {
			const siblings = ["libero/MEMORY.md", "other/MEMORY.md"].map((rfilename) => ({ rfilename, blobId: "none" }));
			res.end(JSON.stringify({ sha: "abc", siblings }));
		} else {
			asked.push(decodeURIComponent(req.url?.split("/resolve/abc/")[1] ?? ""));
			res.end("# published\n");
		}
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const names = [
		"HF_ENDPOINT",
		"HF_HUB_OFFLINE",
		"HF_TOKEN",
		"PI_EMBODIED_MEMORY_REVISION",
		"PI_EMBODIED_MEMORY_REPO",
	];
	const saved = names.map((name) => [name, process.env[name]] as const);
	t.after(() => {
		server.close();
		for (const [name, value] of saved)
			if (value === undefined) delete process.env[name];
			else process.env[name] = value;
	});
	for (const name of names) delete process.env[name];
	process.env.HF_ENDPOINT = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	const base = realpathSync.native(mkdtempSync(join(tmpdir(), "guard-sync-")));
	let homes = 0;
	const session = async (layout: (home: string) => void) => {
		const home = join(base, `memory${homes++}`);
		mkdirSync(home);
		layout(home);
		asked.length = 0;
		const s = await guarded({ cell: "10_t2_s0", home, explore: false });
		assert.equal(s.mem.configError(), undefined);
		return { ...s, fetched: [...asked] };
	};
	// The robot's directory: its own corpus, into it.
	const plain = await session((home) => mkdirSync(join(home, "libero")));
	assert.deepEqual(plain.fetched, ["libero/MEMORY.md"]);
	assert.equal(readFileSync(join(plain.home, "libero", "MEMORY.md"), "utf8"), "# published\n");
	// A link to a directory of another name goes by that name, as it always did: nothing of this robot's
	// corpus is written into it (nor, with a pinned revision, anything of it removed).
	const linked = await session((home) => {
		mkdirSync(join(home, "other"));
		symlinkSync(join(home, "other"), join(home, "libero"));
	});
	assert.deepEqual(linked.fetched, ["other/MEMORY.md"]);
	// So does a link to another spelling of the robot's name: where the volume tells case apart that is
	// another directory, and the robot's corpus does not go beside it, nor, with a pinned revision, is anything
	// there removed.
	const elsewhere = join(base, "elsewhere");
	mkdirSync(join(elsewhere, "LIBERO"), { recursive: true });
	const folds = existsSync(join(elsewhere, "libero"));
	if (!folds) {
		mkdirSync(join(elsewhere, "libero"));
		writeFileSync(join(elsewhere, "libero", "keep.md"), "someone else's\n");
	}
	process.env.PI_EMBODIED_MEMORY_REVISION = "abc";
	const spelled = await session((home) => symlinkSync(join(elsewhere, "LIBERO"), join(home, "libero")));
	delete process.env.PI_EMBODIED_MEMORY_REVISION;
	assert.deepEqual(spelled.fetched, folds ? ["libero/MEMORY.md"] : []);
	if (!folds) assert.equal(readFileSync(join(elsewhere, "libero", "keep.md"), "utf8"), "someone else's\n");
	// A dangling link to such a spelling: nothing is made under the robot's spelling beside it.
	const dangling = await session((home) => symlinkSync(join(base, "nowhere", "LIBERO"), join(home, "libero")));
	assert.deepEqual(dangling.fetched, []);
	assert.equal(existsSync(join(base, "nowhere")), false);
	// Where the volume does not tell case apart, a directory stored under another spelling of the robot's name
	// is the robot's, and gets the robot's corpus. The guard's root is the stored name. (Asked before the sync,
	// which makes the robot's directory where the volume tells case apart.)
	mkdirSync(join(base, "Probe"));
	if (!existsSync(join(base, "probe"))) return t.skip("this volume tells case apart");
	const cased = await session((home) => mkdirSync(join(home, "Libero")));
	assert.deepEqual(cased.fetched, ["libero/MEMORY.md"]);
	assert.equal(readFileSync(join(cased.home, "Libero", "MEMORY.md"), "utf8"), "# published\n");
	assert.equal(cased.handed("read", join(cased.home, "libero", "MEMORY.md")), join(cased.home, "Libero", "MEMORY.md"));
	// /memory sync on its own asks for the robot's corpus too.
	const alone = await session((home) => mkdirSync(join(home, "Libero")));
	rmSync(join(alone.home, "Libero", "MEMORY.md"));
	asked.length = 0;
	await alone.command("memory", "sync");
	assert.deepEqual(asked, ["libero/MEMORY.md"]);
});

test("a pipe or a device is not a file: no tool is sent to wait on it", async (t) => {
	const { home, run, call, handed } = await guarded({ cell: "10_t2_s0", readable: () => ["/dev"] });
	const tier = join(home, "libero", "global");
	const pipe = join(run, "10_t2_s0_pipe");
	if (spawnSync("mkfifo", [pipe]).status !== 0 || spawnSync("mkfifo", [join(tier, "pipe.md")]).status !== 0)
		return t.skip("no mkfifo here");
	for (const tool of FILE_TOOLS)
		assert.equal(call(tool, pipe)?.reason, `file access denied: ${pipe} is not a file or a directory`, tool);
	// Through a link it is refused under the link's name.
	symlinkSync("10_t2_s0_pipe", join(run, "10_t2_s0_to_pipe"));
	assert.equal(
		call("read", "10_t2_s0_to_pipe")?.reason,
		`file access denied: ${join(run, "10_t2_s0_to_pipe")} is not a file or a directory`,
	);
	assert.match(call("read", "/dev/null")?.reason ?? "", /is not a file or a directory$/);
	// The directory that holds one is still listed and searched.
	assert.equal(handed("ls", tier), tier);
	assert.equal(handed("grep", tier), tier);
	assert.equal(handed("ls", run), run);
});

test("a missing file is the tool's to report, unless read would look for it under another spelling", async (t) => {
	const { base, home, call, handed } = await guarded({ cell: "10_t2_s0" });
	if (respelled(base)) return t.skip("the temp dir's own path has another spelling");
	const lib = join(home, "libero");
	// An ordinary name that is not there: the tool gets the call and answers.
	assert.equal(handed("read", join(lib, "global", "nothing.md")), join(lib, "global", "nothing.md"));
	// pi's read looks for a missing file under other spellings of the whole path (resolveReadPath). A link
	// stored under such a spelling would be read by asking for the plain one, which the guard judged.
	symlinkSync(join(base, "secret.txt"), join(lib, "global", "it\u2019s.md"));
	symlinkSync(join(base, "secret.txt"), join(lib, "global", "shot 9.41\u202fAM.md"));
	for (const name of ["it's.md", "shot 9.41 AM.md", "shot 9.41 pm.md", "caf\u00e9.md"])
		assert.match(call("read", join(lib, "global", name))?.reason ?? "", /^no such file: /, name);
	// Through a link the refusal names the link.
	symlinkSync("it's gone.md", join(lib, "global", "gone.md"));
	assert.equal(
		call("read", join(lib, "global", "gone.md"))?.reason,
		`no such file: ${join(lib, "global", "gone.md")}`,
	);
	// The rules come first: outside the boundary such a name is refused like any other, there or not.
	assert.match(call("read", join(base, "it's.md"))?.reason ?? "", /only the memory/);
	// Only `read` looks elsewhere: a listing, a search or a write of such a name stays with the tool.
	for (const tool of ["ls", "grep", "find"])
		assert.equal(handed(tool, join(lib, "global", "it's.md")), join(lib, "global", "it's.md"), tool);
	// The spelling can be in a directory's name: an output dir with an apostrophe and a twin beside it.
	const out = join(base, "bob's out");
	const twin = join(base, "bob\u2019s out");
	mkdirSync(out);
	mkdirSync(twin);
	writeFileSync(join(twin, "10_t2_s0_notes.md"), "a file of the twin directory\n");
	const there = await guarded({ cell: "10_t2_s0", output: out });
	assert.match(there.call("read", join(twin, "10_t2_s0_notes.md"))?.reason ?? "", /only the memory/);
	assert.match(there.call("read", join(out, "10_t2_s0_notes.md"))?.reason ?? "", /^no such file: /);
	assert.equal(there.handed("write", join(out, "10_t2_s0_notes.md")), join(out, "10_t2_s0_notes.md"));
	// The rule looks at the judged path, which is what read is handed: through a plainly named link to that
	// directory the argument has no apostrophe, and read would still try the twin.
	symlinkSync(out, join(base, "plain"));
	const via = await guarded({ cell: "10_t2_s0", output: join(base, "plain") });
	assert.match(via.call("read", join(base, "plain", "10_t2_s0_notes.md"))?.reason ?? "", /^no such file: /);
});

test("the spellings pi's read tries for a missing file are the ones the guard refuses", async (t) => {
	// The guard knows them by a rule of its own. This compares the rule with pi's read itself: every path it
	// probes is recorded, and a missing name is refused exactly when read would probe another spelling.
	const { run, handed } = await guarded({ cell: "10_t2_s0" });
	const { resolveReadPathAsync } = await import(PI_PATH_UTILS.href);
	const probed: string[] = [];
	const access = fsPromises.access;
	fsPromises.access = ((path: PathLike, mode?: number) => {
		probed.push(String(path));
		return access(path, mode);
	}) as typeof fsPromises.access;
	syncBuiltinESMExports();
	t.after(() => {
		fsPromises.access = access;
		syncBuiltinESMExports();
	});
	let seed = 20261113;
	const rnd = () => {
		seed = (seed * 1103515245 + 12345) & 0x7fffffff;
		return seed / 0x7fffffff;
	};
	const parts = [
		"a",
		"B",
		"7",
		"-",
		"_",
		".",
		" ",
		"'",
		"\u2019",
		"\u02bc",
		" AM.",
		" PM.",
		" am.",
		"AM.",
		" A M.",
		" AM",
		" pm",
		"\u202fPM.",
	];
	parts.push(
		"\u00e9",
		"e\u0301",
		"\u00c5",
		"A\u030a",
		"\u4e2d",
		"\u{1f600}",
		"\ufb01",
		"\u00df",
		"I",
		"\u0131",
		"\u212b",
	);
	const names = ["plain.md", "it's.md", "shot 9.41 AM.md", "caf\u00e9.md", "cafe\u0301.md", "a - b.md", "a...b.md"];
	// Every printable ASCII character once, so that a spelling pi adds for any of them changes a name.
	for (let c = 0x20; c < 0x7f; c++) {
		const ch = String.fromCharCode(c);
		if (ch !== "/") names.push(`x${ch}y.md`, `x${ch}${ch}${ch}y.md`, `xy${ch}`);
	}
	for (let i = 0; i < 400; i++) {
		let name = "";
		for (let k = 1 + Math.floor(rnd() * 6); k > 0; k--) name += parts[Math.floor(rnd() * parts.length)];
		names.push(`${name}.md`);
	}
	const seen = { refused: 0, passed: 0 };
	for (const name of names) {
		const asked = join(run, `10_t2_s0_${name}`);
		const answer = String(handed("read", asked));
		const refused = answer.startsWith("no such file: ");
		// What the tool would be given had the guard passed the call on: the judged path.
		probed.length = 0;
		await resolveReadPathAsync(canonicalPath(asked, run), run);
		assert.equal(refused, new Set(probed).size > 1, JSON.stringify([name, answer, probed]));
		seen[refused ? "refused" : "passed"]++;
	}
	assert.ok(seen.refused > 100 && (seen.passed > 20 || respelled(run)), JSON.stringify(seen));
});

test("the guard reads an argument as pi's tools do, and they leave the path it hands them as it is", async () => {
	// written() and the unicode spaces are the guard's copies of what pi's tools do to a path. This asks pi.
	const { base, run, handed } = await guarded({ cell: "10_t2_s0" });
	const { resolveToCwd } = await import(PI_PATH_UTILS.href);
	// The argument, in the forms pi knows and in some it does not: the guard resolves what the tool would open.
	// A ".." written in the argument is folded as text, as the tools fold it, also behind a link.
	mkdirSync(join(base, "elsewhere", "deep"), { recursive: true });
	symlinkSync(join(base, "elsewhere", "deep"), join(run, "hop"));
	const url = pathToFileURL(join(run, "x.md")).href;
	const forms = ["", ".", "x.md", "./x.md", "x/", "../x.md", "hop/../x.md", "~", "~/x.md", "~user/x.md", "@x.md"];
	forms.push("@@x.md", "@~/x.md", "$HOME/x.md", url, `@${url}`, url.replace("file:", "FILE:"));
	forms.push("a\u00a0b.md", "a\u3000b.md", " x.md", "x.md ", "\tx.md", homedir(), "~\\x.md");
	const where = (input: string) => {
		try {
			return canonicalPath(input, run);
		} catch (e) {
			return `threw ${(e as Error).message}`;
		}
	};
	const home = process.env.HOME;
	try {
		// Also with an empty home, where pi reads "~/x.md" as a name below the cwd.
		for (const value of [home, ""]) {
			if (value === undefined) continue;
			process.env.HOME = value;
			for (const input of forms)
				assert.equal(where(input), where(resolveToCwd(input, run)), `${JSON.stringify(input)}, HOME=${value}`);
		}
	} finally {
		if (home === undefined) delete process.env.HOME;
		else process.env.HOME = home;
	}
	// What the hook hands on, pi's own reading returns unchanged: for every character of the basic plane inside
	// a name (20 to a name), and for white space, controls, punctuation and symbols at its end, where a trim
	// would bite.
	let asked = 0;
	let passed = 0;
	const check = (name: string) => {
		asked++;
		const got = String(handed("write", join(run, `10_t2_s0_${name}`)));
		if (!got.startsWith(run)) return; // a refusal: nothing is handed on
		passed++;
		assert.equal(resolveToCwd(got, run), got, JSON.stringify(name));
	};
	const chars: string[] = [];
	for (let c = 1; c < 0x10000; c++) if (c !== 0x2f && (c < 0xd800 || c > 0xdfff)) chars.push(String.fromCharCode(c));
	for (let i = 0; i < chars.length; i += 20) check(`a${chars.slice(i, i + 20).join("")}b`);
	for (const c of chars) if (/[\s\p{Z}\p{Cc}\p{Cf}\p{P}\p{S}]/u.test(c)) check(`a${c}`);
	// Every name is handed on: one the volume refused to look up would leave its characters unasked.
	assert.equal(passed, asked);
	// A path of a megabyte below a name that is not there: walked in one pass (a resolver that joins the rest name
	// by name takes seconds for 40,000 names and minutes for 300,000).
	for (const n of [40_000, 300_000]) {
		const long = join(run, "10_t2_s0_fresh") + `${sep}a`.repeat(n);
		const start = performance.now();
		assert.equal(canonicalPath(long, run), long);
		const ms = performance.now() - start;
		assert.ok(ms < 3000, `${n} names: ${Math.round(ms)} ms`);
	}
});

test("pi's path rules are the ones the guard was written against", () => {
	// written(), the unicode spaces and respelled (../src/capabilities/memory/index.ts) are copies of rules in
	// pi's tools, and the tests above can only ask about the characters and forms they know. When this fails,
	// those rules changed: read the texts hashed here again, bring the guard's copies in line, and set the hash.
	const pi = (file: string) => readFileSync(new URL(`../../coding-agent/src/${file}`, import.meta.url), "utf8");
	const paths = pi("utils/paths.ts");
	const cut = (from: string, to: string) => {
		const at = paths.indexOf(from);
		assert.notEqual(at, -1, `utils/paths.ts no longer has ${JSON.stringify(from)}`);
		return paths.slice(at, paths.indexOf(to, at) + to.length);
	};
	const rules = [
		pi("core/tools/path-utils.ts"),
		cut("const UNICODE_SPACES = ", "\n"),
		cut("export function normalizePath(", "\n}\n"),
		cut("export function resolvePath(", "\n}\n"),
	];
	assert.equal(
		createHash("sha256").update(rules.join("")).digest("hex"),
		"8b0a9126834af35bd59c1c601531b1b4fd21791b0a521e987d49fddace372a94",
	);
	// The six tools must still take their path from those functions, unchanged, or the questions above go to
	// the wrong ones.
	const takes: Record<string, string> = {
		read: "const absolutePath = await resolveReadPathAsync(path, ctx?.cwd || cwd);",
		ls: 'const dirPath = resolveToCwd(path || ".", ctx?.cwd || cwd);',
		grep: 'const searchPath = resolveToCwd(searchDir || ".", ctx?.cwd || cwd);',
		find: 'const searchPath = resolveToCwd(searchDir || ".", ctx?.cwd || cwd);',
		write: "const absolutePath = resolveToCwd(path, ctx?.cwd || cwd);",
		edit: "const absolutePath = resolveToCwd(path, ctx?.cwd || cwd);",
	};
	for (const [tool, line] of Object.entries(takes)) {
		const text = pi(`core/tools/${tool}.ts`);
		assert.equal(text.split(line).length, 2, `${tool}.ts: ${line}`);
		const name = line.split(" ")[1];
		assert.equal(text.match(new RegExp(`\\b${name}\\s*=[^=>]`, "g"))?.length, 1, `${tool}.ts assigns ${name} once`);
	}
});

test("canonicalPath is the system's path: realpath for what exists, and where a write lands for what does not", () => {
	// Random trees of directories, files and links (relative, absolute, through another entry and "..", to
	// nothing), and paths into them. A link names only entries made before it, so none is a loop (those are a
	// test of their own), and none leads above the directory that holds the tree's root. For a path that does
	// not exist the system's answer is taken by writing through it as pi's write tool does: mkdir of the
	// parent, then the file.
	const base = realpathSync.native(mkdtempSync(join(tmpdir(), "guard-paths-")));
	let seed = 20261113;
	const rnd = () => {
		seed = (seed * 1103515245 + 12345) & 0x7fffffff;
		return seed / 0x7fffffff;
	};
	const pick = <T>(xs: T[]): T => xs[Math.floor(rnd() * xs.length)];
	const files = (dir: string, out: string[] = []): string[] => {
		for (const e of readdirSync(dir, { withFileTypes: true })) {
			if (e.isDirectory()) files(join(dir, e.name), out);
			else if (e.isFile()) out.push(join(dir, e.name));
		}
		return out;
	};
	/** Whether `path` names, at some component, a link whose target holds a "..": where text and system part. */
	const turns = (path: string): boolean => {
		try {
			if (readlinkSync(path).split("/").includes("..")) return true;
		} catch {}
		return path.length > base.length && turns(dirname(path));
	};
	const seen = { existing: 0, landed: 0, refused: 0, turned: 0 };
	for (let t = 0; t < 60; t++) {
		const tree = join(base, `t${t}`);
		const root = join(tree, "root");
		mkdirSync(root, { recursive: true });
		const dirs: string[] = []; // below the root: what a link may name outright
		const names: string[] = [];
		const held = new Map<string, string[]>([[root, []]]); // a directory -> the names made in it
		const any = () => `n${Math.floor(rnd() * 14)}`;
		for (let s = 0; s < 14; s++) {
			const parent = pick([root, ...dirs]);
			const at = join(parent, `n${s}`);
			const n = () => `n${Math.floor(rnd() * s)}`;
			// A name beside the link. Behind a link to a directory, a ".." is where text and system part.
			const beside = () => pick(held.get(parent)?.length ? (held.get(parent) as string[]) : ["fresh"]);
			const r = s < 2 ? 0 : rnd(); // two directories first: a link then has something to name
			if (r < 0.3) {
				mkdirSync(at);
				dirs.push(at);
				held.set(at, []);
			} else if (r < 0.4) writeFileSync(at, "x");
			else {
				const through = [`${beside()}/../${n()}`, `${beside()}/../${beside()}`, `${n()}/../${n()}/${n()}`];
				const plain = [pick(dirs), `${pick(dirs)}/`, n(), `${n()}/${n()}`, join(pick(dirs), "fresh"), "fresh"];
				// A ".." at the front leaves the link's own directory, which must not be the tree's root.
				if (parent !== root) through.push("..", `../${n()}`);
				else through.push("fresh/../other");
				symlinkSync(pick(rnd() < 0.6 ? through : plain), at);
			}
			held.get(parent)?.push(`n${s}`);
			names.push(at);
		}
		for (let k = 0; k < 14; k++) {
			let ask = pick(names);
			for (let extra = Math.floor(rnd() * 3); extra > 0; extra--) ask = join(ask, pick([any(), "fresh"]));
			let got: string | undefined;
			try {
				got = canonicalPath(ask, root);
			} catch {}
			// What exists is what the system can stat (on macOS realpath alone also answers for a ".." below a file).
			let real: string | undefined;
			try {
				statSync(ask);
				real = realpathSync.native(ask);
			} catch {}
			if (real !== undefined) {
				seen.existing++;
				if (turns(ask)) seen.turned++;
				assert.equal(got, real, `existing ${ask}`);
				continue;
			}
			const before = new Set(files(tree));
			try {
				mkdirSync(dirname(ask), { recursive: true });
				writeFileSync(ask, `marker ${t} ${k}`);
			} catch {}
			const landed = files(tree).filter((f) => !before.has(f));
			if (landed.length) {
				seen.landed++;
				if (turns(ask)) seen.turned++;
				assert.deepEqual(landed, [got], `a write through ${ask}`);
				rmSync(landed[0]);
			} else seen.refused++;
		}
	}
	// Nothing was made beside the trees' own directories.
	assert.deepEqual(readdirSync(base).sort(), Array.from({ length: 60 }, (_, t) => `t${t}`).sort());
	rmSync(base, { recursive: true, force: true });
	// The trees must have asked every kind of question for the comparison to mean anything.
	assert.ok(seen.existing > 100 && seen.landed > 150 && seen.refused > 100 && seen.turned > 60, JSON.stringify(seen));
});

test("the audit is a write_audit call: the model's fields plus the runtime's facts, as JSON in <output>/<cell>.json", async () => {
	const facts = { suite: "libero_goal", task_id: 3, seed: 7, regime: "strict_perception", libero_terminated: true };
	const { run, tools, mem } = await guarded({ cell: "goal_t3_s7", audit: { facts: () => facts } });
	assert.ok(mem.tools.includes("write_audit"));
	const tool = tools.get("write_audit");
	// The schema is what the model fills; the runtime's facts are not in it.
	assert.deepEqual(Object.keys(tool.parameters.properties).sort(), [
		"attempts",
		"memory_files_read",
		"pick_result",
		"strategy_notes",
		"terminated",
	]);
	assert.deepEqual(tool.parameters.required.sort(), ["memory_files_read", "strategy_notes", "terminated"]);
	const params = {
		terminated: true,
		strategy_notes: "segmented the bowl, picked it, placed it on the plate",
		memory_files_read: ["global/bowl.md"],
		pick_result: { bowl_pi0_pick_step3: { success: true } },
		// A runtime fact the model restates wrongly is overridden.
		seed: 0,
	};
	const r = await tool.execute("a", params);
	const path = join(run, "goal_t3_s7.json");
	assert.equal(r.details.path, path);
	const written = JSON.parse(readFileSync(path, "utf8"));
	assert.deepEqual(written, { ...params, ...facts });
	assert.equal(written.seed, 7);
	// Without the audit option nothing is registered and the tool list is the file tools.
	const plain = await guarded({ cell: "goal_t3_s7" });
	assert.equal(plain.tools.size, 0);
	assert.deepEqual(plain.mem.tools, ["read", "ls", "grep", "find", "write"]);
});

test("guard fails closed without a cell, output dir or session_start", async () => {
	for (const setup of [{ start: false, cell: "10_t2_s0" }, { output: "", cell: "10_t2_s0" }, { cell: undefined }]) {
		const { home, run, call } = await guarded(setup);
		for (const [tool, path] of [
			["read", join(home, "libero", "MEMORY.md")],
			["write", join(run, "10_t2_s0.json")],
			["ls", run],
			["find", home],
		] as const)
			assert.match(call(tool, path)?.reason ?? "", /file access is disabled/, `${JSON.stringify(setup)} ${tool}`);
	}
	// A bad cell tag is a start problem the robot reads (robot.ts), not a thrown handler.
	const bad = await guarded({ cell: "../x" });
	assert.match(bad.mem.configError() ?? "", /invalid memory cell tag/);
	assert.match(bad.call("read", join(bad.home, "libero", "MEMORY.md"))?.reason ?? "", /file access is disabled/);
});

test("a robot without a published corpus defaults to its local memory, empty or not; asking for it explicitly still needs one", async () => {
	const home = realpathSync(mkdtempSync(join(tmpdir(), "unpublished-")));
	const start = async (flags: Record<string, unknown>) => {
		const hooks = new Map<string, (event: unknown, ctx: unknown) => unknown>();
		const pi = {
			on: (name: string, fn: (event: unknown, ctx: unknown) => unknown) => hooks.set(name, fn),
			registerFlag: () => {},
			registerCommand: () => {},
			getFlag: (name: string) => flags[name],
		} as unknown as ExtensionAPI;
		const m = memory(pi, {
			robot: "maniskill",
			home: () => home,
			cell: () => ({ tag: "maniskill_PickCube-v1_s0", reference: "maniskill_PickCube-v1_s0" }),
			published: false,
		});
		const ctx = { cwd: home, hasUI: false, sessionManager: { getSessionDir: () => home, getBranch: () => [] } };
		await hooks.get("session_start")?.({}, ctx);
		return m;
	};
	// No HF sync is attempted (it would need the network) and the missing corpus is no error.
	const plain = await start({});
	assert.equal(plain.profile, "local");
	assert.equal(plain.configError(), undefined);
	// Bug 38: the missing corpus is recorded for robot.ts's fail-closed start, not thrown
	// (pi logs a throwing session_start handler and the episode would run memory-less).
	assert.match((await start({ "memory-profile": "local" })).configError() ?? "", /local memory corpus not found/);
});

test("a start refused after its guard was drawn up leaves the file tools closed", async () => {
	// The corpus check comes after the guard is drawn up: a refused start must not leave it in place.
	const s = await guarded({
		cell: "10_t2_s0",
		explore: false,
		profile: "local",
		dir: join(tmpdir(), "no-such-corpus"),
	});
	assert.match(s.mem.configError() ?? "", /local memory corpus not found/);
	assert.match(s.call("write", join(s.run, "10_t2_s0.json"))?.reason ?? "", /^file access is disabled/);
});

test("memory mounted without a cell option leaves the file tools to pi", async () => {
	const hooks = new Map<string, (event: unknown, ctx: unknown) => unknown>();
	const pi = {
		on: (name: string, fn: (event: unknown, ctx: unknown) => unknown) => hooks.set(name, fn),
		registerFlag: () => {},
		registerCommand: () => {},
		getFlag: () => undefined,
	} as unknown as ExtensionAPI;
	memory(pi, {});
	const ctx = { cwd: tmpdir(), hasUI: false, sessionManager: { getSessionDir: () => "", getBranch: () => [] } };
	await hooks.get("session_start")?.({}, ctx);
	for (const tool of FILE_TOOLS)
		assert.equal(
			hooks.get("tool_call")?.({ type: "tool_call", toolName: tool, toolCallId: "t", input: { path: "/" } }, ctx),
			undefined,
		);
});
