import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type MemoryOptions, memory } from "../src/capabilities/memory/index.ts";
import { useDeployment } from "./helpers/deployment.ts";

// A deployment of its own: a developer's config that names dirs.memory would move the library under test.
useDeployment({});

const reference = "maniskill_PickCube-v1_s0";
const tag = "maniskill_PickCube-v1_s2";
const section = readFileSync(new URL("../src/capabilities/memory/section.md", import.meta.url), "utf8").trim();
const required = readFileSync(new URL("../src/capabilities/memory/required.md", import.meta.url), "utf8").trim();

function fixture(t: { after: (fn: () => void) => void }, options: Partial<MemoryOptions> = {}) {
	const home = mkdtempSync(join(tmpdir(), "memory-section-"));
	t.after(() => rmSync(home, { recursive: true, force: true }));
	const root = join(home, "maniskill");
	const flags: Record<string, unknown> = {};
	const hooks = new Map<string, (event: unknown, ctx: ExtensionContext) => unknown>();
	const pi = {
		on: (name: string, handler: (event: unknown, ctx: ExtensionContext) => unknown) => hooks.set(name, handler),
		registerFlag: () => {},
		registerCommand: () => {},
		getFlag: (name: string) => flags[name],
	} as unknown as ExtensionAPI;
	const mem = memory(pi, {
		robot: "maniskill",
		home: () => home,
		cell: () => ({ tag, reference }),
		published: false,
		...options,
	});
	const ctx = {
		cwd: home,
		hasUI: false,
		sessionManager: { getSessionDir: () => home, getBranch: () => [] },
	} as unknown as ExtensionContext;
	return { root, flags, mem, start: () => hooks.get("session_start")?.({}, ctx) };
}

// "tiers a merge left empty": mergeMemory creates every tier before it looks at the inbox, so an
// exploration that published nothing leaves these directories and no file.
const MERGED_EMPTY = ["global", "suite", "task_only", "_internal/inbox", "_internal/merged", "_internal/conflicts"];
for (const layout of [
	"missing corpus",
	"empty corpus",
	"empty task_only",
	"unrelated reference",
	"tiers a merge left empty",
]) {
	test(`default memory section skips reads with ${layout}`, async (t) => {
		const f = fixture(t);
		if (layout !== "missing corpus") mkdirSync(f.root);
		if (layout === "empty task_only" || layout === "unrelated reference") mkdirSync(join(f.root, "task_only"));
		if (layout === "unrelated reference") writeFileSync(join(f.root, "task_only", `${tag}.json`), "{}");
		if (layout === "tiers a merge left empty")
			for (const tier of MERGED_EMPTY) mkdirSync(join(f.root, tier), { recursive: true });
		await f.start();
		assert.equal(f.mem.configError(), undefined);
		const text = f.mem.section() ?? "";
		assert.match(text, /No memory is available for this cell at session start/);
		assert.match(text, /Skip memory reads and searches/);
		assert.doesNotMatch(text, /MEMORY\.md|global\/|suite\/|task_only\/|Reading memory is a required step/);
		assert.match(text, /In your `finish` summary, say that none matched/);
		assert.deepEqual(f.mem.tools, ["read", "ls", "grep", "find", "write"]);
	});
}

for (const path of [
	"MEMORY.md",
	"global/leaf.md",
	"suite/leaf.md",
	`task_only/${reference}.json`,
	`task_only/${reference}_recipe.jsonl`,
]) {
	test(`default memory section is unchanged when the library holds only ${path}`, async (t) => {
		const f = fixture(t);
		const target = join(f.root, path);
		mkdirSync(dirname(target), { recursive: true });
		writeFileSync(target, "");
		await f.start();
		assert.equal(f.mem.configError(), undefined);
		assert.equal(f.mem.section(), f.mem.render(`${section}\n\n${required}`, { task: "" }));
	});
}

test("memory availability is captured at session start and refreshed for the next session", async (t) => {
	const f = fixture(t);
	await f.start();
	const absent = f.mem.section();
	assert.match(absent ?? "", /No memory is available/);
	mkdirSync(f.root);
	writeFileSync(join(f.root, "MEMORY.md"), "# Memory");
	assert.equal(f.mem.section(), absent);
	await f.start();
	assert.equal(f.mem.section(), f.mem.render(`${section}\n\n${required}`, { task: "" }));
	rmSync(f.root, { recursive: true });
	await f.start();
	assert.equal(f.mem.section(), absent);
});

test("a cell without a reference has no task pair to name: any task_only file is memory", async (t) => {
	// The real robots' cells (`reference: ""`): the reference pair would be `task_only/.json`.
	const f = fixture(t, { cell: () => ({ tag, reference: "" }) });
	mkdirSync(join(f.root, "task_only"), { recursive: true });
	await f.start();
	assert.match(f.mem.section() ?? "", /No memory is available for this cell at session start/);
	writeFileSync(join(f.root, "task_only", `${tag}.json`), "{}");
	await f.start();
	assert.equal(f.mem.section(), f.mem.render(`${section}\n\n${required}`, { task: "" }));
});

test("custom templates, memory off and sessions without a cell keep their existing behavior", async (t) => {
	const custom = fixture(t, { prompt: () => "Read {{memory_dir}}/custom.md." });
	await custom.start();
	assert.equal(custom.mem.section(), custom.mem.render(`Read {{memory_dir}}/custom.md.\n\n${required}`));
	const off = fixture(t, { optional: true });
	off.flags["memory-profile"] = "off";
	await off.start();
	assert.equal(off.mem.section(), undefined);
	assert.deepEqual(off.mem.tools, []);
	const noCell = fixture(t, { cell: () => undefined });
	await noCell.start();
	assert.equal(noCell.mem.section(), undefined);
});
