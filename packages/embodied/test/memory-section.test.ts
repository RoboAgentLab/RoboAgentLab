import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
	chmodSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
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
	// The real path: the module names its directories that way, and the notice below is compared in full.
	const home = realpathSync(mkdtempSync(join(tmpdir(), "memory-section-")));
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
	return { home, root, flags, mem, start: () => hooks.get("session_start")?.({}, ctx) };
}

/** The notice of a library that holds nothing, written out: the cell's own tag and the session's output dir. */
const notice = (f: { home: string }) =>
	`# Memory\nCell \`${tag}\`. No memory is available for this cell at session start. Skip memory reads and searches; proceed with the task. You write only to \`${f.home}/\`, and you read nothing else on disk. In your \`finish\` summary, say that none matched.`;

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
		assert.equal(f.mem.section(), notice(f));
		assert.deepEqual(f.mem.tools, ["read", "ls", "grep", "find", "write"]);
	});
}

test(
	"one subdirectory that cannot be listed does not empty its tier",
	{ skip: process.getuid?.() === 0 },
	async (t) => {
		// Root lists any directory, so the case cannot be built for it. The leaf sits in a sibling directory, so
		// that a walk cannot find it without going past the locked one; and the two names swap roles: whatever
		// order a directory is listed in, the locked one comes first in one of the two libraries.
		for (const locked of ["a", "b"]) {
			const f = fixture(t);
			const tier = join(f.root, "global");
			mkdirSync(tier, { recursive: true });
			for (const name of ["a", "b"]) {
				mkdirSync(join(tier, name), name === locked ? { mode: 0o000 } : {});
				if (name !== locked) writeFileSync(join(tier, name, "leaf.md"), "");
			}
			try {
				await f.start();
				assert.equal(f.mem.section(), f.mem.render(`${section}\n\n${required}`, { task: "" }), `locked ${locked}`);
			} finally {
				// Before the fixture's cleanup, which cannot remove a directory it cannot list.
				chmodSync(join(tier, locked), 0o700);
			}
		}
	},
);

test("a library behind a linked directory is not announced as empty, nor refused as missing", async (t) => {
	// The real path: the links below name it, and a relative one would be read from where each link lies.
	const outside = realpathSync(mkdtempSync(join(tmpdir(), "memory-section-outside-")));
	t.after(() => rmSync(outside, { recursive: true, force: true }));
	mkdirSync(join(outside, "notes"));
	mkdirSync(join(outside, "empty"));
	writeFileSync(join(outside, "notes", "leaf.md"), "");
	const layouts: [string, boolean, (root: string) => void][] = [
		[
			"a linked directory that holds a leaf, a level below the tier",
			true,
			(root) => {
				mkdirSync(join(root, "suite", "shared"), { recursive: true });
				symlinkSync(join(outside, "notes"), join(root, "suite", "shared", "linked"));
			},
		],
		[
			"a linked directory whose target goes through another link, then up",
			true,
			(root) => {
				mkdirSync(join(root, "suite"), { recursive: true });
				symlinkSync(join(outside, "empty"), join(root, "suite", "hop"));
				// Written out, because join() would fold "hop/.." away. The system reads it as <outside>/notes.
				symlinkSync("hop/../notes", join(root, "suite", "shared"));
			},
		],
		[
			"links that hold no leaf: one to an empty directory, one to nothing, one to a file",
			false,
			(root) => {
				mkdirSync(join(root, "global"), { recursive: true });
				symlinkSync(join(outside, "empty"), join(root, "global", "empty"));
				symlinkSync(join(outside, "gone.md"), join(root, "global", "gone.md"));
				symlinkSync(join(outside, "notes", "leaf.md"), join(root, "global", "leaf.md"));
			},
		],
	];
	for (const [layout, holds, build] of layouts) {
		// The section of a run that did not ask for memory.
		const f = fixture(t);
		build(f.root);
		await f.start();
		assert.equal(f.mem.configError(), undefined, layout);
		if (holds) assert.equal(f.mem.section(), f.mem.render(`${section}\n\n${required}`, { task: "" }), layout);
		else assert.equal(f.mem.section(), notice(f), layout);
		// The start check of a run that asked for local memory.
		const asked = fixture(t);
		asked.flags["memory-profile"] = "local";
		build(asked.root);
		await asked.start();
		if (holds) assert.equal(asked.mem.configError(), undefined, layout);
		else assert.match(asked.mem.configError() ?? "", /local memory corpus not found/, layout);
	}
});

test("links that lead back into a tier end the walk", (t) => {
	// In a child with a time limit: a walk that does not know where it has been does not return, and nothing
	// in this process could stop it. Five ways round, so that the system's limit on the links in one path does
	// not end such a walk within the time either.
	const home = realpathSync(mkdtempSync(join(tmpdir(), "memory-section-loops-")));
	t.after(() => rmSync(home, { recursive: true, force: true }));
	const tiers = ["no leaf", "a leaf past the links"].map((name) => {
		const tier = join(home, name);
		mkdirSync(join(tier, "a"), { recursive: true });
		symlinkSync(tier, join(tier, "a", "up"));
		symlinkSync(join(tier, "a"), join(tier, "again"));
		for (const back of ["b", "c", "d"]) symlinkSync(tier, join(tier, back));
		return tier;
	});
	mkdirSync(join(tiers[1], "z"));
	writeFileSync(join(tiers[1], "z", "leaf.md"), "");
	const corpus = new URL("../src/capabilities/memory/corpus.ts", import.meta.url).href;
	const walk =
		"import(process.argv[1]).then((m) => console.log(JSON.stringify(process.argv.slice(2).map((d) => m.hasFiles(d)))))";
	const r = spawnSync(process.execPath, ["--experimental-strip-types", "-e", walk, corpus, ...tiers], {
		encoding: "utf8",
		timeout: 30_000,
		killSignal: "SIGKILL",
	});
	assert.equal(r.stdout.trim(), "[false,true]", r.error?.message ?? r.stderr);
});

for (const path of [
	"MEMORY.md",
	"global/leaf.md",
	"suite/leaf.md",
	`task_only/${reference}.json`,
	`task_only/${reference}_recipe.jsonl`,
]) {
	test(`the default section is rendered in full when the library holds only ${path}`, async (t) => {
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
	assert.equal(absent, notice(f));
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
	assert.equal(f.mem.section(), notice(f));
	writeFileSync(join(f.root, "task_only", `${tag}.json`), "{}");
	await f.start();
	assert.equal(f.mem.section(), f.mem.render(`${section}\n\n${required}`, { task: "" }));
});

test("a robot on the published corpus gets the notice when its synced library holds nothing", async (t) => {
	// Offline, the sync returns without the network and the library is what is on disk.
	const offline = process.env.HF_HUB_OFFLINE;
	process.env.HF_HUB_OFFLINE = "1";
	t.after(() => {
		if (offline === undefined) delete process.env.HF_HUB_OFFLINE;
		else process.env.HF_HUB_OFFLINE = offline;
	});
	const f = fixture(t, { published: true });
	await f.start();
	assert.equal(f.mem.profile, "hf");
	assert.equal(f.mem.configError(), undefined);
	assert.equal(f.mem.section(), notice(f));
	mkdirSync(f.root);
	writeFileSync(join(f.root, "MEMORY.md"), "# Memory");
	await f.start();
	assert.equal(f.mem.section(), f.mem.render(`${section}\n\n${required}`, { task: "" }));
});

test("a robot's own template is rendered and closed by required.md whatever the library holds; memory off and a session without a cell have no section", async (t) => {
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
