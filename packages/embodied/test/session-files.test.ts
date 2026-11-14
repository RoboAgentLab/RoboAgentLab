// The session-file rule (../src/infra/session-files.ts): the cases it shares with the flywheel's Python copy, the kinds of files it refuses, and pi's own sessions.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { test } from "node:test";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { sessionFile } from "../src/capabilities/replay/index.ts";
import { hasSessionHeader, isSessionFile, SESSION_FILE_NAME, sessionFiles } from "../src/infra/session-files.ts";

const header = (id: string, pad = "") => `${JSON.stringify({ type: "session", version: 3, id, cwd: "/", pad })}\n`;
const FIXTURES = JSON.parse(readFileSync(new URL("./fixtures/session-files.json", import.meta.url), "utf8")) as {
	cases: { name: string; first?: string; first_base64?: string; session: boolean; why: string }[];
};

test("the session-file rule decides the cases it shares with the flywheel's Python copy", (t) => {
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "session-files-")));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	for (const c of FIXTURES.cases) {
		const first = c.first_base64 === undefined ? Buffer.from(c.first ?? "") : Buffer.from(c.first_base64, "base64");
		writeFileSync(join(dir, c.name), Buffer.concat([first, Buffer.from("\n{}\n")]));
		assert.equal(isSessionFile(join(dir, c.name)), c.session, `${c.name}: ${c.why}`);
	}
});

test("a session file is a regular file: not a link, a directory or a pipe; a header past 1 MiB is none", (t) => {
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "session-files-kinds-")));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const elsewhere = join(dir, "elsewhere");
	writeFileSync(elsewhere, header("l"));
	symlinkSync(elsewhere, join(dir, "2026-01-01T00-00-00-000Z_l.jsonl"));
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-000Z_l.jsonl")), false);
	mkdirSync(join(dir, "2026-01-01T00-00-00-001Z_d.jsonl"));
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-001Z_d.jsonl")), false);
	assert.equal(spawnSync("mkfifo", [join(dir, "2026-01-01T00-00-00-002Z_p.jsonl")]).status, 0);
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-002Z_p.jsonl")), false, "a pipe is not opened");
	// A long header is read past the first chunk; one past the bound is not a header.
	writeFileSync(join(dir, "2026-01-01T00-00-00-003Z_a.jsonl"), header("a", "x".repeat(200_000)));
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-003Z_a.jsonl")), true);
	writeFileSync(join(dir, "2026-01-01T00-00-00-004Z_a.jsonl"), header("a", "x".repeat(1_100_000)));
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-004Z_a.jsonl")), false);
	// The bound is the line's, not the JSON's: a header followed on its line by more than 1 MiB of spaces is none;
	// a line of exactly 1 MiB is read, one byte more is not.
	const exactly = (id: string, bytes: number) => {
		const h = header(id).trimEnd();
		return `${h}${" ".repeat(bytes - Buffer.byteLength(h))}\n`;
	};
	writeFileSync(join(dir, "2026-01-01T00-00-00-006Z_a.jsonl"), exactly("a", 1 << 20));
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-006Z_a.jsonl")), true);
	writeFileSync(join(dir, "2026-01-01T00-00-00-007Z_a.jsonl"), exactly("a", (1 << 20) + 1));
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-007Z_a.jsonl")), false);
	writeFileSync(join(dir, "2026-01-01T00-00-00-005Z_a.jsonl"), `${header("a").trimEnd()}${" ".repeat(1_100_000)}\n`);
	assert.equal(isSessionFile(join(dir, "2026-01-01T00-00-00-005Z_a.jsonl")), false);
});

// The order is the contract (Node's readdirSync returns names sorted already).
test("a directory's session files come by name, oldest first, whatever order they were written in", (t) => {
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "session-files-order-")));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	for (const name of ["2026-01-03T00-00-00-000Z_c", "2026-01-01T00-00-00-000Z_a", "2026-01-02T00-00-00-000Z_b"])
		writeFileSync(join(dir, `${name}.jsonl`), header(name.split("_")[1]));
	writeFileSync(join(dir, "cell_notes.jsonl"), header("s"));
	assert.deepEqual(
		sessionFiles(dir).map((p) => basename(p)),
		["2026-01-01T00-00-00-000Z_a.jsonl", "2026-01-02T00-00-00-000Z_b.jsonl", "2026-01-03T00-00-00-000Z_c.jsonl"],
	);
});

test("a session a directory search passes over is a file, or a link to one, that begins with a session header", (t) => {
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "session-files-header-")));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	writeFileSync(join(dir, "session-export.jsonl"), header("s"));
	symlinkSync(join(dir, "session-export.jsonl"), join(dir, "2026-01-01T00-00-00-000Z_s.jsonl"));
	writeFileSync(join(dir, "cell_recipe.jsonl"), `${JSON.stringify({ action: "move", xyz: [0, 0, 1] })}\n`);
	// A first line that is another entry (MotorMind's truth log, a custom entry) is no session's.
	writeFileSync(join(dir, "cell_mm_truth.jsonl"), `${JSON.stringify({ type: "custom", id: "s" })}\n`);
	mkdirSync(join(dir, "cell_notes.jsonl"));
	assert.deepEqual(
		[
			"session-export.jsonl",
			"2026-01-01T00-00-00-000Z_s.jsonl",
			"cell_recipe.jsonl",
			"cell_mm_truth.jsonl",
			"cell_notes.jsonl",
			"none.jsonl",
		].map((f) => hasSessionHeader(join(dir, f))),
		[true, true, false, false, false, false],
	);
});

test("the session file pi writes is one by this rule", (t) => {
	// pi's own naming (coding-agent session-manager.ts): if it changes, every reader of episode records must follow.
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "session-files-pi-")));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	for (const id of [undefined, "custom.id-1"]) {
		const sm = SessionManager.create(dir, dir, id ? { id } : undefined);
		sm.appendMessage({ role: "user", content: "Solve the task.", timestamp: Date.now() });
		const file = sm.getSessionFile();
		assert.ok(file && SESSION_FILE_NAME.test(basename(file)), file);
		assert.equal(isSessionFile(file), true);
	}
	assert.equal(sessionFiles(dir).length, 2);
});

test("a session dir replays the session pi wrote, not a newer .jsonl of the cell shaped as one", (t) => {
	const dir = realpathSync(mkdtempSync(join(tmpdir(), "session-files-replay-")));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const real = join(dir, "2026-01-01T00-00-00-000Z_s.jsonl");
	writeFileSync(real, header("s"));
	// Written later: a planner's own file and a recipe whose first line carries a call's arguments.
	writeFileSync(join(dir, "cell_notes.jsonl"), header("s"));
	writeFileSync(join(dir, "cell_recipe.jsonl"), `${JSON.stringify({ action: "move", type: "session", id: "s" })}\n`);
	writeFileSync(join(dir, "2026-01-02T00-00-00-000Z_t.jsonl"), header("u"));
	assert.equal(sessionFile(dir), real);
	rmSync(real);
	assert.throws(
		() => sessionFile(dir),
		/^Error: no session file pi wrote in .* \(a regular file <time>_<session id>\.jsonl, with that id in its first line\); a session saved under another name or linked is read when named$/,
	);
});
