import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createFindToolDefinition } from "../src/core/tools/find.ts";
import { createGrepToolDefinition } from "../src/core/tools/grep.ts";
import { getToolPath } from "../src/utils/tools-manager.ts";

/**
 * Fork addition (FORK.md): `readIgnoreFiles: false` keeps ripgrep and fd from opening any ignore file.
 * The ignore files in, below and above the searched directory, `.git/info/exclude` and git's config
 * files are named pipes here: a search that opens one waits for a writer and never returns. The global
 * excludes files (git's and fd's), which ripgrep and fd open only as regular files, are links to rules
 * that would hide a searched file. Inside a git repository and outside one, which find tells apart.
 */
const IGNORE_NAMES = [".gitignore", ".ignore", ".rgignore", ".fdignore"];
// Found as pi finds them (its bin dir or PATH), never downloaded here.
const rg = getToolPath("rg");
const fd = getToolPath("fd");

type Result = { content: Array<{ type: string; text?: string }> };
const lines = (r: Result) =>
	(r.content[0]?.text ?? "")
		.split("\n")
		.filter((l) => l && !l.startsWith("["))
		.sort();

async function grep(cwd: string, path: string, readIgnoreFiles?: boolean) {
	const def = createGrepToolDefinition(cwd, readIgnoreFiles === undefined ? undefined : { readIgnoreFiles });
	const ctx = {} as Parameters<typeof def.execute>[4];
	return lines(
		(await def.execute("g", { pattern: "hello", path }, AbortSignal.timeout(10_000), undefined, ctx)) as Result,
	);
}

async function find(cwd: string, path: string, readIgnoreFiles?: boolean) {
	const def = createFindToolDefinition(cwd, readIgnoreFiles === undefined ? undefined : { readIgnoreFiles });
	const ctx = {} as Parameters<typeof def.execute>[4];
	return lines(
		(await def.execute("f", { pattern: "*.txt", path }, AbortSignal.timeout(10_000), undefined, ctx)) as Result,
	);
}

/** Whether `dir` or a directory above it holds a .git, as find decides whether it searches a git repository. */
function inRepo(dir: string): boolean {
	for (let d = realpathSync(dir); ; d = dirname(d)) {
		if (existsSync(join(d, ".git"))) return true;
		if (dirname(d) === d) return false;
	}
}

for (const repo of [true, false])
	describe.skipIf(process.platform === "win32" || (!repo && inRepo(tmpdir())))(
		`search tools that read no ignore files (readIgnoreFiles: false), ${repo ? "in" : "outside"} a git repository`,
		() => {
			let base: string;
			const env = { HOME: process.env.HOME, XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME };

			beforeAll(() => {
				base = realpathSync(mkdtempSync(join(tmpdir(), "pi-no-ignore-")));
				const pipe = (path: string) => execFileSync("mkfifo", [path]);
				mkdirSync(join(base, "home"));
				mkdirSync(join(base, "xdg", "git"), { recursive: true });
				mkdirSync(join(base, "xdg", "fd"), { recursive: true });
				for (const p of ["home/.gitconfig", "xdg/git/config"]) pipe(join(base, p));
				writeFileSync(join(base, "rules.txt"), "top.txt\n");
				for (const p of ["xdg/git/ignore", "xdg/fd/ignore"]) symlinkSync(join(base, "rules.txt"), join(base, p));
				// A git repository above the searched directory, so that its ignore rules would apply.
				if (repo) {
					mkdirSync(join(base, "repo", ".git", "info"), { recursive: true });
					pipe(join(base, "repo", ".git", "info", "exclude"));
				}
				mkdirSync(join(base, "repo", "searched", "sub"), { recursive: true });
				for (const dir of ["repo", "repo/searched", "repo/searched/sub"])
					for (const name of IGNORE_NAMES) pipe(join(base, dir, name));
				writeFileSync(join(base, "repo", "searched", "top.txt"), "hello\n");
				writeFileSync(join(base, "repo", "searched", "sub", "deep.txt"), "hello\n");
				process.env.HOME = join(base, "home");
				process.env.XDG_CONFIG_HOME = join(base, "xdg");
			});

			afterAll(() => {
				for (const [k, v] of Object.entries(env)) {
					if (v === undefined) delete process.env[k];
					else process.env[k] = v;
				}
				rmSync(base, { recursive: true, force: true });
			});

			it.skipIf(!rg)("grep opens none of them and searches every file", async () => {
				const searched = join(base, "repo", "searched");
				expect(await grep(base, searched, false)).toEqual(["sub/deep.txt:1: hello", "top.txt:1: hello"]);
			});

			it.skipIf(!fd)("find opens none of them and lists every file", async () => {
				const searched = join(base, "repo", "searched");
				expect(await find(base, searched, false)).toEqual(["sub/deep.txt", "top.txt"]);
			});
		},
	);

describe("search tool descriptions", () => {
	it("say whether .gitignore is respected", () => {
		const cwd = tmpdir();
		expect(createGrepToolDefinition(cwd).description).toContain("Respects .gitignore.");
		expect(createFindToolDefinition(cwd).description).toContain("Respects .gitignore.");
		const grepDef = createGrepToolDefinition(cwd, { readIgnoreFiles: false });
		const findDef = createFindToolDefinition(cwd, { readIgnoreFiles: false });
		expect(grepDef.description).not.toContain("gitignore");
		expect(findDef.description).not.toContain("gitignore");
		expect(grepDef.promptSnippet).toBe("Search file contents for patterns");
		expect(findDef.promptSnippet).toBe("Find files by glob pattern");
	});
});

describe.skipIf(process.platform === "win32")("a regular .gitignore with and without ignore files", () => {
	// pi's default tools read the global ignore and git config files too: an empty HOME and XDG_CONFIG_HOME here.
	let base: string;
	const env = { HOME: process.env.HOME, XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME };

	beforeAll(() => {
		base = realpathSync(mkdtempSync(join(tmpdir(), "pi-ignore-default-")));
		mkdirSync(join(base, "home"));
		mkdirSync(join(base, "xdg"));
		mkdirSync(join(base, "repo", ".git"), { recursive: true });
		mkdirSync(join(base, "repo", "searched"));
		writeFileSync(join(base, "repo", "searched", ".gitignore"), "listed.txt\n");
		writeFileSync(join(base, "repo", "searched", "kept.txt"), "hello\n");
		writeFileSync(join(base, "repo", "searched", "listed.txt"), "hello\n");
		process.env.HOME = join(base, "home");
		process.env.XDG_CONFIG_HOME = join(base, "xdg");
	});

	afterAll(() => {
		for (const [k, v] of Object.entries(env)) {
			if (v === undefined) delete process.env[k];
			else process.env[k] = v;
		}
		rmSync(base, { recursive: true, force: true });
	});

	it.skipIf(!rg)("grep: by default the .gitignore hides what it lists; without ignore files it does not", async () => {
		const searched = join(base, "repo", "searched");
		expect(await grep(base, searched)).toEqual(["kept.txt:1: hello"]);
		expect(await grep(base, searched, false)).toEqual(["kept.txt:1: hello", "listed.txt:1: hello"]);
	});

	it.skipIf(!fd)("find: by default the .gitignore hides what it lists; without ignore files it does not", async () => {
		const searched = join(base, "repo", "searched");
		expect(await find(base, searched)).toEqual(["kept.txt"]);
		expect(await find(base, searched, false)).toEqual(["kept.txt", "listed.txt"]);
	});
});
