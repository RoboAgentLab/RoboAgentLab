/**
 * The session files pi writes, told from every other `.jsonl` in a session dir. pi names a session
 * `<time>_<session id>.jsonl` (the ISO time it was created, with ":" and "." as "-"; coding-agent
 * session-manager.ts) and writes its header, `{"type":"session","id":<that id>,...}`, as the first line.
 * An episode's records (eval.sh's result, --replay, the SFT and flywheel exports) are read from these
 * only: the cell's other files in the same dir, such as the recipe the runtime exports there, are not
 * sessions, whatever their lines say. A session saved under another name (an export, a renamed copy, a
 * link) is read when it is named, not found by a directory search. The flywheel's export has the same rule
 * in Python (services/pi_embodied_services/flywheel/planner_export.py; test/fixtures/session-files.json
 * holds the cases both must agree on).
 */

import { closeSync, lstatSync, openSync, readdirSync, readSync, statSync } from "node:fs";
import { basename, join } from "node:path";

/** pi's session file name; the group is the session id (session-manager.ts assertValidSessionId). */
export const SESSION_FILE_NAME =
	/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{3}Z_([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\.jsonl$/;

/** Whether `path` is a session file pi wrote: a regular file with pi's name whose first line is pi's header with that id. */
export function isSessionFile(path: string): boolean {
	const id = SESSION_FILE_NAME.exec(basename(path))?.[1];
	if (!id) return false;
	try {
		const header = JSON.parse(firstLine(path) ?? "") as { type?: unknown; id?: unknown } | null;
		return header?.type === "session" && header.id === id;
	} catch {
		return false;
	}
}

/**
 * Whether the file at `path` (a link followed) begins with a pi session header, whatever its name: a session a
 * directory search passes over (saved under another name, or linked) unless it is one pi wrote there.
 */
export function hasSessionHeader(path: string): boolean {
	try {
		return (JSON.parse(firstLine(path, true) ?? "") as { type?: unknown } | null)?.type === "session";
	} catch {
		return false;
	}
}

/** The session files pi wrote directly in `dir`, by name (oldest first: a name starts with its creation time). */
export function sessionFiles(dir: string): string[] {
	return readdirSync(dir)
		.sort()
		.map((f) => join(dir, f))
		.filter(isSessionFile);
}

const HEADER_BYTES = 1 << 20;

/**
 * The first line of a regular file (its newline left out; a link is followed only when `follow`), or undefined when
 * it is longer than 1 MiB or the path is no regular file.
 */
function firstLine(path: string, follow = false): string | undefined {
	if (!(follow ? statSync : lstatSync)(path).isFile()) return undefined;
	const fd = openSync(path, "r");
	try {
		const chunks: Buffer[] = [];
		const chunk = Buffer.alloc(64 * 1024);
		for (let total = 0; ; ) {
			const n = readSync(fd, chunk, 0, chunk.length, total);
			const end = chunk.subarray(0, n).indexOf(10);
			chunks.push(Buffer.from(chunk.subarray(0, end === -1 ? n : end)));
			total += end === -1 ? n : end;
			if (total > HEADER_BYTES) return undefined;
			if (n === 0 || end !== -1) return Buffer.concat(chunks).toString("utf8");
		}
	} finally {
		closeSync(fd);
	}
}
