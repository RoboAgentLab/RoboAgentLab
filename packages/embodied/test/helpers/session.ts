/**
 * A session file as pi names it, and its first line (../../src/infra/session-files.ts), for the stand-in pi
 * scripts of the eval tests: they write the header to "$dir/SESSION_FILE", then append the episode's entries.
 */
export const SESSION_FILE = "2026-01-01T00-00-00-000Z_s.jsonl";
export const SESSION_HEADER =
	'{"type":"session","version":3,"id":"s","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/"}';
