/**
 * Tool call ids that a planner model makes itself: `<prefix><n>`, n counted up from 1 (`ft_<n>`, ../modes/finetuned;
 * `psv_<n>`, ../robots/humanclaw/provider.ts).
 *
 * pi builds such a provider again whenever it starts a session over a file (a fork or clone, a resumed or reloaded
 * session, --continue), and the new episode's requests carry the file's earlier calls and their results. Counted from
 * 1 again, the new episode's first id would name an earlier call, and that call's result would be read as the new
 * one's. So a session's count goes on from the highest n among all its entries (every branch, and entries a compaction
 * keeps out of the context): a new call takes none of the ids they hold, also in a fork of the file or for a later
 * model that is sent the transcript.
 */

import type { SessionEntry } from "@earendil-works/pi-coding-agent";

/**
 * The highest n that is counted, and the highest a planner makes (callId). No session makes that many calls; a larger
 * number is another maker's id, and counting on from it would soon reach numbers a double cannot tell apart.
 */
export const LARGEST = 2 ** 50;

/**
 * A planner's call id `<prefix><n>`. Past LARGEST it makes none (it throws, and the planner's turn fails with that
 * error), so the next start counts every id a planner made.
 */
export function callId(prefix: string, n: number): string {
	if (!(n <= LARGEST)) throw new Error(`the session's call ids reached ${prefix}${LARGEST}; start a new session`);
	return `${prefix}${n}`;
}

/**
 * The highest n of the `<prefix><n>` ids in a session's entries, 0 when there is none: the ids of the assistant
 * messages' tool calls (one whose result never came, a run stopped during it, included) and of the tool results. An
 * entry of another shape (a call without an id, an assistant message without content) holds no such id.
 */
export function highestCallNumber(session: { getEntries(): readonly SessionEntry[] }, prefix: string): number {
	let highest = 0;
	const count = (id: unknown) => {
		const digits = typeof id === "string" && id.startsWith(prefix) ? id.slice(prefix.length) : "";
		const n = /^\d+$/.test(digits) ? Number(digits) : 0;
		if (n <= LARGEST && n > highest) highest = n;
	};
	for (const e of session.getEntries()) {
		if (e.type !== "message" || !e.message) continue;
		const m = e.message;
		if (m.role === "toolResult") count(m.toolCallId);
		if (m.role === "assistant" && Array.isArray(m.content))
			for (const c of m.content) if (c?.type === "toolCall") count(c.id);
	}
	return highest;
}
