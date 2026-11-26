import assert from "node:assert/strict";
import { tmpdir } from "node:os";
import { test } from "node:test";
import { type SessionEntry, SessionManager } from "@earendil-works/pi-coding-agent";
import { callId, highestCallNumber, LARGEST } from "../src/planner/call-ids.ts";

/** A session's file as pi reads it: its entries, every branch. */
const file = (...entries: unknown[]) => ({ getEntries: () => entries as SessionEntry[] });
const message = (m: object) => ({ type: "message", id: "e", parentId: null, timestamp: "", message: m });
/** An assistant message calling `ids` (one call each), and the result of the call `id`. */
const call = (...ids: string[]) =>
	message({
		role: "assistant",
		content: [
			{ type: "text", text: "next" },
			...ids.map((id) => ({ type: "toolCall", id, name: "act", arguments: {} })),
		],
	});
const result = (id: string) =>
	message({ role: "toolResult", toolCallId: id, toolName: "act", content: [{ type: "text", text: "ok" }] });
/** Calls and their results. */
const episode = (...ids: string[]) => ids.flatMap((id) => [call(id), result(id)]);

test("a session's highest call number is 0 without calls of the prefix: an empty file, and entries that hold no call", () => {
	assert.equal(highestCallNumber(file(), "ft_"), 0);
	const none = file(
		message({ role: "user", content: "Solve the task." }),
		message({ role: "assistant", content: [{ type: "text", text: "ft_9" }] }),
		{ type: "custom", customType: "finetuned_step", data: { step: 9, id: "ft_9" } },
		{ type: "custom_message", customType: "note", content: "ft_9", display: false },
		{ type: "compaction", summary: "ft_9", firstKeptEntryId: "e", tokensBefore: 9 },
		{ type: "model_change", provider: "finetuned", modelId: "ft_9" },
	);
	assert.equal(highestCallNumber(none, "ft_"), 0);
});

test("a session's highest call number is the highest n of the prefix's calls and results anywhere in the file", () => {
	const cases: [string, unknown[], number][] = [
		["one episode's calls and results", episode("ft_1", "ft_2", "ft_3"), 3],
		// main wrote these: a resumed session opened its episode with ft_1 again.
		["ids that started again from 1 in a file written before", episode("ft_1", "ft_2", "ft_3", "ft_4", "ft_1"), 4],
		["a call whose result never came (a run stopped during it)", [...episode("ft_1"), call("ft_2")], 2],
		["a result whose call the file does not hold", [...episode("ft_1"), result("ft_3")], 3],
		["several calls of one message", [call("ft_1", "ft_5", "ft_2")], 5],
		[
			"other makers' ids, and ids that only resemble the prefix's",
			[
				...episode("ft_2"),
				call("flash_40", "psv_41", "replay_42", "motormind-43", "human_44", "call_45", "toolu_46"),
				call(
					"ft_",
					"ft_x47",
					"ft_48b",
					"FT_49",
					"xft_50",
					" ft_51",
					"ft_ 52",
					"ft_-53",
					"ft_+54",
					"ft_55.0",
					"ft_5e1",
				),
				result("ft_56b"),
			],
			2,
		],
		[
			"calls after a compaction",
			[
				...episode("ft_1", "ft_2"),
				{ type: "compaction", summary: "an episode", firstKeptEntryId: "e", tokensBefore: 9 },
				...episode("ft_3"),
			],
			3,
		],
		[
			"a compaction with no call after it",
			[
				...episode("ft_1", "ft_2"),
				{ type: "compaction", summary: "an episode", firstKeptEntryId: "e", tokensBefore: 9 },
			],
			2,
		],
		["the largest number that is counted", episode("ft_7", "ft_1125899906842624"), 1125899906842624],
		[
			"numbers above the largest that is counted",
			episode("ft_7", "ft_1125899906842625", "ft_9007199254740991", `ft_${"9".repeat(22)}`, "ft_9007199254740993"),
			7,
		],
	];
	for (const [what, entries, highest] of cases)
		assert.equal(highestCallNumber(file(...entries), "ft_"), highest, what);
	// Each prefix counts its own.
	const both = file(...episode("ft_9", "psv_3", "ft_1"));
	assert.deepEqual([highestCallNumber(both, "ft_"), highestCallNumber(both, "psv_")], [9, 3]);
});

test("a session's highest call number holds no id of an entry of another shape, and reads on past it: a call without an id or with a number, an empty part, a result without its call's id, a message without content or with content that is no list, an entry without a message or with a null one", () => {
	const odd = [
		message({ role: "assistant", content: [{ type: "toolCall", name: "act", arguments: {} }] }),
		message({ role: "assistant", content: [{ type: "toolCall", id: 7, name: "act", arguments: {} }] }),
		message({ role: "assistant", content: [null] }),
		message({ role: "toolResult", toolName: "act", content: [] }),
		message({ role: "toolResult", toolCallId: 7, toolName: "act", content: [] }),
		message({ role: "assistant", content: null }),
		message({ role: "assistant" }),
		message({ role: "assistant", content: { type: "toolCall", id: "ft_9", name: "act", arguments: {} } }),
		{ type: "message", id: "e", parentId: null, timestamp: "" },
		{ type: "message", id: "e", parentId: null, timestamp: "", message: null },
	];
	for (const entry of odd)
		assert.equal(
			highestCallNumber(file(...episode("ft_1"), entry, ...episode("ft_4", "ft_2")), "ft_"),
			4,
			JSON.stringify(entry),
		);
});

test("a session's highest call number reads all of pi's own session: every branch, and calls a compaction keeps out of the context", () => {
	const s = SessionManager.inMemory(tmpdir());
	const prompt = s.appendMessage({ role: "user", content: "Solve the task.", timestamp: 0 });
	/** One call and its result; the call's entry. */
	const step = (id: string) => {
		const at = s.appendMessage(call(id).message as never);
		s.appendMessage(result(id).message as never);
		return at;
	};
	step("ft_1");
	const kept = step("ft_2");
	s.appendCompaction("an episode", kept, 9);
	step("ft_3");
	const inContext = () =>
		s
			.buildContextEntries()
			.flatMap((e) => (e.type === "message" && e.message.role === "toolResult" ? [e.message.toolCallId] : []));
	// The context holds the calls the compaction kept and the later one, not ft_1.
	assert.deepEqual(inContext(), ["ft_2", "ft_3"]);
	assert.equal(highestCallNumber(s, "ft_"), 3);
	// Back on the tree at the prompt: the leaf's path holds none of the calls.
	s.branch(prompt);
	assert.deepEqual(inContext(), []);
	assert.equal(highestCallNumber(s, "ft_"), 3);
});

test("a planner's call id is its prefix and the number, up to the largest number that is counted; past it the planner makes none, so the next start counts every id it made", () => {
	assert.equal(callId("ft_", 1), "ft_1");
	assert.equal(callId("psv_", LARGEST), `psv_${LARGEST}`);
	for (const n of [LARGEST + 1, LARGEST + 2])
		assert.throws(() => callId("ft_", n), {
			message: `the session's call ids reached ft_${LARGEST}; start a new session`,
		});
	assert.equal(highestCallNumber(file(...episode(callId("ft_", LARGEST))), "ft_"), LARGEST);
});
