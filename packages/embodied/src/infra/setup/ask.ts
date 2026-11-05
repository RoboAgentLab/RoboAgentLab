/**
 * `ask_user`: the agent asks the user what a request left open, as choices. It follows what Claude
 * Code's AskUserQuestion, Codex's request_user_input and OpenCode's question tool agree on: one to four
 * questions, each one decision with a short header and two to four options, the recommended option
 * first and marked "(Recommended)", a typed answer always possible, and the answers returned to the
 * model keyed by question. Invalid calls come back as errors the model corrects; a cancelled question
 * stops the asking. Episode sessions do not have it: an agent policy runs without a human.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type Static, Type } from "typebox";

export const ASK_TOOL = "ask_user";
/** The option every question ends with: the user types the answer instead. */
export const TYPE_OWN = "Type something else";
/** The option that ends a multi-select question. */
export const DONE = "Done";
export const RECOMMENDED_MARK = "(Recommended)";
export const LIMITS = { questions: 4, options: 4, header: 12 } as const;

const OptionSchema = Type.Object({
	label: Type.String({
		description: "The choice, 1-5 words. The recommended one is first and ends with (Recommended).",
	}),
	description: Type.Optional(Type.String({ description: "What choosing it means or changes, one short sentence." })),
});
const QuestionSchema = Type.Object({
	header: Type.String({ description: `A label of at most ${LIMITS.header} characters, e.g. "Planner", "GPU".` }),
	question: Type.String({ description: "The full question, one decision, ending with a question mark." }),
	options: Type.Array(OptionSchema, {
		description: `2-${LIMITS.options} mutually exclusive choices (unless multiSelect). Do not add an Other option: one is always added.`,
	}),
	multiSelect: Type.Optional(Type.Boolean({ description: "true when several options may be chosen together." })),
});
export const AskParams = Type.Object({
	questions: Type.Array(QuestionSchema, {
		description: `1-${LIMITS.questions} questions; prefer one unless several independent decisions block progress.`,
	}),
});
export type AskInput = Static<typeof AskParams>;
export type AskAnswer = { header: string; selected: string[]; typed?: string };

/** What is wrong with a call, for the model to correct; none when it is valid. */
export function askErrors(input: AskInput): string[] {
	const errors: string[] = [];
	const qs = input.questions ?? [];
	if (qs.length < 1 || qs.length > LIMITS.questions)
		errors.push(`ask 1-${LIMITS.questions} questions, not ${qs.length}`);
	const headers = new Set<string>();
	for (const q of qs) {
		const name = q.header || "(no header)";
		if (!q.header || q.header.length > LIMITS.header)
			errors.push(`${name}: the header must be 1-${LIMITS.header} characters`);
		if (headers.has(q.header)) errors.push(`${name}: two questions share this header`);
		headers.add(q.header);
		if (!q.question?.trim()) errors.push(`${name}: the question is empty`);
		const n = q.options?.length ?? 0;
		if (n < 2 || n > LIMITS.options) errors.push(`${name}: give 2-${LIMITS.options} options, not ${n}`);
		const labels = (q.options ?? []).map((o) => o.label.trim().toLowerCase());
		if (new Set(labels).size !== labels.length) errors.push(`${name}: two options share a label`);
		if (labels.some((l) => /^(other|type something)/.test(l)))
			errors.push(`${name}: drop the Other option; "${TYPE_OWN}" is always added`);
		const marked = (q.options ?? []).findIndex((o) => o.label.includes(RECOMMENDED_MARK));
		if (marked > 0) errors.push(`${name}: put the ${RECOMMENDED_MARK} option first`);
	}
	return errors;
}

/** An option as the menu shows it: the label, and its description after a dash. */
const shown = (o: { label: string; description?: string }) =>
	o.description ? `${o.label} — ${o.description}` : o.label;

/**
 * Asks the questions in turn as menus (in the TUI and over RPC alike). Undefined when the user
 * cancels: the rest is not asked.
 */
export async function askUser(ctx: ExtensionContext, input: AskInput): Promise<AskAnswer[] | undefined> {
	const answers: AskAnswer[] = [];
	for (const q of input.questions) {
		const opts = q.options.map(shown);
		const title = `${q.header}: ${q.question}`;
		const selected: string[] = [];
		let typed: string | undefined;
		for (;;) {
			const left = q.options.filter((o) => !selected.includes(o.label));
			const menu = q.multiSelect
				? [...(selected.length ? [`${DONE} (${selected.join(", ")})`] : []), ...left.map(shown), TYPE_OWN]
				: [...opts, TYPE_OWN];
			const picked = await ctx.ui.select(title, menu);
			if (picked === undefined) return undefined;
			if (picked.startsWith(DONE)) break;
			if (picked === TYPE_OWN) {
				typed = (await ctx.ui.input(title, ""))?.trim();
				if (typed === undefined) return undefined;
				break;
			}
			const option = q.options[opts.indexOf(picked)];
			if (option) selected.push(option.label);
			if (!q.multiSelect || selected.length === q.options.length) break;
		}
		answers.push({ header: q.header, selected, ...(typed ? { typed } : {}) });
	}
	return answers;
}

/** The answers as the model reads them: one line per question. */
export const answerText = (answers: AskAnswer[]) =>
	answers
		.map((a) => {
			const parts = [...a.selected, ...(a.typed ? [`typed: ${a.typed}`] : [])];
			return `${a.header}: ${parts.length ? parts.join("; ") : "(no answer)"}`;
		})
		.join("\n");

const result = (text: string, details: unknown) => ({ content: [{ type: "text" as const, text }], details });

/** Registers ask_user; `inEpisode` tells whether a robot is loaded (then it is removed from the model's tools). */
export function registerAsk(pi: ExtensionAPI, inEpisode: () => boolean): void {
	pi.registerTool({
		name: ASK_TOOL,
		label: "Ask the user",
		description:
			"Ask the user 1-4 multiple-choice questions and wait for the answers. Use it when a decision is the user's and the request does not settle it (which robot, which setting, where the services run, which planner, which GPU). The user can always type their own answer.",
		promptSnippet: `${ASK_TOOL}: ask the user multiple-choice questions when a decision is theirs`,
		promptGuidelines: [
			`Use ${ASK_TOOL} only for decisions the user's words and the catalog do not settle; never ask what they already said, and never to confirm a plan (the setup's own confirmation does that).`,
			`Build the options from the catalog you were given, never invented names; put the option you recommend first with "${RECOMMENDED_MARK}" at the end of its label, and say in its description why.`,
			"Ask independent decisions together (up to 4); ask a decision that depends on an earlier answer after it.",
			"If the user cancels, stop and ask in plain text what they want instead.",
		],
		parameters: AskParams,
		async execute(_id, params, _signal, _update, ctx) {
			if (inEpisode())
				return result("ask_user is not available in an episode: the policy runs without a human.", {});
			if (!ctx.hasUI) return result("No UI to ask in: ask the user in your reply instead.", {});
			const errors = askErrors(params);
			if (errors.length) return result(`Not asked; fix the call: ${errors.join("; ")}`, { errors });
			const answers = await askUser(ctx, params);
			if (!answers)
				return result("The user cancelled the questions. Stop and ask in plain text what they want.", {});
			return result(answerText(answers), { answers });
		},
	});
	// An episode's planner never gets it.
	pi.on("session_start", () => {
		if (inEpisode()) pi.setActiveTools(pi.getActiveTools().filter((t) => t !== ASK_TOOL));
	});
}
