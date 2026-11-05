/**
 * `/embodied-setup <request>`: what a free-text request settles before the wizard asks. The text
 * ("reproduce RPent on LIBERO-PRO", "franka units privileged on me@gpu", "Code as Policies, maniskill,
 * openai/gpt-6") names a robot (by id or label; a name several robots share narrows their menu), a
 * work of the survey or a ported setting (by short name or id), a cell or one of its levels (O1–O3,
 * A1 Code / A1 Model / A2 / A3, or units / tools / code / privileged / sam3), single or multi-turn
 * (CaP-X's S or M tiers), a VLA run alone with no agent (the VLA-alone baseline, outside the grid), a
 * planner (a registry model by its provider/id, its id or its display name; the fine-tuned student;
 * flash/replay), and where the services run (remote, an ssh host). Nothing here asks: ./index.ts takes
 * what is settled and asks for the rest, in its usual menus.
 */

import { type A, type Cell, type O, PAPERS, type Paper, WORKS, type Work } from "./axes.ts";

export type Kind = "sim" | "real";
type RobotLike = { id: string; label: string; kind: Kind };
/** A planner the text may name: its provider/id, and the display name the registry shows (Claude Fable 5.1). */
export type ModelRef = string | { ref: string; name?: string };
/** The planners that are no registry model: the robot's fine-tuned student, and Flash's replay of a recorded plan. */
export const FINETUNED_REF = "finetuned";
export const FLASH_REF = "flash/replay";

export type Request<R extends RobotLike = RobotLike> = {
	/** What the text settled, in words, for the confirmation. */
	settled: string[];
	kind?: Kind;
	/** The robots the text names: one is taken, several narrow the menu. */
	robots?: R[];
	/** A ported setting the text names (its cell comes with it). */
	paper?: Paper;
	/** A work of the survey the text names (its cells come with it; its ported settings, when any, narrow the picker). */
	work?: Work;
	/** The levels the text names; both make a cell. */
	o?: O;
	a?: A;
	/** Single-turn (one program: CaP-X's S tiers) or multi-turn (its M tiers, every other setting): narrows the picker. */
	turns?: "single" | "multi";
	/** A VLA run alone, no agent planning (`--vla-alone`): a baseline outside the grid, not A1 Model. */
	vlaAlone?: boolean;
	where?: "here" | "remote";
	host?: string;
	/** A planner the text names: a registry model (provider/id), FINETUNED_REF or FLASH_REF. */
	model?: string;
	/** Names in the text that are no robot, setting, work, level or planner: possibly something not supported yet. */
	unknown: string[];
};

/** Letters and digits only, lower case: "LIBERO-PRO", "libero pro" and "libero-pro" are one key. */
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The ASCII tokens of the text (names, ids, hosts), and each pair of neighbours, normalized. */
function keys(text: string): string[] {
	const tokens = text.match(/[A-Za-z0-9][A-Za-z0-9._+/:@~-]*/g) ?? [];
	const out = new Set<string>();
	for (let i = 0; i < tokens.length; i++) {
		out.add(norm(tokens[i]));
		if (i + 1 < tokens.length) out.add(norm(`${tokens[i]}${tokens[i + 1]}`));
		if (i + 2 < tokens.length) out.add(norm(`${tokens[i]}${tokens[i + 1]}${tokens[i + 2]}`));
	}
	out.delete("");
	return [...out];
}

const O_WORDS: [RegExp, O][] = [
	[/\bo1\b|cameras? only|sensory inputs?/, "O1"],
	[/\bo2\b|perception tools?|detections?|\bsam3\b|\bsegment\b/, "O2"],
	[/\bo3\b|privileged|ground[ -]truth|oracle/, "O3"],
];
const A_WORDS: [RegExp, A][] = [
	[/\ba1[ -]?code\b|\bunits?\b|action units?/, "A1C"],
	[/\ba1[ -]?model\b|\bvla\b|learned executor/, "A1M"],
	[/\ba2\b|tool orchestration|\btools\b/, "A2"],
	[/\ba3\b|code synthesis|\bcode\b/, "A3"],
];

/** A VLA with no agent around it: "VLA only", "VLA alone", "no agent", "without an agent", "VLA baseline". */
const VLA_ALONE =
	/vla[ -]?(only|alone|baseline)\b|\bno[ -]agent\b|\bwithout (an |the |any )?agent\b|\bagent[ -]?free\b/;
const TURNS: [RegExp, "single" | "multi"][] = [
	[/\bmulti[ -]?turns?\b/, "multi"],
	[/\bsingle[ -]?turns?\b|\bone[ -]?shot\b|\bone program\b/, "single"],
];
/** The planners that are no registry model, by the words a request uses for them. */
const PLANNER_WORDS: [RegExp, string][] = [
	[/\bfine[ -]?tuned (student|adapter|planner|policy)\b|\bstudent\b|\bfinetuned\/|\bdistilled\b/, FINETUNED_REF],
	[/\bflash\b|\breplay\b/, FLASH_REF],
];
/** Words only, lower case, single-spaced and padded: " claude fable 5 1 ", to find a display name in. */
const words = (s: string) =>
	` ${s
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, " ")
		.trim()} `;

export function parseRequest<R extends RobotLike>(text: string, robots: R[], models: ModelRef[]): Request<R> {
	const r: Request<R> = { settled: [], unknown: [] };
	const t = text.trim();
	if (!t) return r;
	const lower = t.toLowerCase();
	const ks = keys(t);
	const has = (k: string) => ks.includes(k);

	if (/\b(sim|simulation|simulator)\b|仿真|模拟/.test(lower)) r.kind = "sim";
	else if (/\b(real|physical)\b|真机|实机|实物/.test(lower)) r.kind = "real";

	// A robot: an exact id or label; else every robot whose id or label contains a token, or whose label
	// without its qualifier is one (franka: all three Frankas).
	const exact = robots.filter((x) => has(norm(x.id)) || has(norm(x.label)));
	// "franka" is the id of one robot and the bare label of two: then every Franka is a candidate.
	const bare = robots.filter((x) => has(norm(x.label.replace(/\(.*\)/, ""))));
	const unique = exact.length === 1 && bare.every((x) => x === exact[0]);
	const partial = unique
		? []
		: robots.filter((x) => ks.some((k) => k.length >= 5 && (norm(x.id).includes(k) || norm(x.label).includes(k))));
	const found = unique ? exact : partial.length ? partial : exact;
	if (found.length) {
		r.robots = found;
		r.settled.push(
			found.length === 1 ? `robot ${found[0].label}` : `robot: one of ${found.map((x) => x.label).join(" / ")}`,
		);
		if (!r.kind && found.every((x) => x.kind === found[0].kind)) r.kind = found[0].kind;
	}

	// A ported setting by its short name (rpent, capx-S2), else a work of the survey by its name or id.
	const paper = PAPERS.find((p) => has(norm(p.id)) || has(norm(p.name)));
	if (paper) {
		r.paper = paper;
		r.settled.push(`${paper.name}'s setting (${paper.cell.o} × ${paper.cell.a}, ${paper.flag})`);
	} else {
		// A name as short as a word (REAL) does not count; an id always does. A work named like a robot the
		// text names (RoboTwin, ManiSkill) is that robot: "CaP-X on RoboTwin" is CaP-X, not the RoboTwin paper.
		const robotKeys = (r.robots ?? []).flatMap((x) => [
			norm(x.id),
			norm(x.label),
			norm(x.label.replace(/\(.*\)/, "")),
		]);
		const isRobot = (w: Work) => robotKeys.some((k) => k.startsWith(norm(w.name)));
		const works = WORKS.filter(
			(w) => ((norm(w.name).length >= 5 && has(norm(w.name))) || has(norm(w.id))) && !isRobot(w),
		);
		// The longest name wins ("Code as Policies" over "Code").
		const work = works.sort((x, y) => y.name.length - x.name.length)[0];
		if (work) {
			r.work = work;
			r.settled.push(`${work.name} (the survey files it in ${work.cells.join(", ")})`);
		}
	}

	for (const [re, turns] of TURNS) if (re.test(lower)) r.turns ??= turns;
	if (r.turns) r.settled.push(`${r.turns}-turn`);
	// A VLA alone is no agent's executor: not A1 Model ("vla" alone would read as it).
	if (VLA_ALONE.test(lower)) {
		r.vlaAlone = true;
		r.settled.push("the VLA alone, no agent (the VLA-alone baseline)");
	}

	// Levels, unless a work or setting already places the cell, or there is no agent to place.
	if (!r.paper && !r.work && !r.vlaAlone) {
		for (const [re, o] of O_WORDS) if (re.test(lower)) r.o ??= o;
		for (const [re, a] of A_WORDS) if (re.test(lower)) r.a ??= a;
		if (r.o || r.a) r.settled.push(`${[r.o, r.a].filter(Boolean).join(" × ")}`);
	}

	const host = (t.match(/\b[\w.-]+@[\w.-]+\b/) ?? [])[0];
	if (host || /\b(remote|ssh)\b|远程/.test(lower)) {
		r.where = "remote";
		if (host) r.host = host;
		r.settled.push(host ? `services on ${host}` : "services on a remote box");
	} else if (/\b(here|local|locally)\b|本机|本地/.test(lower)) {
		r.where = "here";
		r.settled.push("services here");
	}

	const tokens = t.match(/[A-Za-z0-9][A-Za-z0-9._/-]*/g) ?? [];
	const refs = models.map((m) => (typeof m === "string" ? { ref: m } : m));
	const spaced = words(t);
	// A display name of at least five letters ("Claude Fable 5.1"), the longest that is in the text.
	const named = refs
		.filter((m) => m.name && norm(m.name).length >= 5 && spaced.includes(words(m.name)))
		.sort((x, y) => (y.name ?? "").length - (x.name ?? "").length)[0];
	const model =
		refs.find((m) => tokens.some((x) => x.toLowerCase() === m.ref.toLowerCase()))?.ref ??
		refs.find((m) => tokens.some((x) => x.includes("-") && m.ref.toLowerCase().endsWith(`/${x.toLowerCase()}`)))
			?.ref ??
		named?.ref ??
		PLANNER_WORDS.find(([re]) => re.test(lower))?.[1];
	if (model) {
		r.model = model;
		r.settled.push(model === FINETUNED_REF ? "planner: the fine-tuned student" : `planner ${model}`);
	}

	// What nothing above took: a token that is no part of a matched name, keyword or stop word.
	const matched = [
		...(r.robots ?? []).flatMap((x) => [norm(x.id), norm(x.label)]),
		...(r.paper ? [norm(r.paper.id), norm(r.paper.name)] : []),
		...(r.work ? [norm(r.work.id), norm(r.work.name)] : []),
		...(r.model
			? [norm(r.model), ...refs.filter((m) => m.ref === r.model && m.name).map((m) => norm(m.name ?? ""))]
			: []),
		...(r.host ? [norm(r.host)] : []),
	];
	const all = t.match(/[A-Za-z0-9][A-Za-z0-9._+/:@~-]*/g) ?? [];
	r.unknown = all.filter((x) => {
		const k = norm(x);
		if (k.length < 3 || /^\d+$/.test(k) || STOP.has(k)) return false;
		if (
			[...O_WORDS, ...A_WORDS, ...TURNS, ...PLANNER_WORDS].some(([re]) => re.test(x.toLowerCase())) ||
			VLA_ALONE.test(x.toLowerCase())
		)
			return false;
		return !matched.some((m) => m.includes(k));
	});
	return r;
}

/** Words a request is made of besides names. */
const STOP = new Set(
	`the a an and or of on in at to for with from by as is it its my our this that run runs running reproduce
	reproducing replicate setup set up use using do try start experiment experiments robot benchmark setting
	settings paper work config configuration please want i we like via box gpu machine server host remote ssh
	here local locally sim simulation simulator real physical units unit tools tool code model privileged oracle
	ground truth sam3 detections segment cameras camera only then also first one all vla learned executor
	perception action behavior behaviour multi single turn turns shot alone baseline agent without free fine tuned
	finetuned student adapter planner policy distilled flash replay openvla`
		.split(/\s+/)
		.map(norm),
);

/** The cell a settled request names, when both levels are there. */
export const requestCell = (r: Request): Cell | undefined => (r.o && r.a ? { o: r.o, a: r.a } : undefined);
