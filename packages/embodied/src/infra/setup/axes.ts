/**
 * The survey's Agent-as-Policy taxonomy (its Figure A1a) as /embodied-setup's grid: observations
 * (O1 Sensory Inputs, O2 Perception Tools, O3 Privileged State) by execution authority (A1 Behavior
 * Selection with its Code and Model sub-columns, A2 Tool Orchestration, A3 Code Synthesis). A robot
 * supports a level when it names it in its `Axes`, with the flag that level adds (empty for its
 * defaults). The survey's works are filed per cell (./works.json, from the survey's papers.json by
 * ../../scripts/works-catalog.mjs); the ported ones (`PAPERS`) run as a `--preset` or `--tier`.
 *
 * The grid and the work picker are pi custom components (./index.ts): gridMove, gridLines, pickMatches
 * and pickerLines are their keyboard and rendering, pure so the tests drive them without a terminal.
 */

import { readFileSync } from "node:fs";

export type O = "O1" | "O2" | "O3";
export type A = "A1C" | "A1M" | "A2" | "A3";
export type Cell = { o: O; a: A };
export const OS: O[] = ["O1", "O2", "O3"];
export const AS: A[] = ["A1C", "A1M", "A2", "A3"];

/** A level: the survey's name, section and definition, and what it is in this harness. */
type Level = { name: string; section: string; paper: string; here: string };
export const OBSERVATION: Record<O, Level> = {
	O1: {
		name: "Sensory Inputs",
		section: "§3.1",
		paper: "Decisions start from sensory inputs; agent-built perception remains O1.",
		here: "camera frames only; the planner reads the scene itself",
	},
	O2: {
		name: "Perception Tools",
		section: "§3.2",
		paper: "Decisions use supplied semantic or geometric perception tools or outputs.",
		here: "SAM3 masks with ids next to the frames (detect, select_detection, reject_detection; the arms' segment)",
	},
	O3: {
		name: "Privileged State",
		section: "§3.3",
		paper: "Decisions use oracle state unavailable at deployment, assessed per evaluated phase.",
		here: "the simulator's ground-truth object poses (ground_truth_poses); the result is marked privileged",
	},
};
export const EXECUTION: Record<A, Level> = {
	A1C: {
		name: "Behavior Selection · Code",
		section: "§3.4.1",
		paper: "The agent specifies behaviors, targets, or trajectories; a fixed programmed executor realizes them.",
		here: "Show-Harness action units (act: move, rotate, grasp, release) run by the robot's own motion code; no VLA",
	},
	A1M: {
		name: "Behavior Selection · Model",
		section: "§3.4.2",
		paper: "The agent specifies behaviors, targets, or trajectories; a fixed learned executor realizes them.",
		here: "the robot's VLA skill (Pi0.5, RLDX-1, LingBot) realizes each behavior request; the run requires its server",
	},
	A2: {
		name: "Tool Orchestration",
		section: "§3.5",
		paper: "The agent selects existing execution APIs, supplies arguments, and coordinates their use.",
		here: "the robot's own tools (VLA, segmentation, motion) composed by the planner",
	},
	A3: {
		name: "Code Synthesis",
		section: "§3.6",
		paper: "The agent writes executable logic, including conditions, loops, or computations, and may reuse APIs.",
		here: "run_code: Python over the robot's primitives (CaP-X) with conditions, loops and computation",
	},
};
/** The column's short name in a cell's name: `O1 × A1 Code`. */
export const COLUMN: Record<A, string> = { A1C: "A1 Code", A1M: "A1 Model", A2: "A2", A3: "A3" };

/** The levels a robot has, each with the flag it adds (empty: the robot's default). Absent = no such level. */
export type Axes = Partial<Record<O | A, string>>;

export const has = (axes: Axes, level: O | A): boolean => axes[level] !== undefined;
export const supported = (axes: Axes, c: Cell): boolean => has(axes, c.o) && has(axes, c.a);
/** Row-major (O1 first), the order the grid cursor starts from. */
export const supportedCells = (axes: Axes): Cell[] =>
	OS.flatMap((o) => AS.map((a) => ({ o, a }))).filter((c) => supported(axes, c));
export const cellName = (c: Cell) => `${c.o} × ${COLUMN[c.a]}`;
/** The cell as works.json files it: `O1/A1C`. */
export const cellKey = (c: Cell) => `${c.o}/${c.a}`;
export const cellLabel = (c: Cell) => `${cellName(c)}: ${OBSERVATION[c.o].name} · ${EXECUTION[c.a].name}`;
/** The launch flags of a cell: the execution level's, then the observation level's. */
export const cellFlags = (axes: Axes, c: Cell): string => [axes[c.a], axes[c.o]].filter(Boolean).join(" ");

export type Dir = "up" | "down" | "left" | "right";
/** The nearest supported cell in a direction, or the cell itself when there is none. */
export function gridMove(axes: Axes, cur: Cell, dir: Dir): Cell {
	const [di, dj] = dir === "up" ? [-1, 0] : dir === "down" ? [1, 0] : dir === "left" ? [0, -1] : [0, 1];
	let i = OS.indexOf(cur.o) + di;
	let j = AS.indexOf(cur.a) + dj;
	while (i >= 0 && i < OS.length && j >= 0 && j < AS.length) {
		const c = { o: OS[i], a: AS[j] };
		if (supported(axes, c)) return c;
		i += di;
		j += dj;
	}
	return cur;
}

export type Paint = { accent(s: string): string; dim(s: string): string; bold(s: string): string };
const plain: Paint = { accent: (s) => s, dim: (s) => s, bold: (s) => s };
const ROW = 22;
const COL = 14;
const pad = (s: string, n: number) => s + " ".repeat(Math.max(0, n - [...s].length));

/** The grid (A1's two sub-columns under one head), the cursor's cell explained, and the keys. */
export function gridLines(title: string, axes: Axes, cur: Cell, paint: Paint = plain): string[] {
	const head1 = pad("", ROW) + pad("A1 Behavior Selection", 2 * COL) + pad("A2 Tool", COL) + pad("A3 Code", COL);
	const head2 =
		pad("", ROW) + pad("Code", COL) + pad("Model", COL) + pad("Orchestration", COL) + pad("Synthesis", COL);
	const rows = OS.map(
		(o) =>
			pad(`${o} ${OBSERVATION[o].name}`, ROW) +
			AS.map((a) => {
				const c = { o, a };
				if (!supported(axes, c)) return paint.dim(pad(" –", COL));
				return c.o === cur.o && c.a === cur.a ? paint.accent(paint.bold(pad("[●]", COL))) : pad("[ ]", COL);
			}).join(""),
	);
	const O = OBSERVATION[cur.o];
	const A = EXECUTION[cur.a];
	const flags = cellFlags(axes, cur);
	const works = worksIn(cur).length;
	const ported = new Set(PAPERS.filter((p) => p.cell.o === cur.o && p.cell.a === cur.a).map((p) => p.work)).size;
	return [
		paint.bold(title),
		"",
		paint.dim(head1),
		paint.dim(head2),
		...rows,
		"",
		paint.bold(`${cellName(cur)}: ${O.section} ${O.name} × ${A.section} ${A.name}`),
		`  ${O.paper}`,
		`  ${A.paper}`,
		paint.dim(`  here, O: ${O.here}`),
		paint.dim(`  here, A: ${A.here}`),
		paint.dim(`  flags: ${flags || "none (the robot's defaults)"}`),
		paint.dim(`  works the survey files here: ${works}${ported ? `, ${ported} ported` : ""}`),
		"",
		paint.dim("←↑↓→ move   enter select   esc cancel   – not on this robot"),
	];
}

export type Key = Dir | "enter" | "escape";
/** Raw terminal input the components understand besides the configured select keybindings: arrows and vi keys. */
const KEYS: Record<string, Key> = {
	"\x1b[A": "up",
	"\x1bOA": "up",
	k: "up",
	"\x1b[B": "down",
	"\x1bOB": "down",
	j: "down",
	"\x1b[C": "right",
	"\x1bOC": "right",
	l: "right",
	"\x1b[D": "left",
	"\x1bOD": "left",
	h: "left",
	"\r": "enter",
	"\n": "enter",
	"\x1b": "escape",
	q: "escape",
};
/** `vi`: whether h/j/k/l/q count (not while the input is also typed text). */
export const keyOf = (data: string, vi = true): Key | undefined => (vi || data.length > 1 ? KEYS[data] : undefined);
/** Typed text: one printable character; backspace is `erase`. */
export const typed = (data: string): { char: string } | "erase" | undefined =>
	data === "\x7f" || data === "\b" ? "erase" : data.length === 1 && data >= " " ? { char: data } : undefined;

/** A work of the survey, filed in one or more cells (./works.json). */
export type Work = { id: string; name: string; title: string; year?: number; url?: string; cells: string[] };
export const WORKS: Work[] = JSON.parse(readFileSync(new URL("./works.json", import.meta.url), "utf8"));
const byName = (a: Work, b: Work) => a.name.localeCompare(b.name);
/** The survey's works filed in a cell, by name. */
export const worksIn = (c: Cell): Work[] => WORKS.filter((w) => w.cells.includes(cellKey(c))).sort(byName);

/**
 * A ported work's setting: the `--preset` / `--tier` that reproduces it (../tiers.ts expands it to the
 * flags of that setting; the cell adds none of its own), in the cell the survey files the work (`work`:
 * its works.json id), on the robots that have what it needs (`robots`: id prefixes; absent = any robot
 * that has the cell).
 */
/** A live web panel a work or a robot brings, besides the dashboard: its pi flags and the port it serves. */
export type Panel = { name: string; flags: string; port: number };
/** CaP-X's Viser scene (viser_debug), ported as `--viser` (src/observation/viser.ts): LIBERO and the Frankas. */
export const VISER: Panel = {
	name: "Viser 3D view (CaP-X's viser_debug: point clouds, camera frustums, EEF, grasp candidates)",
	flags: "--viser --viser-port 8080",
	port: 8080,
};
export type Paper = {
	id: string;
	name: string;
	work: string;
	cell: Cell;
	flag: string;
	note: string;
	robots?: string[];
	/** The work's own panels, preferred over the dashboard alone when the robot can serve them. */
	panels?: Panel[];
};
const capx = (tier: string, o: O, note: string): Paper => ({
	id: `capx-${tier}`,
	name: `CaP-X ${tier}`,
	work: "capx",
	cell: { o, a: "A3" },
	flag: `--tier ${tier}`,
	note,
	panels: [VISER],
});
export const PAPERS: Paper[] = [
	{
		id: "showharness",
		name: "Show-Harness",
		work: "showharness",
		cell: { o: "O1", a: "A1C" },
		flag: "--preset showharness",
		note: "action units with image feedback, no VLA",
	},
	{
		id: "motormind",
		name: "MotorMind",
		work: "motormind2026",
		cell: { o: "O2", a: "A1C" },
		flag: "--preset motormind",
		note: "LIBERO side camera, settle ticks, no cross-task memory",
		robots: ["libero"],
	},
	{
		id: "humanclaw",
		name: "HumanCLAW",
		work: "humanclaw",
		cell: { o: "O1", a: "A1M" },
		flag: "--preset humanclaw",
		note: "the paper's mode: its units, run by the motion model",
		robots: ["humanclaw"],
	},
	{
		id: "rpent",
		name: "RPent",
		work: "harnessvla",
		cell: { o: "O1", a: "A2" },
		flag: "--preset rpent",
		note: "tools with memory and a VLA skill",
		robots: ["libero", "robocasa", "franka", "dual-franka"],
	},
	{
		id: "openeta",
		name: "OpenETA",
		work: "openeta",
		cell: { o: "O2", a: "A2" },
		flag: "--preset openeta",
		note: "tools with visual differencing (VDM) and the anchor image",
	},
	capx("S1", "O3", "one program, high-level API, simulator ground truth"),
	capx("S2", "O2", "one program, high-level API"),
	capx("S3", "O2", "one program, low-level API with examples"),
	capx("S4", "O2", "one program, low-level API, no examples"),
	capx("M1", "O2", "multi-turn, text feedback, high-level API"),
	capx("M2", "O2", "multi-turn, image feedback, high-level API"),
	capx("M3", "O2", "multi-turn, VDM feedback, high-level API"),
	capx("M4", "O2", "multi-turn, VDM feedback, low-level API"),
];
/** The ported works whose setting is this cell on this robot (one that has the cell and what the port needs). */
export const papersFor = (robot: { id: string; axes: Axes }, c: Cell): Paper[] =>
	PAPERS.filter(
		(p) =>
			p.cell.o === c.o &&
			p.cell.a === c.a &&
			supported(robot.axes, c) &&
			(!p.robots || p.robots.some((r) => robot.id.startsWith(r))),
	);

/**
 * The work picker's entries: none, then the ported settings of the cell. The survey's other works filed
 * there (worksIn) are not offered until they are ported; the grid counts them.
 */
export type Pick = { id: string; text: string; paper?: Paper };
export function pickEntries(papers: Paper[], none: string): Pick[] {
	return [
		{ id: "none", text: none },
		...papers.map((p) => ({ id: p.id, text: `${p.id}: ${p.name}, ${p.note} (${p.flag})`, paper: p })),
	];
}
/** The entries a typed filter keeps: the id or the text (name, note) contains it (case-insensitive). */
export function pickMatches(entries: Pick[], q: string): Pick[] {
	const s = q.trim().toLowerCase();
	if (!s) return entries;
	return entries.filter((e) => e.id.toLowerCase().includes(s) || e.text.toLowerCase().includes(s));
}
/** Entries shown at once; the window follows the cursor. */
export const WINDOW = 12;
/** The picker: the filter line, a window of the matching entries with the cursor, the keys. */
export function pickerLines(title: string, entries: Pick[], q: string, idx: number, paint: Paint = plain): string[] {
	const shown = pickMatches(entries, q);
	const start = Math.max(0, Math.min(idx - Math.floor(WINDOW / 2), shown.length - WINDOW));
	const end = Math.min(shown.length, start + WINDOW);
	const rows = shown
		.slice(start, end)
		.map((e, k) => (start + k === idx ? paint.accent(`→ ${e.text}`) : `  ${e.text}`));
	return [
		paint.bold(title),
		`${paint.dim("  type to filter: ")}${q}_${paint.dim(`   ${shown.length} of ${entries.length}`)}`,
		"",
		...(start > 0 ? [paint.dim(`  … ${start} above`)] : []),
		...(shown.length ? rows : [paint.dim("  nothing matches; backspace erases")]),
		...(end < shown.length ? [paint.dim(`  … ${shown.length - end} below`)] : []),
		"",
		paint.dim("↑↓ move   enter select   esc cancel   type to filter"),
	];
}
