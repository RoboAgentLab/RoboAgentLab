/**
 * The survey's Agent-as-Policy taxonomy as /embodied-setup's grid: observations (O1 agent-built
 * perception, O2 supplied perception, O3 privileged state) by execution authority (A1 behavior
 * selection, A2 tool orchestration, A3 code synthesis). A robot supports a level when it has the
 * flag for it (`Axes`); O1 and A2 are every robot's defaults and add no flag. A paper's setting
 * (`PAPERS`) is a `--preset` or `--tier` that sits in one cell, offered after the cell is chosen.
 *
 * The grid is a pi custom component (../setup/index.ts): ./gridMove and ./gridLines are its
 * keyboard and rendering, pure so the test drives them without a terminal.
 */

export type O = "O1" | "O2" | "O3";
export type A = "A1" | "A2" | "A3";
export type Cell = { o: O; a: A };
export const OS: O[] = ["O1", "O2", "O3"];
export const AS: A[] = ["A1", "A2", "A3"];

type Level = { short: string; name: string; what: string };
export const OBSERVATION: Record<O, Level> = {
	O1: {
		short: "cameras",
		name: "O1 agent-built perception",
		what: "camera frames only; the planner reads the scene itself",
	},
	O2: {
		short: "perception tools",
		name: "O2 supplied perception",
		what: "SAM3 masks with ids next to the frames (detect, select_detection, reject_detection; the arms' segment)",
	},
	O3: {
		short: "privileged",
		name: "O3 privileged state",
		what: "the simulator's ground-truth object poses (ground_truth_poses); the result is marked privileged",
	},
};
export const EXECUTION: Record<A, Level> = {
	A1: {
		short: "units",
		name: "A1 behavior selection",
		what: "Show-Harness action units (act: move, rotate, grasp, release) run by a fixed executor; no VLA",
	},
	A2: {
		short: "tools",
		name: "A2 tool orchestration",
		what: "the robot's own tools (VLA, segmentation, motion) composed by the planner",
	},
	A3: {
		short: "code",
		name: "A3 code synthesis",
		what: "run_code: Python over the robot's primitives (CaP-X) with conditions, loops and computation",
	},
};

/** The flag a level adds on this robot; absent = the robot has no such level. O1 and A2 are the defaults. */
export type Axes = { O2?: string; O3?: string; A1?: string; A3?: string };

export const has = (axes: Axes, level: O | A): boolean => level === "O1" || level === "A2" || axes[level] !== undefined;
export const supported = (axes: Axes, c: Cell): boolean => has(axes, c.o) && has(axes, c.a);
/** Row-major (O1 first), the order the grid cursor starts from. */
export const supportedCells = (axes: Axes): Cell[] =>
	OS.flatMap((o) => AS.map((a) => ({ o, a }))).filter((c) => supported(axes, c));
export const cellName = (c: Cell) => `${c.o} × ${c.a}`;
export const cellLabel = (c: Cell) => `${cellName(c)}: ${OBSERVATION[c.o].short} · ${EXECUTION[c.a].short}`;
/** The launch flags of a cell: the execution level's, then the observation level's. */
export const cellFlags = (axes: Axes, c: Cell): string =>
	[c.a === "A2" ? undefined : axes[c.a], c.o === "O1" ? undefined : axes[c.o]].filter(Boolean).join(" ");

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
const ROW = 20;
const COL = 14;
const pad = (s: string, n: number) => s + " ".repeat(Math.max(0, n - [...s].length));

/** The grid, the cursor's cell explained, and the keys. */
export function gridLines(title: string, axes: Axes, cur: Cell, paint: Paint = plain): string[] {
	const head = pad("", ROW) + AS.map((a) => pad(`${a} ${EXECUTION[a].short}`, COL)).join("");
	const rows = OS.map(
		(o) =>
			pad(`${o} ${OBSERVATION[o].short}`, ROW) +
			AS.map((a) => {
				const c = { o, a };
				if (!supported(axes, c)) return paint.dim(pad(" –", COL));
				return c.o === cur.o && c.a === cur.a ? paint.accent(paint.bold(pad("[●]", COL))) : pad("[ ]", COL);
			}).join(""),
	);
	const flags = cellFlags(axes, cur);
	return [
		paint.bold(title),
		"",
		paint.dim(head),
		...rows,
		"",
		paint.bold(`${cellName(cur)}: ${OBSERVATION[cur.o].name} × ${EXECUTION[cur.a].name}`),
		`  ${OBSERVATION[cur.o].what}`,
		`  ${EXECUTION[cur.a].what}`,
		paint.dim(`  flags: ${flags || "none (the robot's defaults)"}`),
		"",
		paint.dim("←↑↓→ move   enter select   esc cancel   – not on this robot"),
	];
}

export type Key = Dir | "enter" | "escape";
/** Raw terminal input the grid understands besides the configured select keybindings: arrows and vi keys. */
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

/** The paper picker's entries: none first, then the papers of the cell. */
export type Pick = { id: string; text: string; paper?: Paper };
export const pickEntries = (papers: Paper[], none: string): Pick[] => [
	{ id: "none", text: none },
	...papers.map((p) => ({ id: p.id, text: `${p.id}: ${p.name}, ${p.note}`, paper: p })),
];
/** The entries a typed filter keeps: the short name or the paper's name contains it (case-insensitive). */
export function pickMatches(entries: Pick[], q: string): Pick[] {
	const s = q.trim().toLowerCase();
	if (!s) return entries;
	return entries.filter((e) => e.id.toLowerCase().includes(s) || (e.paper?.name.toLowerCase().includes(s) ?? false));
}
/** The picker: the filter line, the matching entries with the cursor, the keys. */
export function pickerLines(title: string, entries: Pick[], q: string, idx: number, paint: Paint = plain): string[] {
	const shown = pickMatches(entries, q);
	return [
		paint.bold(title),
		`${paint.dim("  type to filter: ")}${q}_`,
		"",
		...(shown.length
			? shown.map((e, i) => (i === idx ? paint.accent(`→ ${e.text}`) : `  ${e.text}`))
			: [paint.dim("  nothing matches; backspace erases")]),
		"",
		paint.dim("↑↓ move   enter select   esc cancel   type to filter"),
	];
}

/**
 * A paper's setting: the `--preset` / `--tier` that reproduces a ported repository's configuration
 * (../tiers.ts expands it; a flag it sets is not given again), in the cell it occupies, on the robots
 * that have what it needs (`robots`: id prefixes; absent = any robot that has the cell). `fixesO`:
 * the tier sets --privileged itself (CaP-X S tiers), so the cell's O flag is dropped.
 */
export type Paper = {
	id: string;
	name: string;
	cell: Cell;
	flag: string;
	note: string;
	robots?: string[];
	fixesO?: boolean;
};
const capx = (tier: string, o: O, note: string): Paper => ({
	id: `capx-${tier}`,
	name: `CaP-X ${tier}`,
	cell: { o, a: "A3" },
	flag: `--tier ${tier}`,
	note,
	fixesO: tier.startsWith("S"),
});
export const PAPERS: Paper[] = [
	{
		id: "showharness",
		name: "Show-Harness",
		cell: { o: "O1", a: "A1" },
		flag: "--preset showharness",
		note: "action units with image feedback, no VLA",
	},
	{
		id: "motormind",
		name: "MotorMind",
		cell: { o: "O1", a: "A1" },
		flag: "--preset motormind",
		note: "LIBERO side camera, settle ticks, no cross-task memory",
		robots: ["libero"],
	},
	{
		id: "rpent",
		name: "RPent",
		cell: { o: "O1", a: "A2" },
		flag: "--preset rpent",
		note: "tools with memory and a VLA skill",
		robots: ["libero", "robocasa", "franka", "dual-franka"],
	},
	{
		id: "openeta",
		name: "OpenETA",
		cell: { o: "O1", a: "A2" },
		flag: "--preset openeta",
		note: "tools with visual differencing (VDM) and the anchor image",
	},
	{
		id: "humanclaw",
		name: "HumanCLAW",
		cell: { o: "O1", a: "A2" },
		flag: "--preset humanclaw",
		note: "the paper's mode: units next to the tools",
		robots: ["humanclaw"],
	},
	capx("S1", "O3", "one program, high-level API, simulator ground truth"),
	capx("S2", "O1", "one program, high-level API"),
	capx("S3", "O1", "one program, low-level API with examples"),
	capx("S4", "O1", "one program, low-level API, no examples"),
	capx("M1", "O1", "multi-turn, text feedback, high-level API"),
	capx("M2", "O1", "multi-turn, image feedback, high-level API"),
	capx("M3", "O1", "multi-turn, VDM feedback, high-level API"),
	capx("M4", "O1", "multi-turn, VDM feedback, low-level API"),
];
/** The papers whose setting is this cell on this robot (one that has the cell and what the paper needs). */
export const papersFor = (robot: { id: string; axes: Axes }, c: Cell): Paper[] =>
	PAPERS.filter(
		(p) =>
			p.cell.o === c.o &&
			p.cell.a === c.a &&
			supported(robot.axes, c) &&
			(!p.robots || p.robots.some((r) => robot.id.startsWith(r))),
	);
/** The launch flags of a cell under a paper's setting: the paper's flag for the cell's A flag, the O flag unless the paper fixes it. */
export const paperFlags = (axes: Axes, c: Cell, p: Paper): string =>
	[p.flag, p.fixesO || c.o === "O1" ? undefined : axes[c.o]].filter(Boolean).join(" ");
