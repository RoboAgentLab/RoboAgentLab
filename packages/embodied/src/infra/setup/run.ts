/**
 * `/embodied-run`: start an episode of an experiment /embodied-setup installed. What is installed is
 * read, not asked: an experiment directory is one whose `.pi/settings.json` loads a robot extension
 * (the one /embodied-setup wrote, or any hand-made one), its setting is the `.pi/embodied-setup.json`
 * record /embodied-setup leaves there (robot, cell, ported setting, flags), and it is ready when the
 * services venv setup.sh built (`<services>/.venv-<robot>/pi-embodied.env`) exists. The GPUs come from
 * nvidia-smi. Nothing here asks or runs: ./index.ts asks for what is not recorded and starts pi.
 */

import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:net";
import { hostname } from "node:os";
import { basename, join, resolve } from "node:path";
import { type Cell, cellName, type Panel } from "./axes.ts";

type RobotLike = { id: string; label: string; kind: "sim" | "real"; extension: string; task: string };

/** What /embodied-setup records in the experiment directory (`.pi/embodied-setup.json`). */
export type SetupRecord = {
	robot: string;
	cell: Cell;
	/** The ported setting's id, when one was chosen. */
	paper?: string;
	/** The setting's flags (the cell's, or the ported setting's preset / tier, and the fine-tuned planner's). */
	flags: string;
	/** The robot's example task flags at setup time. */
	task: string;
	services: string;
	planner?: string;
	created: string;
};
export const RECORD = ".pi/embodied-setup.json";

export type Experiment<R extends RobotLike = RobotLike> = {
	dir: string;
	/** The robot the directory's settings load; with no record, the first of `robots` whose venv exists. */
	robot: R;
	/** The robots that share that extension (LIBERO's three), when no record says which. */
	robots: R[];
	record?: SetupRecord;
	services: string;
	/** The env file setup.sh wrote for the robot, when it exists: the experiment is ready. */
	env?: string;
	planner?: string;
};

const EXTENSION = /^(.*\/packages\/embodied)\/src\/robots\/([^/]+)\/index\.ts$/;

/** The experiment in a directory, when its settings load a robot extension. */
export function readExperiment<R extends RobotLike>(dir: string, robots: R[]): Experiment<R> | undefined {
	const file = join(dir, ".pi/settings.json");
	if (!existsSync(file)) return undefined;
	let settings: { extensions?: unknown[]; defaultProvider?: string; defaultModel?: string };
	try {
		settings = JSON.parse(readFileSync(file, "utf8"));
	} catch {
		return undefined;
	}
	for (const e of settings.extensions ?? []) {
		const m = typeof e === "string" ? EXTENSION.exec(e) : null;
		if (!m) continue;
		const candidates = robots.filter((r) => r.extension === `src/robots/${m[2]}/index.ts`);
		if (!candidates.length) continue;
		const services = resolve(m[1], "../../services");
		let record: SetupRecord | undefined;
		try {
			if (existsSync(join(dir, RECORD))) record = JSON.parse(readFileSync(join(dir, RECORD), "utf8"));
		} catch {
			record = undefined;
		}
		const envOf = (r: R) => {
			const p = join(record?.services ?? services, `.venv-${r.id}`, "pi-embodied.env");
			return existsSync(p) ? p : undefined;
		};
		const robot =
			(record && candidates.find((r) => r.id === record?.robot)) ?? candidates.find(envOf) ?? candidates[0];
		const planner =
			settings.defaultProvider && settings.defaultModel
				? `${settings.defaultProvider}/${settings.defaultModel}`
				: undefined;
		return {
			dir,
			robot,
			robots: candidates,
			record,
			services: record?.services ?? services,
			env: envOf(robot),
			planner,
		};
	}
	return undefined;
}

/** The experiments in a directory and its immediate subdirectories, the directory itself first. */
export function scanExperiments<R extends RobotLike>(cwd: string, robots: R[]): Experiment<R>[] {
	const dirs = [cwd];
	try {
		for (const e of readdirSync(cwd, { withFileTypes: true }))
			if (e.isDirectory() && !e.name.startsWith(".") && e.name !== "node_modules") dirs.push(join(cwd, e.name));
	} catch {
		// unreadable: the directory itself only
	}
	return dirs
		.filter((d) => {
			try {
				return statSync(d).isDirectory();
			} catch {
				return false;
			}
		})
		.map((d) => readExperiment(d, robots))
		.filter((e): e is Experiment<R> => e !== undefined);
}

/** One line per experiment for a menu: its directory, robot, setting and readiness. */
export function experimentLabel(e: Experiment): string {
	const setting = e.record
		? `${cellName(e.record.cell)}${e.record.paper ? `, ${e.record.paper}` : ""}`
		: "setting not recorded";
	return `${basename(e.dir)}: ${e.robot.label}, ${setting}${e.env ? "" : " (not installed)"}`;
}

export type Gpu = { index: number; freeMiB: number; totalMiB: number };
/** The GPUs nvidia-smi reports, the freest first; none without it. */
export function gpus(run: typeof spawnSync = spawnSync): Gpu[] {
	try {
		const r = run("nvidia-smi", ["--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], {
			encoding: "utf8",
			timeout: 5000,
		});
		if (r.status !== 0 || typeof r.stdout !== "string") return [];
		return r.stdout
			.trim()
			.split("\n")
			.map((line) => line.split(",").map((x) => Number(x.trim())))
			.filter((xs) => xs.length === 3 && xs.every((x) => Number.isFinite(x)))
			.map(([index, used, total]) => ({ index, freeMiB: total - used, totalMiB: total }))
			.sort((a, b) => b.freeMiB - a.freeMiB);
	} catch {
		return [];
	}
}
export const gpuLabel = (g: Gpu) => `GPU ${g.index}: ${g.freeMiB} MiB free of ${g.totalMiB}`;

/** The dashboard's fixed port, so a tunnel or an ssh LocalForward can be set up once. */
export const DASHBOARD_PORT = 8765;
/** The first free port at or after `start` on `host` (pi refuses a taken one). */
export function freePort(start = DASHBOARD_PORT, host = "127.0.0.1", tries = 20): Promise<number> {
	const probe = (port: number): Promise<boolean> =>
		new Promise((ok) => {
			const s = createServer();
			s.once("error", () => ok(false));
			s.listen(port, host, () => s.close(() => ok(true)));
		});
	return (async () => {
		for (let p = start; p < start + tries; p++) if (await probe(p)) return p;
		return start;
	})();
}
/** The panel flags of an episode: the dashboard on its port, and the chosen extra panels'. */
export const panelFlags = (port: number, panels: Panel[] = []) =>
	[`--dashboard=true --dashboard-port ${port}`, ...panels.map((p) => p.flags)].join(" ");
/** The episode's pi flags: the task's, the setting's, the budget, the panels'. */
export const episodeFlags = (task: string, setting: string, budget: string, panels = "--dashboard=true") =>
	[task, setting, budget, panels].filter(Boolean).join(" ");
/** The panels' URLs on the box, one per line. */
export const panelUrls = (port: number, panels: Panel[] = []) => [
	`dashboard http://127.0.0.1:${port}/`,
	...panels.map((p) => `${p.name.split(" (")[0]} http://127.0.0.1:${p.port}/`),
];
/**
 * When pi runs over ssh (SSH_CONNECTION is set), the command that brings the panels' ports to the user's
 * machine; a LocalForward of the same ports in their ssh config makes it unnecessary.
 */
export function tunnelHint(
	ports: number[],
	env: NodeJS.ProcessEnv = process.env,
	host = hostname(),
): string | undefined {
	if (!env.SSH_CONNECTION) return undefined;
	return `ssh -N ${ports.map((p) => `-L ${p}:127.0.0.1:${p}`).join(" ")} ${host}`;
}

/** The instruction every episode starts with; pi takes it as the first message (its positional `message` argument). */
export const PROMPT = "Solve the task.";
/**
 * The shell command of one interactive episode in the experiment directory, on one GPU when given: pi
 * starts on the instruction at once (after the trust prompt the first time) instead of waiting for it.
 */
export function runCommand(e: Experiment, flags: string, gpu?: number, prompt = PROMPT): string {
	const dev = gpu === undefined ? "" : `CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} `;
	return `cd ${e.dir} && source ${e.env} && ${dev}pi ${flags} ${JSON.stringify(prompt)}`;
}
/** The same episode without the terminal UI: trust is approved, the prompt is given, pi prints and exits. */
export function headlessCommand(e: Experiment, flags: string, gpu?: number, prompt = PROMPT): string {
	const dev = gpu === undefined ? "" : `CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} `;
	return `cd ${e.dir} && source ${e.env} && ${dev}pi -a -p ${flags} ${JSON.stringify(prompt)}`;
}
