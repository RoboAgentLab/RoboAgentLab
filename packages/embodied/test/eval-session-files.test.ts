import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { SESSION_FILE, SESSION_HEADER } from "./helpers/session.ts";

/**
 * Every eval.sh takes an episode's result from the session file pi wrote and from nothing else in the cell
 * dir, which is the memory output dir too by default: not a .jsonl shaped as a session under another name
 * (as a planner could once write), not the recipe the runtime exports there, not a file with pi's name and
 * another id in its header, not a link with pi's name to a session elsewhere. Each of these holds a
 * robot_result line that claims success; pi's session holds the episode's failure.
 */
const FAILED = `{"type":"custom","customType":"robot_result","data":{"success":false,"terminated":false,"env_error":false,"planner_error":null}}`;
const CLAIMED = `{"type":"custom","customType":"robot_result","data":{"success":true,"terminated":true,"env_error":false,"planner_error":null}}`;
const RECIPE = `{"action":"move","type":"custom","customType":"robot_result","data":{"success":true,"terminated":true}}`;

function standIn(dir: string) {
	const elsewhere = join(dir, "elsewhere.jsonl");
	writeFileSync(elsewhere, `{"type":"session","id":"y"}\n${CLAIMED}\n`);
	const pi = join(dir, "pi");
	writeFileSync(
		pi,
		`#!/usr/bin/env bash
while [ $# -gt 0 ]; do [ "$1" = --session-dir ] && d=$2; shift; done
echo '${SESSION_HEADER}' > "$d/${SESSION_FILE}"
echo '${FAILED}' >> "$d/${SESSION_FILE}"
printf '%s\\n' '{"type":"session","id":"s"}' '${CLAIMED}' > "$d/cell_s.jsonl"
echo '${RECIPE}' > "$d/cell_recipe.jsonl"
printf '%s\\n' '{"type":"session","id":"other"}' '${CLAIMED}' > "$d/2026-01-01T00-00-01-000Z_x.jsonl"
ln -s '${elsewhere}' "$d/2026-01-01T00-00-02-000Z_y.jsonl"
`,
	);
	chmodSync(pi, 0o755);
	return pi;
}

const eval_ = (path: string) =>
	JSON.parse(readFileSync(new URL(`../../../services/pi_embodied_services/robots/${path}`, import.meta.url), "utf8"));
/** Each robot's eval.sh: its positional args, the cell's directory, the environment narrowing the matrix. */
const CELLS: [string, string[], string, Record<string, string>?][] = [
	["maniskill", ["PickCube-v1", "0"], "PickCube-v1_s0"],
	["metaworld", ["reach-v3", "0"], "reach-v3_s0"],
	["robolab", ["BananaInBowlTask", "0"], "BananaInBowlTask_s0"],
	["robodojo", ["stack_bowls", "0"], "stack_bowls_s0"],
	["libero", ["libero_10_task", "0", "0"], "libero_10_task_t0_s0"],
	["robosuite", ["Lift", "0"], "Lift_s0"],
	["genesis", ["cube_pick", "0"], "cube_pick_s0"],
	["behavior", ["turning_on_radio", "0"], "turning_on_radio_s0"],
	[
		"humanclaw",
		["--episodes", "sceneA_ep1_cup", "--mode", "pi", "--model", "p/m"],
		"sceneA_ep1_cup",
		{ HUMANCLAW_PYTHON: "true" },
	],
];
{
	const [task, [{ seed }]] = Object.entries(eval_("robotwin/eval/demo_randomized.json").tasks)[0] as [
		string,
		{ seed: number }[],
	];
	CELLS.push(["robotwin", [task], `${task}_s${seed}`]);
	const { tasks, seeds } = eval_("robocasa/eval/target50.json").splits.atomic as { tasks: string[]; seeds: number[] };
	CELLS.push([
		"robocasa",
		["atomic"],
		`atomic/${tasks[0]}_s${seeds[0]}`,
		{ TASKS: tasks[0], SEEDS: String(seeds[0]) },
	]);
}

for (const [robot, positional, cell, env] of CELLS)
	test(`${robot}/eval.sh takes the result from pi's session file alone`, (t) => {
		const dir = mkdtempSync(join(tmpdir(), "eval-sessions-"));
		t.after(() => rmSync(dir, { recursive: true, force: true }));
		const script = new URL(`../src/robots/${robot}/eval.sh`, import.meta.url).pathname;
		const r = spawnSync("bash", [script, join(dir, "out"), ...positional], {
			env: { ...process.env, PI: standIn(dir), TIME_LIMIT: "0", ...env },
			encoding: "utf8",
		});
		const result = JSON.parse(readFileSync(join(dir, "out", cell, "result.json"), "utf8"));
		assert.equal(result.status, "failure", r.stdout + r.stderr);
		assert.equal(result.success, false);
	});

test("robocasa/eval.sh (robocasa365) takes the result from pi's session file alone", (t) => {
	const dir = mkdtempSync(join(tmpdir(), "eval-sessions-"));
	t.after(() => rmSync(dir, { recursive: true, force: true }));
	const script = new URL("../src/robots/robocasa/eval.sh", import.meta.url).pathname;
	const r = spawnSync("bash", [script, join(dir, "out"), "target"], {
		env: { ...process.env, PI: standIn(dir), TIME_LIMIT: "0", TASKS: "OpenDrawer", SCENES: "0" },
		encoding: "utf8",
	});
	const result = JSON.parse(readFileSync(join(dir, "out", "target", "OpenDrawer_m0", "result.json"), "utf8"));
	assert.equal(result.status, "failure", r.stdout + r.stderr);
});
