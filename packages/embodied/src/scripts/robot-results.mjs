#!/usr/bin/env node
/**
 * robot-results.mjs <dir>: the robot_result entries (their data) of the session files pi wrote in <dir>, as a JSON
 * array on stdout. Every robot's eval.sh records an episode's result from it; the cell's other .jsonl files, the
 * recipe among them, are not read (../infra/session-files.ts).
 */
import { readFileSync } from "node:fs";
import { sessionFiles } from "../infra/session-files.ts";

const results = [];
for (const file of sessionFiles(process.argv[2]))
	for (const line of readFileSync(file, "utf8").split("\n")) {
		if (!line.includes('"robot_result"')) continue;
		const e = JSON.parse(line);
		if (e.type === "custom" && e.customType === "robot_result") results.push(e.data);
	}
console.log(JSON.stringify(results));
