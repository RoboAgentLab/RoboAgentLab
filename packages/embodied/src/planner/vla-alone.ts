/**
 * The VLA-alone baseline (`--vla-alone <adapter>`): a mounted VLA adapter (../primitives/vla-adapters.ts)
 * runs the episode in a closed loop on the task instruction instead of the planner, the way
 * `--code-oracle` runs a reference program (../modes/code). The first prompt starts it and nothing is
 * sent to the model; the robot executes the policy's chunks until the task succeeds, the episode ends or
 * `--vla-alone-steps` env steps are spent (0: the task's registered episode length, the robot's
 * default). The result is an ordinary `robot_result` with `planner: "vla"` and the run under
 * `vla_alone`; a run that broke (an unreachable server, a refused checkpoint) is an env_error.
 *
 *   pi -p -e packages/embodied/src/robots/maniskill --arm widowx250s --task PutCarrotOnPlateInScene-v1 \
 *      --vla-adapter openvla --vla-alone openvla "Solve the task."
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { message } from "../robot.ts";
import type { Planner } from "./kind.ts";

/** What the robot supplies: the run itself and the base's hooks (../robot.ts). */
export type VlaAloneSpec = {
	/** Whether the robot is up and unbroken. */
	ready: () => boolean;
	/** The adapters this run mounted (`--vla-adapter` names), read at the first prompt. */
	mounted: () => readonly string[];
	/** Run `adapter` alone for at most `steps` env steps (0: the robot's default); returns the run's record. */
	run: (adapter: string, steps: number) => Promise<Record<string, unknown>>;
	/** The base's `ranWithoutPlanner`. */
	ran: (by: Planner, error?: string) => void;
};

/** Register `--vla-alone` and `--vla-alone-steps`; the result's `vla_alone` and the start-time refusal. */
export function vlaAlone(pi: ExtensionAPI, spec: VlaAloneSpec) {
	pi.registerFlag("vla-alone", {
		type: "string",
		default: "",
		description:
			"Run this mounted VLA adapter (a --vla-adapter name) alone in a closed loop on the task instead of the planner: the VLA-alone baseline (planner: vla)",
	});
	pi.registerFlag("vla-alone-steps", {
		type: "string",
		default: "0",
		description: "--vla-alone: the env-step budget (0 = the task's registered episode length)",
	});
	const adapter = () => String(pi.getFlag("vla-alone") ?? "").trim();
	let record: Record<string, unknown> | undefined;
	pi.on("session_start", () => {
		record = undefined;
	});
	pi.on("input", async (_event, ctx) => {
		const name = adapter();
		if (!name || !spec.ready()) return undefined;
		if (record) return { action: "handled" as const };
		record = { adapter: name };
		let error: string | undefined;
		try {
			record = { adapter: name, ...(await spec.run(name, Number(pi.getFlag("vla-alone-steps")) || 0)) };
		} catch (err) {
			error = message(err);
			record = { adapter: name, error };
		}
		spec.ran("vla", error);
		const summary = `vla-alone ${name}: ${error ?? JSON.stringify(record)}`;
		if (ctx.hasUI) ctx.ui.notify(summary, error ? "warning" : "info");
		else console.error(`[vla-alone] ${summary.slice(0, 2000)}`);
		return { action: "handled" as const };
	});
	return {
		/** Why the robot must not start with these flags (the adapter is not mounted), else undefined. */
		configError(): string | undefined {
			const name = adapter();
			if (!name || spec.mounted().includes(name)) return undefined;
			return `--vla-alone ${name} runs a mounted adapter: add it to --vla-adapter`;
		},
		/** The result's `vla_alone` (the adapter, the run's steps and success), once it ran. */
		result: () => (record ? { vla_alone: record } : {}),
	};
}
