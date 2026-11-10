/**
 * The VLA-alone baseline (`--vla-alone <vla>`): a mounted VLA adapter (../primitives/vla-adapters.ts) or
 * the robot's own VLA skill (../capabilities/skills.ts: LIBERO's `pi0`, RoboCasa's `rldx`, RoboTwin's
 * `lingbot`) runs the episode in a closed loop on the task instruction instead of the planner, the way
 * `--code-oracle` runs a reference program (../modes/code). The first prompt starts it and nothing is
 * sent to the model; the robot executes the policy's chunks until the task succeeds, the episode ends or
 * `--vla-alone-steps` env steps are spent (0: the task's registered episode length, the robot's
 * default). The result is an ordinary `robot_result` with `planner: "vla"` and the run under
 * `vla_alone` (`adapter` or `skill`: the name); a run that broke (an unreachable server, a skill that was
 * off at start, a refused checkpoint) is an env_error.
 *
 *   pi -p -e packages/embodied/src/robots/maniskill --arm widowx250s --task PutCarrotOnPlateInScene-v1 \
 *      --vla-adapter openvla --vla-alone openvla "Solve the task."
 *   pi -p -e packages/embodied/src/robots/libero --suite libero_spatial --task 0 --vla-alone pi0 "Solve the task."
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { SkillState } from "../capabilities/skills.ts";
import { message } from "../robot.ts";
import type { Planner } from "./kind.ts";

/** What the robot supplies: the run itself and the base's hooks (../robot.ts). */
export type VlaAloneSpec = {
	/** Whether the robot is up and unbroken. */
	ready: () => boolean;
	/** The adapters this run mounted (`--vla-adapter` names), read at the first prompt. */
	mounted: () => readonly string[];
	/** The robot's own VLA skills (`--require-skills` names), which run without --vla-adapter. */
	skills?: readonly string[];
	/** Run `vla` (an adapter or a skill) alone for at most `steps` env steps (0: the robot's default); returns the run's record. */
	run: (vla: string, steps: number) => Promise<Record<string, unknown>>;
	/** The base's `ranWithoutPlanner`. */
	ran: (by: Planner, error?: string) => void;
};

/** Register `--vla-alone` and `--vla-alone-steps`; the result's `vla_alone` and the start-time refusal. */
export function vlaAlone(pi: ExtensionAPI, spec: VlaAloneSpec) {
	pi.registerFlag("vla-alone", {
		type: "string",
		default: "",
		description:
			"Run this VLA alone in a closed loop on the task instead of the planner, a mounted --vla-adapter name or the robot's own VLA skill (pi0, rldx, lingbot): the VLA-alone baseline (planner: vla)",
	});
	pi.registerFlag("vla-alone-steps", {
		type: "string",
		default: "0",
		description: "--vla-alone: the env-step budget (0 = the robot's default, the task's episode length)",
	});
	const named = () => String(pi.getFlag("vla-alone") ?? "").trim();
	const skills = spec.skills ?? [];
	/** The record's key for the name: a skill's or an adapter's. */
	const key = (name: string) => (skills.includes(name) ? "skill" : "adapter");
	let record: Record<string, unknown> | undefined;
	pi.on("session_start", () => {
		record = undefined;
	});
	pi.on("input", async (_event, ctx) => {
		const name = named();
		if (!name || !spec.ready()) return undefined;
		if (record) return { action: "handled" as const };
		record = { [key(name)]: name };
		let error: string | undefined;
		try {
			record = { [key(name)]: name, ...(await spec.run(name, Number(pi.getFlag("vla-alone-steps")) || 0)) };
		} catch (err) {
			error = message(err);
			record = { [key(name)]: name, error };
		}
		spec.ran("vla", error);
		const summary = `vla-alone ${name}: ${error ?? JSON.stringify(record)}`;
		if (ctx.hasUI) ctx.ui.notify(summary, error ? "warning" : "info");
		else console.error(`[vla-alone] ${summary.slice(0, 2000)}`);
		return { action: "handled" as const };
	});
	return {
		/**
		 * Why the robot must not start with these flags (the adapter is not mounted; the skill's server,
		 * `states` once probed, was off), else undefined.
		 */
		configError(states: Record<string, SkillState | undefined> = {}): string | undefined {
			const name = named();
			if (!name || spec.mounted().includes(name)) return undefined;
			if (skills.includes(name)) {
				const s = states[name];
				return s && !s.on ? `--vla-alone ${name}: the ${name} server is ${s.reason ?? "off"}` : undefined;
			}
			const own = skills.length ? ` or this robot's VLA skill (${skills.join(", ")})` : "";
			return `--vla-alone ${name} runs a mounted adapter${own}: add it to --vla-adapter`;
		},
		/** The result's `vla_alone` (the adapter or skill, the run's steps and success), once it ran. */
		result: () => (record ? { vla_alone: record } : {}),
	};
}
