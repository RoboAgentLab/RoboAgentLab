/**
 * Third-party VLA policies as robot tools. The services' `openvla_server`, `openvla_oft_server` and
 * `gr00t_server` (../../../services/pi_embodied_services/components) all speak the Pi0.5 wire format
 * and answer in their checkpoint's action space (`vla.info`'s `action_space`). `--vla-adapter <name>`
 * mounts `<name>_act` on a robot the adapter declares an embodiment for (endpoint: services.<name>);
 * nothing is mounted when it is off, and `serve.sh` starts the servers.
 *
 * An embodiment says what the adapter needs of that robot: the dataset statistics its checkpoint
 * must un-normalise with (`suites`, checked against `vla.info` at the first call), the action space
 * the robot's env server executes, and which of the robot's cameras feed the wire's main and wrist
 * images. LIBERO runs the actions as they come (`libero_osc`): `<name>_act` is a closed-loop grasp
 * judged by the pick heuristics, as `pi0_pick` is. ManiSkill's bridge twins take OpenVLA's Bridge V2
 * actions (`bridge_ee_delta_euler`), which its env server translates (`env.vla_step`): there
 * `<name>_act` runs action chunks on the task and returns the new observation.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { requireService } from "../infra/config.ts";
import { RpcClient } from "../infra/rpc.ts";

/** What an adapter needs of one robot (see the module comment). */
export type VlaEmbodiment = {
	/** The un-normalisation datasets (`vla.info`'s `suite`) whose actions this robot can execute. */
	suites: readonly string[];
	/** The action space the adapter must answer in (`vla.info`'s `action_space`). */
	action: "libero_osc" | "bridge_ee_delta_euler";
	/** The robot's cameras behind the wire's `main_images` and (absent: none sent) `wrist_images`. */
	cameras: { main: string; wrist?: string };
};
export type VlaAdapter = {
	tool: string;
	flag: string;
	service: "openvla" | "openvla_oft" | "gr00t";
	/** LIBERO's grasp-tool description of the model (`pickDescription`). */
	model: string;
	/** The robots that can mount it, by robot name. */
	embodiments: Partial<Record<string, VlaEmbodiment>>;
};
/** The LIBERO suites the published OpenVLA / OpenVLA-OFT fine-tunes were trained on (their servers' `--suite`). */
const FINE_TUNED_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] as const;
/** LIBERO's embodiment: either camera, the env's own OSC actions; the episode's suite decides (`suiteMismatch`). */
const LIBERO: VlaEmbodiment = {
	suites: FINE_TUNED_SUITES,
	action: "libero_osc",
	cameras: { main: "agentview", wrist: "wrist" },
};
export const VLA_ADAPTERS: readonly VlaAdapter[] = [
	{
		tool: "openvla_act",
		flag: "openvla",
		service: "openvla",
		model: "OpenVLA (7B, LIBERO fine-tune; one action per call)",
		embodiments: {
			libero: LIBERO,
			// The base openvla-7b with its Bridge V2 statistics (its server's --suite bridge_orig) on the
			// bridge twins' WidowX 250 S: the scene's own Bridge camera, no wrist camera (SimplerEnv's setup).
			maniskill: { suites: ["bridge_orig"], action: "bridge_ee_delta_euler", cameras: { main: "agentview" } },
		},
	},
	{
		tool: "openvla_oft_act",
		flag: "openvla-oft",
		service: "openvla_oft",
		model: "OpenVLA-OFT (7B, LIBERO fine-tune; 8-step chunks)",
		embodiments: { libero: LIBERO },
	},
	{
		tool: "gr00t_act",
		flag: "gr00t",
		service: "gr00t",
		model: "GR00T N1.6/N1.7 (LIBERO libero_panda fine-tune; 16-step chunks)",
		embodiments: { libero: LIBERO },
	},
];

/** The adapters that declare an embodiment for `robot`: the tools it registers at load. */
export const adaptersFor = (robot: string) => VLA_ADAPTERS.filter((a) => a.embodiments[robot]);

/** Register `--vla-adapter` on a robot that can mount adapters; returns the names it lists (read at start). */
export function registerVlaAdapterFlag(pi: ExtensionAPI, robot: string): () => string[] {
	pi.registerFlag("vla-adapter", {
		type: "string",
		default: "",
		description: `Third-party VLAs to mount, comma-separated: ${adaptersFor(robot)
			.map((a) => a.flag)
			.join(" | ")} (endpoints: services.<name> of the deployment config)`,
	});
	return () =>
		String(pi.getFlag("vla-adapter") ?? "")
			.split(",")
			.map((x) => x.trim())
			.filter(Boolean);
}

/** One adapter a run mounted: its client on services.<name> and the robot's embodiment. */
export type MountedAdapter = { adapter: VlaAdapter; embodiment: VlaEmbodiment; client: RpcClient };

/**
 * The adapters `names` lists, mounted on `robot` by tool name. Throws for a name that is no adapter,
 * one without an embodiment for `robot`, or one whose endpoint the deployment does not configure.
 */
export function mountAdapters(pi: ExtensionAPI, names: readonly string[], robot: string): Map<string, MountedAdapter> {
	const out = new Map<string, MountedAdapter>();
	for (const name of names) {
		const adapter = VLA_ADAPTERS.find((x) => x.flag === name);
		if (!adapter) throw new Error(`--vla-adapter ${name}: not one of ${VLA_ADAPTERS.map((x) => x.flag).join(", ")}`);
		const embodiment = adapter.embodiments[robot];
		if (!embodiment) {
			const can = adaptersFor(robot).map((x) => x.flag);
			throw new Error(
				`--vla-adapter ${name} has no ${robot} embodiment (its checkpoints run ${Object.keys(adapter.embodiments).join(", ")}); ${robot} mounts ${can.join(", ") || "none"}`,
			);
		}
		const s = requireService(pi, adapter.service, `--vla-adapter ${name}`);
		if ("error" in s) throw new Error(s.error);
		out.set(adapter.tool, { adapter, embodiment, client: new RpcClient(s.url) });
	}
	return out;
}

/** The `pi0_pick` parameters, shared by every closed-loop grasp tool. */
export const PICK_PARAMETERS = Type.Object({
	prompt: Type.String({ description: "VLA instruction, e.g. 'pick up the black bowl'" }),
	max_chunks: Type.Optional(Type.Integer({ description: "Action-chunk budget (default 24)" })),
	lift_thresh: Type.Optional(Type.Number({ description: "Post-descent ascent for success, m (default 0.05)" })),
	gripper_closed_thresh: Type.Optional(
		Type.Number({ description: "Finger separation below which the gripper counts as closed (default 0.06)" }),
	),
	gripper_open_thresh: Type.Optional(
		Type.Number({ description: "Minimum finger separation accepted as holding (default 0.0)" }),
	),
	descent_thresh: Type.Optional(
		Type.Number({ description: "Required descent before lift detection, m (default 0.10)" }),
	),
});
export type PickParams = {
	prompt: string;
	max_chunks?: number;
	lift_thresh?: number;
	gripper_closed_thresh?: number;
	gripper_open_thresh?: number;
	descent_thresh?: number;
};

/** What a `<vla>_act` tool says: a closed-loop grasp with `model`, judged by the pick heuristics. */
export const pickDescription = (model: string) =>
	`${model} closed-loop grasp, run as action chunks until the pick heuristics say the object is lifted or the budget ends. Use it only for the grasp; you do every move_to and release. Success needs a descent then a lift with the gripper partly closed; it is a hint, confirm from gripper opening and the wrist image.`;

/**
 * The pick heuristics of `pi0_pick`, as a tracker fed after every chunk: success is a descent of
 * `descent_thresh`, then a lift of `lift_thresh` above the lowest point, with the gripper between
 * the open and closed thresholds.
 */
export function pickTracker(startZ: number, startGrip: number, p: PickParams) {
	const lift = p.lift_thresh ?? 0.05;
	const closed = p.gripper_closed_thresh ?? 0.06;
	const open = p.gripper_open_thresh ?? 0;
	const descent = p.descent_thresh ?? 0.1;
	let minZ = startZ;
	let postMinPeak = startZ;
	let minGrip = startGrip;
	return {
		/** Feed the state after a chunk; true once the grasp counts as a success. */
		update(z: number, grip: number): boolean {
			if (z < minZ) {
				minZ = z;
				postMinPeak = z;
			} else postMinPeak = Math.max(postMinPeak, z);
			minGrip = Math.min(minGrip, grip);
			return startZ - minZ >= descent && postMinPeak - minZ >= lift && grip >= open && grip < closed;
		},
		/** The measured motion, for the tool result. */
		summary: () => ({
			peak_lift_m: postMinPeak - minZ,
			descent_m: startZ - minZ,
			min_gripper_opening: minGrip,
		}),
	};
}

/** What a VLA server says of itself: its healthz name and, for the adapters, `vla.info`. */
export type VlaInfo = {
	service: string;
	model?: string;
	revision?: string | null;
	suite?: string | null;
	/** OpenVLA-OFT: the dataset statistics it un-normalises with (libero_all reports the key's suite). */
	unnorm_key?: string | null;
	/** The action space its chunks are in (services' vla_adapter_base); a server from before it answers LIBERO's. */
	action_space?: string;
	/** Pi0.5: the embodiment preset its server was started with (`--embodiment`: libero, metaworld, ...). */
	embodiment?: string;
};

/**
 * Which VLA answers on `client`: the server's healthz name and, for the adapters, `vla.info`
 * (`model`, `revision`, the `suite` its actions are un-normalised with, its `action_space`).
 * Pi0.5 answers its `embodiment` only; RLDX (and a Pi0.5 server from before it) has no `vla.info`, and that
 * error leaves the healthz name alone.
 */
export async function vlaInfo(client: RpcClient): Promise<VlaInfo> {
	const health = await client.call<{ service?: string }>("healthz", {}, 10_000);
	const service = health.service ?? "unknown";
	try {
		const info = await client.call<Omit<VlaInfo, "service">>("vla.info", {}, 10_000);
		return {
			service,
			model: info.model,
			revision: info.revision,
			suite: info.suite,
			...(info.action_space ? { action_space: info.action_space } : {}),
			...(info.embodiment ? { embodiment: info.embodiment } : {}),
		};
	} catch {
		return { service };
	}
}

/** `info` for the episode's `robot_result`: the healthz name and `model@revision`. */
export const vlaIdentity = (info: VlaInfo) =>
	`${info.service} ${info.model ?? ""}${info.revision ? `@${info.revision}` : ""}`.trim();

/**
 * The LIBERO suite whose fine-tune applies to an episode suite: a LIBERO-Pro perturbation
 * (`libero_10_task`, `libero_goal_swap`, `libero_object_lan`, `libero_spatial_object`, ...) or a
 * LIBERO-plus variant keeps its base suite's scenes and tasks, so the base decides. Unknown names
 * pass through.
 */
export function baseSuite(suite: string): string {
	const m = /^libero_(spatial|object|goal|10|90)(?:_|$)/.exec(suite);
	return m ? `libero_${m[1]}` : suite;
}

/**
 * Why a VLA fine-tuned on one LIBERO suite must not run on `suite`: the adapters report the
 * checkpoint's suite (OpenVLA-OFT's `libero_all` mixture reports the suite of its `--unnorm-key`,
 * whose statistics it un-normalises with; a custom checkpoint reports none and is trusted), compared
 * with the episode's base suite. Undefined when the checkpoint fits.
 */
export function suiteMismatch(info: VlaInfo, suite: string): string | undefined {
	const base = baseSuite(suite);
	if (!info.suite || info.suite === base) return undefined;
	const episode = base === suite ? suite : `${suite} (base suite ${base})`;
	const fix = (FINE_TUNED_SUITES as readonly string[]).includes(base)
		? `restart its server with --suite ${base}${info.service === "openvla-oft" ? ` or --suite libero_all --unnorm-key ${base}_no_noops` : ""}, or --model-path a checkpoint fine-tuned on ${base},`
		: `no published fine-tune covers ${base}; restart its server with --model-path a checkpoint fine-tuned on it`;
	return `${vlaIdentity(info)} is the ${info.suite} fine-tune, but this episode is ${episode}: ${fix} before using its grasp tool.`;
}

/**
 * Why the checkpoint on `info` cannot drive `robot` through `m` (a robot without per-episode suites:
 * LIBERO uses `suiteMismatch`): it answers in another action space, or un-normalises with another
 * dataset's statistics. A custom checkpoint (no suite) is trusted on its suite, not on its action
 * space. Undefined when it fits.
 */
export function embodimentMismatch(info: VlaInfo, m: MountedAdapter, robot: string): string | undefined {
	const { suites, action } = m.embodiment;
	const space = info.action_space ?? "libero_osc";
	const fix = `restart it with --suite ${suites[0]}`;
	if (space !== action)
		return `${vlaIdentity(info)} answers in ${space}, but ${robot} executes ${action}: ${fix} before using ${m.adapter.tool}.`;
	if (info.suite && !suites.includes(info.suite))
		return `${vlaIdentity(info)} un-normalises with ${info.suite}, but ${robot} runs ${suites.join(" or ")}: ${fix} before using ${m.adapter.tool}.`;
	return undefined;
}
