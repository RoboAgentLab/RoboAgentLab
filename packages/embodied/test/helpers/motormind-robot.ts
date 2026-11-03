/**
 * MotorMind mode on a robot whose env server serves it through services utils/motormind.py
 * (Metaworld, robosuite, RoboCasa, RoboTwin): a fake env server answering `env.mm_observe` and
 * `env.mm_run_step` in that wire format (every reply carries the episode's `env_steps` and latched
 * `terminated`; a segment carries its video `frames`), the robot's own start calls from `base`, and
 * a stub pi session that loads the robot, starts it and runs its tools and `finish`.
 */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { IDENTITY } from "../../src/modes/motormind/frames.ts";
import { type Caps, MOTORMIND_EVENT, type MotorMindHandle } from "../../src/modes/motormind/index.ts";
import { RESULT_ENTRY } from "../../src/robot.ts";
import { type Call, fakeEnv, nd } from "../sim-stub.ts";
import { deployFlags } from "./deployment.ts";

type Json = Record<string, any>;
type Handler = (event: any, ctx: any) => unknown;

/** A capability table as utils/motormind.py `caps` builds it; `o` overrides the robot's fields. */
export function mmCaps(o: Partial<Caps> = {}): Caps {
	return {
		max_translation_m: 0.05,
		max_rotation_rad: 0.5,
		max_speed_m_s: 1.0,
		max_rotation_speed_rad_s: 10,
		default_speed_m_s: 0.05,
		default_rotation_speed_rad_s: Math.PI / 3,
		control_point_offset_m: 0,
		table_z_m: 0,
		supports_contact_stop: true,
		contact_stop_trusted: true,
		supports_home: false,
		gripper_gap_measured: true,
		gripper_open_span_m: 0.078,
		gripper_pad_length_m: 0.02,
		comfortable_reach_m: 0.7,
		reach_m: 0.8,
		tilt_limits_deg: { roll_cw: 90, roll_ccw: 90, pitch_down: 90, pitch_up: 50 },
		tilted_floor_m: 0.1,
		tilt_reach_deg: 80,
		max_object_height_m: 0.16,
		subgoal_median_s: 21.0,
		subgoal_verdict_s: 4.6,
		arrival_tolerance_m: 0.003,
		notes: [],
		...o,
	};
}

/** An image as the env server sends one: base64 JPEG bytes (a marker and an end). */
const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xd9]).toString("base64");

export type MmServer = {
	/** Env steps a tick takes (Metaworld 4, the 20 Hz OSC robots 1). */
	stepsPerTick?: number;
	/** The hand cannot rotate: rotate segments are refused (Metaworld). */
	fixedHand?: boolean;
	/** Success latches at this env step. */
	successAt?: number;
	/** How the server was started (its observation's `config`). */
	config: Json;
	caps?: Partial<Caps>;
	views?: string[];
	/** The robot's own fields of every observation (its `_mm_extra`), from the server's state. */
	extra?: (s: { steps: number; success: boolean; successStep: number | null }) => Json;
};

/**
 * A fake env server: `base` answers the robot's own calls (reset, meta, language, code.api); the
 * mm RPCs move a kinematic tool by whole ticks of `max_step`, every tick `stepsPerTick` env steps
 * and one video frame.
 */
export async function fakeMmServer(o: MmServer, base: (c: Call) => unknown) {
	const per = o.stepsPerTick ?? 1;
	const s = {
		steps: 0,
		episode: 1,
		obsId: 0,
		pos: [0.45, 0, 0.3],
		rot: IDENTITY as number[][],
		success: false,
		successStep: null as number | null,
		segments: [] as Json[],
		homes: 0,
	};
	const fresh = () =>
		Object.assign(s, { steps: 0, success: false, successStep: null, pos: [0.45, 0, 0.3], rot: IDENTITY });
	const caps = () => mmCaps({ ...o.caps, supports_home: o.config.mm_home === "reset" });
	const view = () => ({ marked768: JPEG, marked512: JPEG, raw512: JPEG });
	const observation = (images: boolean) => ({
		obs_id: ++s.obsId,
		frame_clock_s: s.steps * 0.05,
		episode_id: s.episode,
		ticks: s.steps / per,
		env_steps: s.steps,
		base_pose: { body: "base", position_m: [0, 0, 0], rotation: IDENTITY, quat_xyzw: [0, 0, 0, 1] },
		caps: caps(),
		pose: { position_m: [...s.pos], rotation: s.rot, quat_xyzw: [0, 0, 0, 1] },
		gripper: { opening_mm: 70, commanded: "open", channel: -1, moving: false },
		grasp: { holding: null, reason: "the last gripper command was not close" },
		table_z_m: 0,
		terminated: s.success,
		truncated: false,
		episode_over: s.success,
		...o.extra?.(s),
		degraded: { table_z: false, side_mount: false, side_mount_reason: null },
		camera_names: o.views ?? ["scene", "side", "wrist"],
		home_mode: o.config.mm_home,
		config: o.config,
		...(images
			? {
					cameras: Object.fromEntries((o.views ?? ["scene", "side", "wrist"]).map((v) => [v, view()])),
					mosaic_jpeg: JPEG,
				}
			: {}),
	});
	const frame = () => nd("uint8", [2, 2, 3], Buffer.alloc(12));
	/** `ticks` ticks of motion: their env steps, the success latch, one frame each. */
	const run = (ticks: number) => {
		const frames = [];
		for (let k = 0; k < ticks; k++) {
			s.steps += per;
			if (o.successAt && s.steps >= o.successAt && !s.success) {
				s.success = true;
				s.successStep = s.steps;
			}
			frames.push(frame());
		}
		return frames;
	};
	const answer = (c: Call): unknown => {
		const { method, kwargs } = c;
		if (method === "env.mm_observe") return observation(kwargs.images !== false);
		if (method !== "env.mm_run_step") {
			if (method === "env.reset") fresh();
			return base(c);
		}
		const seg = kwargs.seg;
		s.segments.push(seg);
		const reply = (out: Json, frames: Json[]) => {
			const obs = observation(kwargs.observe === "full");
			return {
				abort_reason: "",
				moved_m: 0,
				along_axis_m: 0,
				lateral_m: 0,
				turned_rad: 0,
				...out,
				kind: seg.kind,
				ticks: frames.length,
				elapsed_s: 0.01,
				frames,
				pose_after: obs.pose,
				episode_id: s.episode,
				env_steps: obs.env_steps,
				terminated: obs.terminated,
				episode_over: obs.episode_over,
				observation: obs,
			};
		};
		if (seg.kind === "home") {
			if (o.config.mm_home !== "reset") return reply({ outcome: "refused", message: "home is not available" }, []);
			s.homes++;
			s.episode++;
			fresh();
			return reply({ outcome: "done", message: "env.reset()", episode_reset: true }, []);
		}
		if (s.success) return reply({ outcome: "refused", message: "the episode is over" }, []);
		if (seg.kind === "rotate" && o.fixedHand)
			return reply({ outcome: "refused", message: "this robot's hand cannot rotate" }, []);
		if (seg.kind === "translate") {
			const d = seg.goal_position_m.map((v: number, i: number) => v - s.pos[i]);
			const moved = Math.hypot(...d);
			const frames = run(Math.max(1, Math.ceil(moved / seg.max_step - 1e-9)));
			s.pos = [...seg.goal_position_m];
			return reply({ outcome: "done", moved_m: moved, along_axis_m: moved }, frames);
		}
		if (seg.kind === "rotate") {
			const frames = run(Math.max(1, Math.ceil(seg.angle_rad / seg.max_step - 1e-9)));
			s.rot = seg.goal_rotation;
			return reply({ outcome: "done", turned_rad: seg.angle_rad }, frames);
		}
		return reply({ outcome: "done" }, run(seg.kind === "gripper" ? 32 : seg.ticks));
	};
	const server = await fakeEnv(answer);
	return { ...server, state: s };
}

/** A robot's extension in a stub pi on `env` (flags as on the command line); `model` is the session's model. */
export async function mmSession(
	t: { after: (fn: () => void) => void },
	robot: (pi: ExtensionAPI) => void,
	name: string,
	env: { url: string },
	values: Record<string, unknown>,
	model?: { provider: string; id: string },
) {
	const cwd = mkdtempSync(join(tmpdir(), "motormind-"));
	const flags: Record<string, unknown> = deployFlags({ "env-url": env.url, ...values });
	const handlers = new Map<string, Handler[]>();
	const registered: Record<string, unknown> = {};
	const tools = new Map<string, any>();
	const entries: { type: string; data: any }[] = [];
	let active: string[] = [];
	let handle: MotorMindHandle | undefined;
	const errors: string[] = [];
	const api: Record<string, unknown> = {
		on: (n: string, fn: Handler) => handlers.set(n, [...(handlers.get(n) ?? []), fn]),
		registerFlag: (n: string, f: { default?: unknown }) => {
			registered[n] = n in flags ? flags[n] : f.default;
		},
		getFlag: (n: string) => registered[n],
		registerTool: (tool: any) => tools.set(tool.name, tool),
		setActiveTools: (names: string[]) => {
			active = names;
		},
		getActiveTools: () => active,
		appendEntry: (type: string, data: any) => entries.push({ type, data }),
		events: {
			emit: (channel: string, data: any) => {
				if (channel === MOTORMIND_EVENT) handle = data;
			},
			on: () => () => {},
		},
	};
	const pi = new Proxy(api, { get: (target, k: string) => target[k] ?? (() => {}) }) as unknown as ExtensionAPI;
	const ctx = {
		hasUI: true,
		cwd,
		model,
		ui: new Proxy(
			{ notify: (m: string) => errors.push(m) },
			{ get: (target: any, k: string) => target[k] ?? (() => {}) },
		),
		shutdown: () => {},
		abort: () => {},
		sessionManager: {
			getBranch: () => [],
			getSessionDir: () => cwd,
			getSessionFile: () => undefined,
			getSessionId: () => "s",
		},
	};
	async function emit(n: string, event: Record<string, unknown> = {}) {
		let result: any;
		for (const fn of handlers.get(n) ?? []) result = (await fn({ type: n, ...event }, ctx)) ?? result;
		return result;
	}
	const call = async (tool: string, params: Record<string, unknown>) => {
		assert.ok(active.includes(tool), `${tool} is not active: ${active.join(", ")}`);
		try {
			const r = await tools.get(tool).execute(`id-${tool}`, params, undefined, undefined, ctx);
			return { ...r, isError: false };
		} catch (err) {
			return {
				content: [{ type: "text", text: String(err instanceof Error ? err.message : err) }],
				details: {},
				isError: true,
			};
		}
	};
	robot(pi);
	const loaded = [...tools.keys()];
	await emit("session_start", { reason: "startup" });
	t.after(() => void emit("session_shutdown", { reason: "quit" }));
	const finish = async () => {
		await emit("before_agent_start", { prompt: "Solve the task.", systemPrompt: "" });
		await emit("agent_start");
		await call("finish", { status: "failure", summary: "test" });
		await emit("agent_end", { messages: [] });
		return entries.find((e) => e.type === RESULT_ENTRY)?.data;
	};
	return {
		call,
		emit,
		errors,
		finish,
		flags: registered,
		loaded,
		active: () => active,
		handle: () => handle,
		tools: () => [...tools.keys()],
		/** The start failed closed: the UI was told why. */
		unavailable: () => errors.some((e) => e.startsWith(`${name} unavailable: `)),
	};
}

/** A tool result's JSON body. */
export const body = (r: { content: { text?: string }[] }) => JSON.parse(r.content[0].text as string);
