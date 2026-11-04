import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { RpcClient } from "../src/infra/rpc.ts";
import { loadManifest, toolSchema } from "../src/primitives/manifest.ts";
import { codePrimitives } from "../src/primitives/registry.ts";
import {
	adaptersFor,
	baseSuite,
	embodimentMismatch,
	mountAdapters,
	PICK_PARAMETERS,
	pickTracker,
	suiteMismatch,
	VLA_ADAPTERS,
	vlaIdentity,
	vlaInfo,
} from "../src/primitives/vla-adapters.ts";
import { toolSections } from "../src/robot.ts";
import libero from "../src/robots/libero/index.ts";
import { ROBOT_IDS, ROBOTS } from "../src/robots/maniskill/index.ts";
import { useDeployment } from "./helpers/deployment.ts";

type Tool = { name: string; description: string; parameters: unknown };

/**
 * Load the LIBERO robot into a stub pi that, like pi, answers getFlag with the flag's default while the
 * extension loads and applies the command-line `values` only afterwards; returns its registered tools.
 */
function loaded(values: Record<string, unknown> = {}) {
	const flags: Record<string, unknown> = {};
	const tools: Tool[] = [];
	const pi = {
		on: () => {},
		registerFlag: (name: string, o: { default?: unknown }) => {
			flags[name] = o.default;
		},
		getFlag: (name: string) => flags[name],
		registerTool: (t: Tool) => tools.push(t),
		registerCommand: () => {},
		registerProvider: () => {},
		registerMessageRenderer: () => {},
		registerShortcut: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	libero(pi);
	Object.assign(flags, values);
	return { flags, tools: new Map(tools.map((t) => [t.name, t])) };
}
const schema = (t: Tool | undefined) => JSON.parse(JSON.stringify(t?.parameters));

test("the adapter flags have no default", () => {
	const { flags } = loaded();
	for (const a of VLA_ADAPTERS) assert.equal(flags[a.flag], undefined, a.flag);
});

test("every adapter's grasp tool is registered at load with pi0_pick's parameters (flags are read at start)", () => {
	// pi applies --openvla etc. after the extension loaded, so registration cannot depend on them; the
	// tools become active at episode start only when their flag names a server.
	const bare = loaded().tools;
	const flagged = loaded({ openvla: "http://127.0.0.1:18600", gr00t: "127.0.0.1:18800" }).tools;
	assert.deepEqual([...bare.keys()], [...flagged.keys()]);
	for (const a of VLA_ADAPTERS) {
		assert.ok(bare.has(a.tool), a.tool);
		assert.deepEqual(schema(bare.get(a.tool)), schema(bare.get("pi0_pick")), a.tool);
	}
	assert.deepEqual(schema(bare.get("pi0_pick")), JSON.parse(JSON.stringify(PICK_PARAMETERS)));
	assert.match(bare.get("gr00t_act")!.description, /GR00T .* grasp/);
	assert.match(bare.get("openvla_act")!.description, /OpenVLA .* grasp/);
});

for (const file of ["SYSTEM.md", "compact/SYSTEM.md"])
	test(`the LIBERO prompt (${file}) describes an adapter only when its tool is active`, () => {
		const text = readFileSync(new URL(`../src/robots/libero/${file}`, import.meta.url), "utf8");
		const base = ["move_to", "pi0_pick", "release", "finish"];
		assert.doesNotMatch(toolSections(text, base), /openvla|gr00t/i);
		const withOft = toolSections(text, [...base, "openvla_oft_act"]);
		assert.match(
			withOft,
			/`openvla_oft_act` \(OpenVLA-OFT\) are grasp policies with the same contract as `pi0_pick`/,
		);
		assert.doesNotMatch(withOft, /openvla_act|gr00t/);
		const noPi0 = toolSections(text, ["move_to", "gr00t_act", "release", "finish"]);
		assert.match(noPi0, /`gr00t_act` \(GR00T\) are grasp policies with the same contract:/);
		assert.doesNotMatch(noPi0, /pi0/i);
	});

test("the pick heuristics: descend, lift with a partly closed gripper", () => {
	const t = pickTracker(0.3, 0.08, { prompt: "x" });
	assert.equal(t.update(0.25, 0.08), false, "not descended enough");
	assert.equal(t.update(0.18, 0.08), false, "descended 12 cm, no lift yet");
	assert.equal(t.update(0.24, 0.08), false, "lifted, but the gripper is open (0.08 >= 0.06)");
	assert.equal(t.update(0.24, 0.03), true, "lifted 6 cm with the gripper at 0.03");
	assert.deepEqual(t.summary(), { peak_lift_m: 0.24 - 0.18, descent_m: 0.3 - 0.18, min_gripper_opening: 0.03 });
	// A contact skill: lift_thresh 999 never succeeds by lift; an empty grasp (gripper at 0) fails gripper_open_thresh.
	assert.equal(pickTracker(0.3, 0.08, { prompt: "x", lift_thresh: 999 }).update(0.1, 0.03), false);
	const empty = pickTracker(0.3, 0.08, { prompt: "x", gripper_open_thresh: 0.01 });
	empty.update(0.15, 0.0);
	assert.equal(empty.update(0.25, 0.0), false);
});

test("a checkpoint fine-tuned on another suite is refused; unknown suites are trusted", () => {
	const oft = { service: "openvla-oft", model: "moojink/x", revision: "r", suite: "libero_spatial" };
	assert.equal(suiteMismatch(oft, "libero_spatial"), undefined);
	assert.match(
		suiteMismatch(oft, "libero_10")!,
		/moojink\/x@r is the libero_spatial fine-tune, but this episode is libero_10/,
	);
	assert.match(suiteMismatch(oft, "libero_10")!, /--suite libero_10\b/);
	assert.match(
		suiteMismatch(oft, "libero_10")!,
		/--suite libero_all --unnorm-key libero_10_no_noops/,
		"OFT's all-suite fine-tune, keyed to the episode's suite",
	);
	// LIBERO-Pro / plus suites are judged by their base suite.
	const vla = { service: "openvla", model: "openvla/x", revision: "r", suite: "libero_10" };
	for (const pro of ["libero_10_task", "libero_10_swap", "libero_10_lan", "libero_10_object", "libero_10_temp"])
		assert.equal(suiteMismatch(vla, pro), undefined, pro);
	assert.equal(suiteMismatch({ ...vla, suite: "libero_goal" }, "libero_goal_swap"), undefined);
	assert.equal(suiteMismatch({ ...vla, suite: "libero_object" }, "libero_object_object"), undefined);
	assert.equal(baseSuite("libero_spatial_with_mug"), "libero_spatial");
	assert.equal(baseSuite("libero_100"), "libero_100", "not libero_10");
	const wrong = suiteMismatch(vla, "libero_goal_task")!;
	assert.match(wrong, /this episode is libero_goal_task \(base suite libero_goal\)/);
	assert.match(wrong, /--suite libero_goal\b/, "the flag names a suite the server accepts");
	assert.doesNotMatch(wrong, /--suite libero_goal_task/);
	assert.doesNotMatch(wrong, /libero_all/, "OpenVLA has no all-suite fine-tune");
	const none = suiteMismatch(vla, "libero_90")!;
	assert.match(none, /no published fine-tune covers libero_90/);
	assert.doesNotMatch(none, /--suite/);
	// libero_all reports its --unnorm-key's suite: only that suite passes.
	const all = { ...oft, suite: "libero_goal", unnorm_key: "libero_goal_no_noops" };
	assert.equal(suiteMismatch(all, "libero_goal_swap"), undefined);
	assert.match(suiteMismatch(all, "libero_10")!, /--unnorm-key libero_10_no_noops/);
	assert.match(suiteMismatch(all, "libero_90")!, /no published fine-tune covers libero_90/);
	assert.match(
		suiteMismatch({ service: "gr00t", model: "RLinf/x", suite: "libero_spatial" }, "libero_goal")!,
		/gr00t RLinf\/x is the libero_spatial fine-tune/,
	);
	assert.equal(suiteMismatch({ ...oft, suite: null }, "libero_goal"), undefined, "a custom --model-path");
	assert.equal(suiteMismatch({ service: "pi05" }, "libero_goal"), undefined, "Pi0.5 has no vla.info");
	assert.equal(vlaIdentity(oft), "openvla-oft moojink/x@r");
	assert.equal(vlaIdentity({ service: "pi05" }), "pi05");
});

test("vlaInfo reads healthz and vla.info; a server without vla.info is its healthz name alone", async (t) => {
	const serve = (info: Record<string, unknown> | undefined) => {
		const server = createServer((req, res) => {
			let body = "";
			req.on("data", (c) => {
				body += c;
			});
			req.on("end", () => {
				const { method } = JSON.parse(body);
				if (method === "healthz")
					res.end(JSON.stringify({ ok: true, result: { status: "ok", service: "openvla" } }));
				else if (method === "vla.info" && info) res.end(JSON.stringify({ ok: true, result: info }));
				else res.end(JSON.stringify({ ok: false, error: `unknown method ${method}` }));
			});
		});
		return new Promise<string>((r) =>
			server.listen(0, "127.0.0.1", () => {
				t.after(() => {
					server.closeAllConnections();
					server.close();
				});
				r(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
			}),
		);
	};
	const adapter = await vlaInfo(
		new RpcClient(await serve({ model: "openvla/x", revision: "abc", suite: "libero_10", horizon: 1 })),
	);
	assert.deepEqual(adapter, { service: "openvla", model: "openvla/x", revision: "abc", suite: "libero_10" });
	const bare = await vlaInfo(new RpcClient(await serve(undefined)));
	assert.deepEqual(bare, { service: "openvla" });
});

test("every adapter runs on LIBERO; OpenVLA's Bridge statistics also on ManiSkill's bridge WidowX", () => {
	assert.deepEqual(
		adaptersFor("libero").map((a) => a.tool),
		["openvla_act", "openvla_oft_act", "gr00t_act"],
	);
	assert.deepEqual(
		adaptersFor("maniskill").map((a) => a.tool),
		["openvla_act"],
	);
	const ms = VLA_ADAPTERS[0].embodiments.maniskill!;
	assert.deepEqual(ms, { suites: ["bridge_orig"], action: "bridge_ee_delta_euler", cameras: { main: "agentview" } });
	// The arms whose env server translates that action space: the server's VLA_ACTION_SPACES, the same rows.
	const server = readFileSync(
		new URL("../../../services/pi_embodied_services/robots/maniskill/env_server.py", import.meta.url),
		"utf8",
	);
	const table = /^VLA_ACTION_SPACES = \{"([a-z_]+)": \(\(([^)]*)\)/m.exec(server)!;
	assert.equal(table[1], ms.action);
	assert.deepEqual(
		[...table[2].matchAll(/"([a-z0-9_]+)"/g)].map((m) => m[1]),
		ROBOT_IDS.filter((id) => (ROBOTS[id] as { vlaActions?: readonly string[] }).vlaActions?.includes(ms.action)),
	);
	// The Bridge checkpoint the embodiment names is the one the OpenVLA server publishes.
	const openvla = readFileSync(
		new URL("../../../services/pi_embodied_services/components/openvla_server.py", import.meta.url),
		"utf8",
	);
	assert.match(
		openvla,
		/"bridge_orig": Checkpoint\(\s*"openvla\/openvla-7b",\s*"47a0ec7fc4ec123775a391911046cf33cf9ed83f",\s*"bridge_orig",\s*False,\s*BRIDGE_EE_DELTA_EULER,/,
	);
});

test("mountAdapters: a client per named adapter, refused without an embodiment, a known name or an endpoint", () => {
	const pi = { getFlag: () => "" } as unknown as ExtensionAPI;
	useDeployment({ services: { openvla: "http://127.0.0.1:18600", gr00t: "http://127.0.0.1:18800" } });
	const m = mountAdapters(pi, ["openvla"], "maniskill");
	assert.deepEqual([...m.keys()], ["openvla_act"]);
	assert.equal(m.get("openvla_act")!.embodiment.action, "bridge_ee_delta_euler");
	assert.deepEqual([...mountAdapters(pi, ["openvla", "gr00t"], "libero").keys()], ["openvla_act", "gr00t_act"]);
	assert.throws(
		() => mountAdapters(pi, ["gr00t"], "maniskill"),
		/--vla-adapter gr00t has no maniskill embodiment \(its checkpoints run libero\); maniskill mounts openvla/,
	);
	assert.throws(
		() => mountAdapters(pi, ["octo"], "maniskill"),
		/--vla-adapter octo: not one of openvla, openvla-oft, gr00t/,
	);
	assert.throws(
		() => mountAdapters(pi, ["openvla-oft"], "libero"),
		/--vla-adapter openvla-oft needs services\.openvla_oft/,
	);
	assert.equal(mountAdapters(pi, [], "maniskill").size, 0);
});

test("embodimentMismatch: the action space must be the rig's; a known suite must be one it runs", () => {
	const m = {
		adapter: VLA_ADAPTERS[0],
		embodiment: VLA_ADAPTERS[0].embodiments.maniskill!,
		client: undefined as never,
	};
	const bridge = {
		service: "openvla",
		model: "openvla/openvla-7b",
		revision: "r",
		suite: "bridge_orig",
		action_space: "bridge_ee_delta_euler",
	};
	assert.equal(embodimentMismatch(bridge, m, "maniskill"), undefined);
	assert.equal(embodimentMismatch({ ...bridge, suite: null }, m, "maniskill"), undefined, "a custom checkpoint");
	assert.match(
		embodimentMismatch({ ...bridge, suite: "libero_spatial", action_space: "libero_osc" }, m, "maniskill")!,
		/^openvla openvla\/openvla-7b@r answers in libero_osc, but maniskill executes bridge_ee_delta_euler: restart it with --suite bridge_orig before using openvla_act\.$/,
	);
	// A server from before action_space answers LIBERO's.
	assert.match(
		embodimentMismatch({ service: "openvla", suite: "libero_10" }, m, "maniskill")!,
		/answers in libero_osc/,
	);
	assert.match(
		embodimentMismatch({ ...bridge, suite: "fractal20220817_data" }, m, "maniskill")!,
		/un-normalises with fractal20220817_data, but maniskill runs bridge_orig/,
	);
	// LIBERO keeps its per-episode check: a Bridge checkpoint is another suite's.
	assert.match(suiteMismatch(bridge, "libero_goal")!, /is the bridge_orig fine-tune, but this episode is libero_goal/);
});

test("ManiSkill's openvla_act is a tool only: declared in its manifest, never a program's primitive, its RPC methods internal", () => {
	const m = loadManifest("maniskill");
	const e = m.primitives.find((p) => p.name === "openvla_act")!;
	assert.equal(e.side, "ts");
	assert.equal(e.tier, "high");
	assert.deepEqual(JSON.parse(JSON.stringify(toolSchema(e).properties.chunks)), {
		description: "Action chunks to run closed-loop, stopping at success (default 1; each is one 0.2 s action)",
		minimum: 1,
		maximum: 60,
		type: "integer",
	});
	for (const tier of ["high", "low", "raw", "privileged"] as const)
		assert.ok(!codePrimitives(m, tier, () => true).some((p) => p.name === "openvla_act"), tier);
	assert.deepEqual(m.internal, ["env.vla_observation", "env.vla_step"]);
});
