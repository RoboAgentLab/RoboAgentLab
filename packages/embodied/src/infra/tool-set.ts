// The tools a robot's planner may call: the ones the robot's base and its modules activate.

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * The grant a robot keeps over its session's tools: ../robot.ts's tool_call gate refuses every other
 * tool, and its `finish` hides their declarations from every request.
 *
 * pi activates tools by itself as well: the tools of an MCP server in the operator's pi config, or
 * `codemode`, which calls them; a tool an extension registers after the robot's start; a `--tools`
 * entry, again whenever any tool registers. The robot's readiness and budget checks let such a tool
 * through, and the memory guard judges only its six file tools, by path. From `ownToolSet(pi)` on,
 * `pi.setActiveTools` also sets the grant, and `pi.getActiveTools` lists the active tools that are
 * granted, so a module that adds a tool to the active ones (`[...pi.getActiveTools(), "x"]`) grants
 * that tool and nothing pi added. The same `pi` serves the base and every module it mounts.
 *
 * The grant is by name. The gate sees the calls the planner makes and the calls a tool makes through
 * `ctx.executeTool`; a robot tool calls other tools that way, never through `ctx.tools[i].execute`.
 */
export interface ToolSet {
	/** Whether the robot's code activated `tool` (by name; pi ignores a name it has no tool for). */
	granted(tool: string): boolean;
	/** Deactivates the active tools that are not granted, and returns their names. */
	withdraw(): string[];
}

const owned = new WeakMap<ExtensionAPI, ToolSet>();

/** The grant over `pi`'s tools, installed on its first call and the same one afterwards. */
export function ownToolSet(pi: ExtensionAPI): ToolSet {
	const had = owned.get(pi);
	if (had) return had;
	const activate = pi.setActiveTools.bind(pi);
	const active = pi.getActiveTools.bind(pi);
	let grant = new Set<string>();
	pi.setActiveTools = (names: string[]) => {
		grant = new Set(names);
		activate(names);
	};
	pi.getActiveTools = () => active().filter((t) => grant.has(t));
	const set: ToolSet = {
		granted: (tool) => grant.has(tool),
		withdraw() {
			const now = active();
			const extra = now.filter((t) => !grant.has(t));
			if (extra.length) activate(now.filter((t) => grant.has(t)));
			return extra;
		},
	};
	owned.set(pi, set);
	return set;
}
