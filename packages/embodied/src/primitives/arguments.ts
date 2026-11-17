/**
 * The facade's kwargs for a manifest tool call. A manifest entry's parameters exist in tool mode,
 * code mode or both (`modes`, ./manifest.ts); the env server's method takes the code-mode ones (its
 * startup self-check pins them to the method's signature: services' components/manifest.py
 * `_signature_mismatch`). A tool-only parameter is pi's, in one of two ways:
 *
 *   - it maps onto code-mode parameters: `point: [row, col]` is the tool's form of the method's
 *     `row` and `col` (align_wrist, ./wrist.ts), which is the one mapping the manifests declare;
 *   - it is pi-side state a caller speaking to the server directly has no access to: LIBERO's `step`
 *     (the recorded state a pixel came from) and `resolution` (which of pi's two image sizes).
 *
 * pi's tool wrappers and the MCP server (../integrations/mcp) both go through here, so the
 * adaptation is written once: `rpcParams` is the parameter set a direct caller may give (the MCP
 * tool's schema), `rpcArguments` the kwargs the method receives.
 *
 * Both also ask here which arguments a tool call may carry at all (`strayArguments`): pi's validation
 * keeps the keys a schema does not declare, and most tools hand their arguments to the server's
 * method, which can take more keywords than the tool offers (a program's options, keywords no mode
 * declares).
 */

import { listVariable, type ManifestEntry, type ManifestParam } from "./manifest.ts";

/** Whether a parameter exists in `mode` (default both). */
export const inMode = (p: ManifestParam, mode: "tool" | "code") => !p.modes || p.modes.includes(mode);
const toolOnly = (p: ManifestParam) => p.modes?.includes("tool") === true && !p.modes.includes("code");

/**
 * The arguments a call of `e`'s tool may carry on pi's path: the names of its tool-mode parameters.
 * One that a session's schema leaves out is among them (an enum over an empty list: `arm` on one
 * arm, `camera` with no cameras): it reaches the tool's handler, as it always did.
 */
export const toolArguments = (e: ManifestEntry): string[] =>
	Object.entries(e.params ?? {}).flatMap(([k, p]) => (inMode(p, "tool") ? [k] : []));

/**
 * How many stray names a refusal spells out, and how much of each. They are the caller's own text: shown as JSON
 * strings in printable ASCII, every other character escaped (JSON's short escapes, else \uXXXX), so the refusal stays
 * one line and reads as it is written.
 */
const STRAY_NAMES = 5;
const STRAY_NAME_CHARS = 40;
const quoted = (k: string) =>
	JSON.stringify(k).replace(/[^\x20-\x7e]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);

/**
 * What a tool call's arguments carry beyond `taken`, as the tail of a refusal (`no argument "step_m"
 * or "record"; it takes xyz, tol_m`), else undefined. `offered` gives the names to list, the schema's:
 * it is asked only when an argument is stray.
 */
export function strayArguments(
	args: unknown,
	taken: readonly string[],
	offered: () => readonly string[],
): string | undefined {
	if (typeof args !== "object" || args === null) return undefined;
	const stray = Object.entries(args).flatMap(([k, v]) => (v === undefined || taken.includes(k) ? [] : [k]));
	if (!stray.length) return undefined;
	const names = stray
		.slice(0, STRAY_NAMES)
		.map((k) => quoted(k.length > STRAY_NAME_CHARS ? `${k.slice(0, STRAY_NAME_CHARS)}...` : k));
	const takes = offered();
	return `no argument ${names.join(" or ")}${stray.length > STRAY_NAMES ? " or others" : ""}; it takes ${takes.length ? takes.join(", ") : "none"}`;
}

/** The robot base's refusal of such a call (../robot.ts), and the pattern that reads it back with the tool it names. */
export const strayRefusal = (tool: string, stray: string) => `${tool} has ${stray}.`;
export const STRAY_REFUSAL = /^(\S+) has no argument .+; it takes .+\.$/;

/** `point: [row, col]` as the method's `row` and `col` (the pixel's integer coordinates). */
export function pointArgs(point: readonly unknown[]): { row: number; col: number } {
	if (point.length !== 2 || !point.every((v) => Number.isFinite(Number(v))))
		throw new Error(`point must be [row, col], got ${JSON.stringify(point)}`);
	return { row: Math.round(Number(point[0])), col: Math.round(Number(point[1])) };
}

/** Whether `name`, a tool-only parameter of `e`, maps onto the method's parameters. */
function mapped(e: ManifestEntry, name: string): boolean {
	const params = e.params ?? {};
	return name === "point" && ["row", "col"].every((k) => params[k] !== undefined && inMode(params[k], "code"));
}

/** The tool parameters a direct caller of the method may give: the tool-mode ones less the unmapped tool-only ones. */
export function rpcParams(e: ManifestEntry): Record<string, ManifestParam> {
	return Object.fromEntries(
		Object.entries(e.params ?? {}).filter(([k, p]) => inMode(p, "tool") && (!toolOnly(p) || mapped(e, k))),
	);
}

/**
 * The method's kwargs for validated tool `params` of `e`: mapped tool-only parameters become the
 * method's (`point` → `row`, `col`); an unmapped tool-only parameter that is given is an error (the
 * caller asked for pi-side state the server has no notion of), and so is an argument the caller's
 * schema does not offer (`offered`: its property names); undefined values are dropped. No handler
 * stands between this caller and the method, so a parameter its schema leaves out (`arm` on one arm)
 * is refused here too, and the refusal names the list variable that left it out.
 */
export function rpcArguments(
	e: ManifestEntry,
	params: Record<string, unknown>,
	offered: readonly string[],
): Record<string, unknown> {
	const declared = e.params ?? {};
	const piSide = Object.keys(declared).filter((k) => toolOnly(declared[k]));
	const stray = strayArguments(params, [...offered, ...piSide], () => offered);
	if (stray) {
		// A parameter of the entry that this session's schema leaves out: its list variable is empty.
		const left = Object.entries(rpcParams(e)).flatMap(([k, p]) => {
			const list = params[k] !== undefined && !offered.includes(k) ? listVariable(p) : undefined;
			return list ? [`. ${k} is not offered while --var ${list} is empty`] : [];
		});
		throw new Error(`${e.name}: ${stray}${left.join("")}`);
	}
	const out: Record<string, unknown> = {};
	for (const [k, v] of Object.entries(params)) {
		if (v === undefined) continue;
		const p = declared[k];
		if (p && toolOnly(p)) {
			if (!mapped(e, k))
				throw new Error(`${e.name}: ${k} is pi's (its recorded states and image sizes), not the env server's`);
			Object.assign(out, pointArgs(v as readonly unknown[]));
			continue;
		}
		out[k] = v;
	}
	return out;
}
