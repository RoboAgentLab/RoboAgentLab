#!/usr/bin/env node
/**
 * The survey's policy works per cell of the O × A grid, for /embodied-setup's work picker:
 *
 *   node packages/embodied/src/scripts/works-catalog.mjs <papers.json> [out.json]
 *
 * `papers.json` is the survey site's data (RoboAgentLab.github.io, research-data/papers.json);
 * every entry with a Policy category (`O1 / A1 Code`, `O2 / A2`, ...) is kept with its name, title,
 * year, URL and cells (`O1/A1C`, `O1/A1M`, `O2/A2`, `O3/A3`). The default output is
 * src/infra/setup/works.json, which the wizard reads at start.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const [src, dst = join(dirname(fileURLToPath(import.meta.url)), "../infra/setup/works.json")] = process.argv.slice(2);
if (!src) {
	console.error("usage: works-catalog.mjs <papers.json> [out.json]");
	process.exit(2);
}
const CELL = /^(O[123]) \/ (A1 Code|A1 Model|A2|A3)$/;
const column = { "A1 Code": "A1C", "A1 Model": "A1M", A2: "A2", A3: "A3" };
const works = [];
for (const it of JSON.parse(readFileSync(src, "utf8"))) {
	const cells = (it.categories ?? [])
		.map((c) => CELL.exec(c))
		.filter(Boolean)
		.map((m) => `${m[1]}/${column[m[2]]}`);
	if (!cells.length) continue;
	works.push({
		id: it.id,
		name: it.short_name || it.title,
		title: it.title,
		year: it.year,
		url: it.url,
		cells,
	});
}
works.sort((a, b) => a.name.localeCompare(b.name));
writeFileSync(dst, `${JSON.stringify(works, null, "\t")}\n`);
console.log(`${works.length} works, ${new Set(works.flatMap((w) => w.cells)).size} cells -> ${dst}`);
