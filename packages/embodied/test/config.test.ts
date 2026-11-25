import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	auxModel,
	auxModels,
	configProblem,
	cudaDevice,
	deployment,
	deploymentAt,
	deploymentName,
	deploymentRecord,
	dir,
	python,
	SERVICE_DEFAULTS,
	service,
	servicesDir,
} from "../src/infra/config.ts";
import { useDeployment } from "./helpers/deployment.ts";

const pi = (flags: Record<string, unknown> = {}) => ({ getFlag: (n: string) => flags[n] }) as unknown as ExtensionAPI;

/** Run `fn` with a global config file holding `global` and a project dir (cwd) holding `project`. */
function withFiles(global: unknown, project: unknown, fn: () => void) {
	const root = mkdtempSync(join(tmpdir(), "pi-embodied-cfg-"));
	const g = join(root, "global.json");
	writeFileSync(g, JSON.stringify(global));
	mkdirSync(join(root, "proj", ".pi"), { recursive: true });
	if (project !== undefined) writeFileSync(join(root, "proj", ".pi", "embodied.json"), JSON.stringify(project));
	const cwd = process.cwd();
	const env = process.env.PI_EMBODIED_CONFIG;
	process.env.PI_EMBODIED_CONFIG = g;
	process.chdir(join(root, "proj"));
	try {
		fn();
	} finally {
		process.chdir(cwd);
		process.env.PI_EMBODIED_CONFIG = env;
	}
}

test("built-in < deployment; the project file wins key by key; --deployment picks one", () => {
	withFiles(
		{
			deployments: {
				box: { services: { sam3: "http://g:1", molmo: "http://g:2" }, python: { default: "/g/python" } },
				other: { services: { sam3: "http://o:1" } },
			},
		},
		{ deployments: { box: { services: { sam3: "http://p:1" }, python: { libero: "/p/libero" } } } },
		() => {
			const p = pi({ deployment: "box" });
			assert.equal(configProblem(p), undefined);
			assert.equal(service(p, "sam3"), "http://p:1");
			assert.equal(service(p, "molmo"), "http://g:2");
			assert.equal(service(p, "vla"), SERVICE_DEFAULTS.vla);
			assert.equal(service(p, "ik"), "");
			assert.equal(python(p, "libero"), "/p/libero");
			assert.equal(python(p, "robosuite"), "/g/python");
			assert.equal(service(pi({ deployment: "other" }), "sam3"), "http://o:1");
			// Two deployments and no `default`: none is picked, the built-ins apply.
			assert.equal(service(pi(), "sam3"), SERVICE_DEFAULTS.sam3);
			assert.deepEqual(deploymentRecord(pi()), { deployment: "", deployment_sha: null });
			const rec = deploymentRecord(p);
			assert.equal(rec.deployment, "box");
			assert.match(rec.deployment_sha ?? "", /^[0-9a-f]{16}$/);
		},
	);
});

test("an unknown deployment, service key or dir kind, or a malformed file, is a start-stopping problem", () => {
	withFiles({ deployments: { a: {}, b: {} } }, undefined, () => {
		assert.match(configProblem(pi({ deployment: "c" })) ?? "", /--deployment c is not in .*known: a, b/);
		// Bug 39: two deployments and no default is an ambiguity that stops the robot, not an empty deployment.
		assert.match(configProblem(pi()) ?? "", /2 deployments \(a, b\) and no `default`: pass --deployment <name>/);
		assert.equal(configProblem(pi({ deployment: "a" })), undefined);
	});
	withFiles({ deployments: { a: {}, b: {}, default: {} } }, undefined, () => {
		assert.equal(configProblem(pi()), undefined);
	});
	withFiles({ deployments: {} }, undefined, () => {
		assert.equal(configProblem(pi()), undefined, "no deployment at all is the built-in configuration");
	});
	withFiles({ deployments: { default: { services: { sam: "x" } } } }, undefined, () => {
		assert.match(configProblem(pi()) ?? "", /unknown services\.sam/);
	});
	withFiles({ deployments: { default: { dirs: { log: "x" } } } }, undefined, () => {
		assert.match(configProblem(pi()) ?? "", /unknown dirs\.log/);
	});
	const root = mkdtempSync(join(tmpdir(), "pi-embodied-cfg-"));
	writeFileSync(join(root, "bad.json"), "{ nope");
	const env = process.env.PI_EMBODIED_CONFIG;
	process.env.PI_EMBODIED_CONFIG = join(root, "bad.json");
	try {
		assert.match(configProblem(pi()) ?? "", /bad\.json/);
	} finally {
		process.env.PI_EMBODIED_CONFIG = env;
	}
});

test("the eval scripts' per-worker environment wins over the deployment; the only deployment is the default", () => {
	useDeployment({ cuda_device: "0", dirs: { logs: "/cfg/logs" }, services_dir: "/cfg/services" });
	const saved = { ...process.env };
	try {
		assert.equal(cudaDevice(pi()), "0");
		assert.equal(dir(pi(), "logs"), "/cfg/logs");
		assert.equal(dir(pi(), "video", "/fallback"), "/fallback");
		assert.equal(servicesDir(pi()), "/cfg/services");
		process.env.PI_EMBODIED_CUDA_DEVICE = "1";
		process.env.PI_EMBODIED_DIRS_LOGS = "/episode";
		assert.equal(cudaDevice(pi()), "1");
		assert.equal(dir(pi(), "logs"), "/episode");
	} finally {
		process.env.PI_EMBODIED_CUDA_DEVICE = saved.PI_EMBODIED_CUDA_DEVICE;
		process.env.PI_EMBODIED_DIRS_LOGS = saved.PI_EMBODIED_DIRS_LOGS;
		if (saved.PI_EMBODIED_CUDA_DEVICE === undefined) delete process.env.PI_EMBODIED_CUDA_DEVICE;
		if (saved.PI_EMBODIED_DIRS_LOGS === undefined) delete process.env.PI_EMBODIED_DIRS_LOGS;
	}
});

test("--aux-model is every auxiliary role's model; aux.<role> overrides one; an unknown role is refused", () => {
	useDeployment({ aux: { attach: "human/operator" } });
	const p = pi({ "aux-model": "selfhost/muse" });
	assert.equal(auxModel(p, "vdm"), "selfhost/muse");
	assert.equal(auxModel(p, "verify"), "selfhost/muse");
	assert.equal(auxModel(p, "attach"), "human/operator");
	assert.deepEqual(auxModels(pi()), { vdm: null, verify: null, attach: "human/operator" });
	useDeployment({ aux: { verifier: "x" } as never });
	assert.match(configProblem(pi()) ?? "", /unknown aux\.verifier/);
	useDeployment({});
});

test("a venv's own variable beats python.default; python.<venv> beats both", () => {
	const saved = process.env.PI_EMBODIED_FLYWHEEL_PYTHON;
	process.env.PI_EMBODIED_FLYWHEEL_PYTHON = "/env/flywheel";
	try {
		useDeployment({ python: { default: "/cfg/default" } });
		assert.equal(python(pi(), "flywheel", ["PI_EMBODIED_FLYWHEEL_PYTHON"]), "/env/flywheel");
		assert.equal(python(pi(), "libero"), "/cfg/default");
		useDeployment({ python: { default: "/cfg/default", flywheel: "/cfg/flywheel" } });
		assert.equal(python(pi(), "flywheel", ["PI_EMBODIED_FLYWHEEL_PYTHON"]), "/cfg/flywheel");
	} finally {
		if (saved === undefined) delete process.env.PI_EMBODIED_FLYWHEEL_PYTHON;
		else process.env.PI_EMBODIED_FLYWHEEL_PYTHON = saved;
		useDeployment({});
	}
});

test("the eval scripts refuse a renamed or moved flag before any cell runs, naming its replacement", () => {
	const sh = fileURLToPath(new URL("../src/scripts/old-flags.sh", import.meta.url));
	const run = (...args: string[]) =>
		spawnSync("bash", ["-c", `. "${sh}"; old_flags "$@"`, "_", ...args], { encoding: "utf8" });
	for (const [old, now] of [
		["--env", /--env-url/],
		["--robot-env=http://x", /--env-url/],
		["--env-id", /--task/],
		["--task-name", /--task/],
		["--robot", /--arm/],
		["--arm-id", /--arm/],
		["--eval-seed", /--layout-set/],
		["--vdm-model", /--aux-model/],
		["--sam3", /services\.<name>/],
		["--openvla", /--vla-adapter openvla/],
		["--contact-graspnet", /--grasp/],
		["--cuda-device", /cuda_device/],
		["--memory-dir", /dirs\.<kind>/],
	] as const) {
		const r = run("runs/x", "0-1", old, "v");
		assert.equal(r.status, 2, old);
		assert.match(r.stderr, now, old);
	}
	assert.equal(run("runs/x", "--env-url", "u", "--task", "t", "--arm", "a", "--aux-model", "m").status, 0);
});

test("deploymentAt is the deployment a pi started in a directory reads: that directory's project file over the global one, the deployment a name asks for, else default, else the only one; one that is no object is empty", () => {
	const root = mkdtempSync(join(tmpdir(), "pi-embodied-cfg-"));
	const g = join(root, "global.json");
	writeFileSync(g, JSON.stringify({ deployments: { default: { ffmpeg: "g" }, lab: { ffmpeg: "lab" } } }));
	const at = (project: unknown) => {
		const d = mkdtempSync(join(root, "run-"));
		if (project !== undefined) {
			mkdirSync(join(d, ".pi"));
			writeFileSync(join(d, ".pi/embodied.json"), JSON.stringify(project));
		}
		return d;
	};
	const env = process.env.PI_EMBODIED_CONFIG;
	process.env.PI_EMBODIED_CONFIG = g;
	try {
		assert.deepEqual(deploymentAt(at(undefined)), { ffmpeg: "g" });
		assert.deepEqual(deploymentAt(at(undefined), "lab"), { ffmpeg: "lab" });
		assert.deepEqual(deploymentAt(at({ deployments: { default: { ffmpeg: "p" } } })), { ffmpeg: "p" });
		assert.deepEqual(deploymentAt(at(undefined), "none"), {});
		// The project file of the directory given, not of the one this process runs in, over the global one.
		assert.deepEqual(deploymentAt(at({ deployments: { lab: { ffmpeg: "only" } } }), "lab"), { ffmpeg: "only" });
		// A deployment that is no object is empty.
		assert.deepEqual(deploymentAt(at({ deployments: { lab: 5 } }), "lab"), {});
		assert.deepEqual(deploymentAt(at({ deployments: { default: "x" } })), {});
		// With no global file: the only deployment, named or not, else none.
		process.env.PI_EMBODIED_CONFIG = join(root, "missing.json");
		assert.deepEqual(deploymentAt(at({ deployments: { lab: { ffmpeg: "only" } } })), { ffmpeg: "only" });
		assert.deepEqual(deploymentAt(at({ deployments: { default: null } })), {});
	} finally {
		if (env === undefined) delete process.env.PI_EMBODIED_CONFIG;
		else process.env.PI_EMBODIED_CONFIG = env;
	}
});

test("a malformed file is the problem reported without reading --deployment: an extension whose flags can no longer be read gets the problem, not a throw", () => {
	const root = mkdtempSync(join(tmpdir(), "pi-embodied-cfg-"));
	const g = join(root, "global.json");
	writeFileSync(g, "{ not json");
	const env = process.env.PI_EMBODIED_CONFIG;
	process.env.PI_EMBODIED_CONFIG = g;
	try {
		const stale = {
			getFlag: () => {
				throw new Error("the extension's flags can no longer be read");
			},
		} as unknown as ExtensionAPI;
		assert.match(configProblem(stale) ?? "", /global\.json: /);
		assert.deepEqual(deployment(stale), {});
		assert.equal(deploymentName(stale), "");
	} finally {
		if (env === undefined) delete process.env.PI_EMBODIED_CONFIG;
		else process.env.PI_EMBODIED_CONFIG = env;
	}
});

test("deploymentAt reads a relative path of the global file from the directory given, as a pi started there does", () => {
	const root = mkdtempSync(join(tmpdir(), "pi-embodied-cfg-"));
	const there = join(root, "run");
	mkdirSync(join(there, "conf"), { recursive: true });
	writeFileSync(join(there, "conf/embodied.json"), JSON.stringify({ deployments: { default: { ffmpeg: "there" } } }));
	const env = process.env.PI_EMBODIED_CONFIG;
	process.env.PI_EMBODIED_CONFIG = join("conf", "embodied.json");
	try {
		assert.deepEqual(deploymentAt(there), { ffmpeg: "there" });
	} finally {
		if (env === undefined) delete process.env.PI_EMBODIED_CONFIG;
		else process.env.PI_EMBODIED_CONFIG = env;
	}
});
