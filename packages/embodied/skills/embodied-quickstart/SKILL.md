---
name: embodied-quickstart
description: Install, preflight and run pi-embodied robots (simulators LIBERO, ManiSkill, Metaworld, Robosuite, Genesis, RoboCasa, RoboTwin, RoboLab, RoboDojo, BEHAVIOR-1K, HumanCLAW; real Franka, dual Franka, Piper, UR5e). Use when setting up a robot or benchmark, after /embodied-setup, when a robot run fails to start, or when the user asks which observation (O1–O3) and execution (A1–A3) setting a paper's configuration maps to or wants to reproduce a paper's setting (Show-Harness, RPent, OpenETA, CaP-X tiers, ...).
---

# pi-embodied quickstart

Paths below are relative to the services checkout (`services/`) and the package (`packages/embodied/`
in the repository; the package root in an npm install). An npm install has no `services/`: clone
`https://github.com/LV-Robotics-Lab/pi-embodied.git` and use its `services/`. On a remote box run
every command over `ssh <host>`; pi and the experiment directory live on the box too.

## 1. Pick

`/embodied-setup` asks for the robot (sim or real, then its name), its cell in the survey's
observation × execution grid (Figure A1a), optionally the work to reproduce in that cell, where
services run, the planner and the experiment directory. A request after the command
(`/embodied-setup reproduce RPent on LIBERO-PRO`; any language,
the names in ASCII) settles what it names, and only the rest is asked: a robot by id or label (a
name several share, `franka`, narrows the menu to them), a ported setting or a work of the survey
by short name or id (a work with several cells on the robot asks which; its ported settings narrow
the picker), levels (`O2`, `A1 Model`, `units`, `privileged`, `sam3`), `here` / `remote` / an ssh
host, a planner (`openai/gpt-6-astra` or `gpt-6-astra`). The confirmation starts with what the
request settled. The wizard and writes the experiment directory's `.pi/settings.json`: the
robot extension, the dashboard, and this package without its onboarding extension. A robot extension
replaces the coding tools, so start pi for the robot only in that directory. Settings cannot hold
flags; the task and the setting go on the command line. With no model configured it writes nothing and
tells the user to `/login` or add a provider to `~/.pi/agent/models.json`.

The grid (`src/infra/setup/axes.ts`) is the survey's Agent-as-Policy taxonomy, Figure A1a: observations
O1–O3 (§3.1–3.3) by execution authority (A1 Behavior Selection §3.4 with its Code §3.4.1 and Model
§3.4.2 sub-columns, A2 Tool Orchestration §3.5, A3 Code Synthesis §3.6). Each level is a flag; O1 and
A2 are a robot's defaults:

| level | the survey's definition | here | flags |
| --- | --- | --- | --- |
| O1 Sensory Inputs | decisions start from sensory inputs; agent-built perception remains O1 | camera frames only | none |
| O2 Perception Tools | decisions use supplied semantic or geometric perception tools or outputs | SAM3 masks with ids (detect, select_detection, reject_detection) | `--detections` (the arms: `--segment`); a SAM3 server at `services.sam3` |
| O3 Privileged State | decisions use oracle state unavailable at deployment | the simulator's ground-truth poses; the result is marked privileged | `--privileged` (simulators only) |
| A1 Code | a fixed programmed executor realizes behavior requests | Show-Harness action units run by the robot's motion code; no VLA | `--units=true` (`--units=both` keeps the tools) |
| A1 Model | a fixed learned executor realizes selected behaviors | the robot's VLA skill realizes each request; the run requires its server | `--require-skills pi0` (LIBERO), `rldx` (RoboCasa), `lingbot` (RoboTwin); `--vla` (Frankas); HumanCLAW's default |
| A2 Tool Orchestration | the agent selects execution APIs, supplies arguments, coordinates them | the robot's own tools: VLA, segmentation, motion | none |
| A3 Code Synthesis | the agent writes executable logic with conditions, loops, computations | `run_code`, Python over the primitives (CaP-X) | `--code=true` (`--code-api high\|low\|low-noexamples`) |

A robot's levels are read from its own spec (`src/infra/setup/modes.ts`): A1 Code when it has `units`,
A3 when it has `code`, O3 when it has a ground truth. A robot without a level has no cell in that row or
column: HumanCLAW is A1 Model only (its units are run by its motion model), real arms have no O3, only
LIBERO, RoboCasa, RoboTwin, HumanCLAW and the Frankas have a VLA for A1 Model. `--units` and `--code` are mutually exclusive.
A1 Model and A2 share the same tools: the cell says what the planner is asked to do (one behavior
request at a time to the VLA, or composing tools); the result row records what it did.

### The work to reproduce

After the cell, `/embodied-setup` lists the ported settings the survey files in it (typing filters by
the short name); a cell with none skips the question. A ported setting runs as a `--preset` / `--tier`
(`src/infra/tiers.ts` expands it to the flags of that setting and refuses a flag given with another
value; the cell adds none of its own). The survey's other works (`src/infra/setup/works.json`, 493 works
over the 12 cells, generated from the survey site's papers.json by `src/scripts/works-catalog.mjs
<papers.json>`; regenerate it when the survey's classification changes) are counted under the grid and
recognized in a request, but not offered until they are ported: a request naming one gets a notice with
its cells, then the grid, and the agent starts from the cell's flags. A name in a request that is no
robot, ported setting, survey work, level or planner is called out as possibly not supported yet, in a
notice and in the confirmation.

When the user names a work ("run it like Show-Harness", "the CaP-X S2 setting", "reproduce RPent",
"the configuration of paper X"), first say which cell the survey files it in and the cell's flags, then
whether it is ported and the preset:

| short name | work | cell | flags | robots |
| --- | --- | --- | --- | --- |
| showharness | Show-Harness | O1 × A1 Code | `--preset showharness` | any with units |
| motormind | MotorMind | O2 × A1 Code | `--preset motormind` (side camera, settle ticks, no memory) | any whose spec has `motormind` (LIBERO today) |
| humanclaw | HumanCLAW | O1 × A1 Model | `--preset humanclaw` | HumanCLAW |
| rpent | RPent | O1 × A2 | `--preset rpent` (memory + VLA skill) | LIBERO, RoboCasa, Franka, dual Franka |
| openeta | OpenETA | O2 × A2 | `--preset openeta` (VDM + anchor image) | any |
| capx-S1 | CaP-X S1 | O3 × A3 | `--tier S1` (ground truth) | simulators with code |
| capx-S2, S3, S4 | CaP-X S2–S4 | O2 × A3 | `--tier S2\|S3\|S4` (one program; high / low / low-noexamples API) | any with code |
| capx-M1 … M4 | CaP-X M1–M4 | O2 × A3 | `--tier M1\|M2\|M3\|M4` (multi-turn; text / image / VDM / VDM + low API) | any with code |
| xpolicylab | XPolicyLab | (Infrastructure in the survey) | `--preset xpolicylab --xpolicy ws://host:port` (a policy server; not in the wizard) | RoboTwin, RoboDojo, Piper, dual Franka |

A work not in works.json is answered the same way: read its observation access (what the agent is given
beyond the frames) and execution interface (behaviors to a programmed or learned executor, tools, or
code) off the taxonomy, name the cell, then the flags.

The planner is a model of the registry or, in A1 Code without a preset, the robot's fine-tuned adapter
(`src/modes/finetuned` extension, `--model finetuned/<adapter>`): a small VLM + LoRA picks each unit,
served by vLLM. Flash (`--model flash/replay`: a recorded plan replayed with live grounding, no planner;
LIBERO, RoboCasa, RoboTwin, ManiSkill, RoboLab, RoboDojo) is not an experiment setting and is started
by hand.

The fine-tuned planner needs two more steps after the install: `FT_ADAPTER=<adapter> services/setup.sh
finetuned` (default `qwen3_5_2b_sim`) fetches the released adapter and its base model with
`finetuned/download.py` (pinned, verified), and `services/pi_embodied_services/finetuned/serve.sh`
(`MODEL`, `LORA`, `VLLM_VENV` as setup.sh prints them; serve.sh's own defaults are
`$PI_EMBODIED_WEIGHTS`, else `<checkout>/.local/checkpoints`, and `<checkout>/.local/venvs/vllm`)
serves them on the vLLM endpoint the deployment's `services.finetuned` names (default
http://127.0.0.1:8010/v1). Start it before the episodes and leave it running. LIBERO's adapters are the user's own (`--model finetuned/local
--ft-model <served adapter>`).

The user confirms the plan once in `/embodied-setup`; pi does not gate each tool call, so ask before
sudo, anything touching real hardware, or a download the plan does not list. The install and the
experiment are separate: install, run the preflight, and end with the launch command; the user starts
episodes with it. Do not start one.

## 2. Install

Prerequisites; check them before the install and tell the user what is missing:

- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`) or the Python the target wants (3.11 for
  most); without either, `setup.sh` stops.
- `node` 22.19 or newer on the user's `PATH` for the preflight (pi's own Node may not be on it: the
  pi.dev installer's is in `~/.local/share/pi-node/current/bin`; otherwise nodejs.org or nvm).
- GitHub reachable: the extras install packages from git (LIBERO's pulls RLinf's openpi fork with
  submodules, LIBERO and SAM3). A stalled `uv pip install` is usually a git clone; it needs a proxy
  (`https_proxy`).
- Hugging Face reachable for assets and `--weights`, or `HF_ENDPOINT` set to a mirror.
- Disk: a robot venv is several GB and uv's cache can reach 16 GB (`UV_CACHE_DIR` moves it).

```bash
services/setup.sh <target> [--weights] [--venv DIR] [--weights-dir DIR] [--no-assets] [--dry-run]
services/setup.sh --help        # targets and what each fetches
```

Run `--dry-run` first and show the user what it will do. Robot targets: libero, libero-pro,
libero-plus, maniskill, metaworld, robosuite, genesis, robocasa, robotwin, robolab, robodojo,
behavior, humanclaw, franka, franka-polymetis, dual-franka, piper, ur5e; others: finetuned,
llamafactory, flywheel (LeRobot export), graspnet1b. Most create `services/.venv-<target>` with the
matching extra; robolab, robodojo, behavior (Isaac Sim 6.1) and humanclaw (Habitat-Sim) run their
install scripts on a checkout under `$HOME` (or `<NAME>_ROOT`). `--weights` go to
`$PI_EMBODIED_WEIGHTS` (default `~/.cache/pi-embodied`), as do most targets' assets; `--no-assets`
skips the assets. libero fetches
the LIBERO scene assets (`RLinf/LIBERO-PRO-assets` pinned at `e4b10c9a`, about 430 MB) into the
Hugging Face cache, checks every file the Hub lists and links the package's `libero/libero/assets`
to them; libero-pro links the same directory to LIBERO-PRO's own verified download. It writes
`<venv>/pi-embodied.env`; `source` it before pi, `serve.sh` and `eval.sh`; it holds absolute paths. Mirrors (for example from
mainland China): `HF_ENDPOINT=https://hf-mirror.com` (it can answer HTTP 429 for datasets with many
files; the LIBERO assets step then fetches two files at a time and retries with backoff, and if it
still fails, `unset HF_ENDPOINT` and rerun with huggingface.co through a proxy), `UV_INDEX_URL` / `PIP_INDEX_URL`, a PyTorch
wheel mirror with `--find-links`, and a proxy for GitHub; the env file does not carry them, so they
must be set in the shell that runs pi too. Gated data (SAM3, HSSD) needs `HF_TOKEN` with the license
accepted.

Where things run is the deployment config, not flags: `deployments.<name>` in `~/.pi/agent/embodied.json`
or `<cwd>/.pi/embodied.json` (`services.*` URLs, `python.<robot>`, `cuda_device`, `dirs.*`),
`--deployment NAME`, `/embodied-config` prints it; its `python.<robot>` beats the env file's
`PI_EMBODIED_PYTHON`. Removed flags (`--env`, `--sam3 URL`, `--python`, ...): docs/flags-migration.md.

Tools mode also needs the model servers: `packages/embodied/src/robots/<robot>/serve.sh` (LIBERO:
Pi0.5 + SAM3, and OpenVLA / OpenVLA-OFT / GR00T when their `*_PYTHON` is set; RoboCasa: RLDX-1, also
in units mode; RoboTwin: LingBot; dual Franka: Pi0.5 + SAM3), or `--serve-models vla,sam3,molmo` per
episode; other robots attach to the deployment's `services.*` (SAM3 for `segment`, Molmo for `--point`).

## 3. Preflight

```bash
source services/.venv-<target>/pi-embodied.env
node packages/embodied/src/infra/check.ts <robot> [--deployment NAME] [--units] [--model provider/id]
```

`<robot>` is the extension's directory name (`dual_franka`; libero-pro / libero-plus are `libero`).
Fix every FAIL before running (LIBERO: a missing scene asset is a FAIL; rerun `setup.sh`). Inside a robot session the same check is `/robot-check`.

## 4. One episode

`/embodied-run [request]` (in the directory above the experiment directories, where /embodied-setup
ran) starts one: it lists the experiment directories found (this one and its subdirectories whose
`.pi/settings.json` loads a robot), takes the setting from the `.pi/embodied-setup.json` record
/embodied-setup wrote (else asks the grid and the picker), asks the task flags, a GPU (nvidia-smi,
the freest first), the budget and the panels, then how: in this terminal (pi takes it over and starts the episode on
`Solve the task.` at once, after the trust prompt the first time), headless (the agent runs `pi -a -p ... "Solve
the task."` in the background and reports `robot_result`), or printed. Panels: the dashboard always,
on a fixed port (`--dashboard-port 8765`, the next free one when taken), and first the work's own panel
when the robot serves it (CaP-X's Viser 3D view, `--viser --viser-port 8080`, on LIBERO and the Frankas;
other settings fall back to the robot's own view or the dashboard alone). The dashboard binds to
loopback on the box, so from another machine the ports are tunnelled: the confirmation prints
`ssh -N -L 8765:127.0.0.1:8765 -L 8080:127.0.0.1:8080 <box>` when pi runs over ssh; a `LocalForward`
of the same ports in the user's ssh config entry for the box makes every ssh session carry them, and
then http://127.0.0.1:8765/ on the user's machine is the dashboard. `--dashboard-language zh-cn`
switches the dashboard's UI language. A request names the robot
(`/embodied-run libero`) or overrides the setting (`metaworld privileged code`). An experiment
whose venv is missing is refused: run /embodied-setup. The command it builds is the one below.


The launch command `/embodied-setup` hands over (`[setting flags]`: the cell's flags from the table
in section 1, or a paper's `--preset` / `--tier`, plus the fine-tuned planner's `--model finetuned/...`
flags; `--model flash/replay` for a replay):

```bash
cd <experiment dir> && source <services>/.venv-<target>/pi-embodied.env && pi <task flags> [setting flags] --max-turns 40 --time-limit 900 --dashboard=true
```

Always give the user the command with a budget: every turn sends camera images, and without
`--max-turns` / `--time-limit` (default no limit) an episode runs until the model calls `finish`,
which against a paid API can cost a lot. With a planner that has thinking levels, `--thinking low`
may be added (pi clamps it to the model's levels).
After the start the robot waits for a prompt: the user types `Solve the task.` (or another
instruction). For a first run without the terminal UI:
`pi -a -p <task flags> [setting flags] --max-turns 40 --time-limit 900 --thinking low "Solve the task."`.
On a machine with several GPUs add `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` (MuJoCo robots).

The first pi in the experiment directory asks whether to trust the project: its `.pi/settings.json`
loads the robot extension, so the user answers yes (`/trust` saves it); declined, pi starts without
the robot. `pi -p` cannot ask and runs untrusted (no robot, the coding tools, the global default
model) unless trust was saved or `-a` / `--approve` is given. The dashboard URL is shown at startup.
Boolean flags take the next word: write `--flag=true`. Without the experiment settings:
`pi -e packages/embodied/src/robots/<robot> ...`. Task flags: `--task` (and `--seed`) everywhere;
LIBERO `--suite`, `--libero-type standard|pro|plus`; HumanCLAW `--units=both --episode`; real arms
`--robot-config`, `--env-url`, UR5e `--arm <serial>`.

Optional, per launch (each registers nothing when off): `--web-tools` keeps web search and page
fetch for the robot, from pi packages installed once in the experiment directory
(`pi install -l npm:pi-web-search@1.6.0 npm:@zeldrisho/pi-web-fetch@0.9.2`; web_search uses the planner provider's
native search, so it needs a provider that has one); `--object-memory` (`--object-memory-dir` to keep
records per scene), and on LIBERO / Franka `--waypoints`, `--align-wrist`, `--grasp-advisor`.
`--replay-thinking=true` sends every earlier turn's thinking back (default false; it is most of a
Qwen / vLLM planner's input). `--model human/operator` lets a person answer as the planner (pi's
dialogs, or the dashboard). The task skills `/skill:embodied-pick`, `embodied-place`,
`embodied-push-pull`, `embodied-stack` and `embodied-object-memory` go in front of the prompt
(`pi ... "/skill:embodied-pick Solve the task."`): a robot's system prompt replaces pi's, so the model
does not see the skill list itself.

## 5. Evaluate

```bash
packages/embodied/src/robots/<robot>/eval.sh <out-dir> <cells...> --model <provider/model> \
  --max-turns 40 --time-limit 900 --thinking low
```

Each robot's `eval.sh` header gives its cells (LIBERO: suite, tasks, seeds; most sims: tasks, seeds).
Run it from a directory without the experiment settings (it loads the robot with `-e`; loading it
twice fails on duplicate flags). Rerunning retries only invalid episodes; a run with other flags in
the same out dir is refused. `src/scripts/eval-parallel.sh [-j N --gpus LIST] [--durable] <robot>
<out-dir> <cells...>` runs the matrix on N workers (`--durable` resumes an interrupted one).

## Common failures

- `<robot> unavailable: ...` at start (`[<robot>] unavailable: ...` with `-p`): the robot did not start.
  Once its env server has bound its port, the message names the server's log (`... (env server log:
  <path>)` or `see <path>`; LIBERO: `/tmp/pi-embodied-env-<suite>-t<task>-s<seed>-<port>.log`), where
  the traceback is; a server that exits before that carries its printed output in the message
  instead. Read it, then run the preflight.
- No robot (no `/robot-task`, the coding tools are back) in the experiment directory: project trust
  was declined, or `pi -p` ran without `-a`; start pi there again and accept, or `/trust`.
- Import errors in the preflight: wrong venv (or the deployment's `python.<robot>` names another).
  Each robot family has its own venv; the extras conflict (Torch/Transformers pins).
- RTX 5090 / sm_120: install `torch==2.7.1 torchvision==0.22.1` from the cu128 index before the extra
  (Genesis: a cu128 torch>=2.8, graspnet1b: a cu128 torch, first; HumanCLAW: `HUMANCLAW_TORCH`); RoboTwin also needs
  cuRobo built against it (services/README.md).
- LIBERO renders black or crashes: `MUJOCO_GL=egl` needs an NVIDIA EGL driver; pick the GPU with
  `MUJOCO_EGL_DEVICE_ID`. MuJoCo must stay `3.3.0`.
- `LIBERO assets: incomplete download of RLinf/LIBERO-PRO-assets ...` or `could not list ...`: rerun
  `setup.sh libero` (it resumes; nothing is linked until every file is there); if a mirror keeps
  answering 429, `unset HF_ENDPOINT` and rerun through a proxy to huggingface.co. `pi-embodied.env`
  is written once the venv is installed, so it can be sourced even while the assets are missing. `... holds files but not <scene>`: an old
  `libero/libero/assets` directory is in the way; move it and rerun.
- LIBERO-PRO assets "ready" but tasks missing files: rerun `setup.sh libero-pro`; its patched
  downloader verifies every file. `FileNotFoundError` under an old venv path: `~/.liberopro/config.yaml`
  (and `~/.libero/`) keep absolute paths; rewrite them after moving the venv, or set `LIBERO_CONFIG_PATH`.
- SAPIEN (ManiSkill, RoboTwin): needs `libvulkan1`; on sm_120 its OIDN denoiser logs "unsupported
  device type": replace SAPIEN's `oidn_library` with OIDN >= 2.3.
- `robotwin-download-assets` fails behind hf-mirror (tree API): download the asset zips directly.
- RoboLab / RoboDojo / BEHAVIOR: Isaac Sim takes about a minute to start (minutes cold); the
  `*_ROOT` checkout must be the patched one.
- A model server port already answers: `serve.sh` leaves it running; `serve.sh stop` stops only its own.
- Molmo and SAM3 on one GPU: start Molmo with `--offload-blocks 20`.
- Real robots: an operator at the e-stop (`--operator=true`; motions need approval, `--approval`);
  Franka needs Ray on the controller node. From Codex / Claude Code (README, "Using the robots from
  Codex or Claude Code") a real arm moves only with `PI_EMBODIED_CONFIRM_FILE` tickets or
  `PI_EMBODIED_MOTION_CONFIRMED=1`.
