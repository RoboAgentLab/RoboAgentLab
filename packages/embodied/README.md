# RoboAgentLab

Robotics extensions for the pi agent harness. Robots and algorithms connect through callable services; skills explain their workflows, and shared primitive declarations also expose capabilities to generated Python programs.

[Website](https://roboagentlab.github.io/) · [Get started](https://roboagentlab.github.io/docs/quickstart.html) · [Documentation](https://roboagentlab.github.io/docs/)

## Quickstart

Requires Node.js 22.19+ and a Linux NVIDIA GPU machine for simulation. Build pi from this fork:

```bash
git clone https://github.com/RoboAgentLab/RoboAgentLab ~/RoboAgentLab
cd ~/RoboAgentLab
npm ci --ignore-scripts
npm run build
npm link --ignore-scripts --workspace @earendil-works/pi-coding-agent
pi install ./packages/embodied
mkdir -p ~/robot-runs
cd ~/robot-runs
pi
```

Inside pi, configure an image-capable planner with `/login`, then run `/embodied-setup`.
The wizard selects a robot, its cell in the survey's observation × execution grid (O1–O3 × A1–A3),
optionally a paper's setting in that cell, the deployment, the planner and the experiment directory;
the agent then installs the services and runs the preflight checks, and nothing else. Afterwards,
run the printed launch command yourself and enter `Solve the task.`.

See the [full Quickstart](https://roboagentlab.github.io/docs/quickstart.html) for installation, one LIBERO episode, and the dashboard.

## Planner × surface

Two independent axes: the **planner** (who returns the next action) and the **action surface** it speaks
(the survey's A1–A3). Every planner drives every surface; the mechanism is the same tool call in every cell,
so a session, its `robot_result` row and the eval scripts never notice which planner ran.

| Planner \ surface | A1 units (`--units`, `act`) | A2 tools (the robot's tools) | A3 code (`--code`, `run_code`) |
| --- | --- | --- | --- |
| Frontier API model (`--model anthropic/...`) | implemented | implemented | implemented |
| Self-hosted model (`--model selfhost/...`) | implemented | implemented | implemented |
| Distilled student (`--model finetuned/<adapter>`) | implemented, released adapters (`--ft-surface units`, default) | implemented, no trained adapter yet (`--ft-surface tools`) | implemented, no trained adapter yet (`--ft-surface code`) |
| No-model replay (`--model flash/replay`) | implemented: `<name>_units.json`, a pure replay (nothing to re-anchor) | implemented: `<name>_plan.json` / `_recipe.jsonl`, re-anchored by the robot's Flash hook | implemented: `<name>_program.py`, run once through `run_code` |

- The student (`src/modes/finetuned`) sends one stateless request per step on every surface: the observation's
  images, then the surface's prompt. `tools` lists the active tools from the robot's manifest schemas and parses one
  JSON call, checked against the schema; anything else runs the read-only `--ft-observe` tool. `code` lists the tier's
  primitives exactly as code mode renders them and parses one fenced program run through `run_code` under code mode's
  tier and whitelist; anything else runs a program that only observes. Training pairs for both come from teacher
  sessions: `src/modes/finetuned/sessions_to_sft.ts` (or `python -m pi_embodied_services.finetuned.sessions_to_sft`),
  which renders the prompts with the same functions.
- The replay (`src/capabilities/flash`) plays whatever kind the plan file holds (`src/capabilities/flash/kinds.ts`
  reads units and program plans by content; a tool plan stays the robot's hook's). `--flash-plan <file>` names a plan
  directly; `src/capabilities/flash/generate.ts --kind units|program --session <s>.jsonl` writes the two new kinds
  from a solved session, `--kind tools` (the default) the anchored tool plan as before. Robots with Flash: the
  "Flash" entries of the table below.

## Guides

- [Environments and robots](https://roboagentlab.github.io/docs/environments.html)
- [Agent interfaces](https://roboagentlab.github.io/docs/modes.html)
- [Services and deployment](https://roboagentlab.github.io/docs/services.html)
- [Evaluation and replay](https://roboagentlab.github.io/docs/evaluation.html)
- [Data collection](https://roboagentlab.github.io/docs/data.html)
- [Develop an integration](https://roboagentlab.github.io/docs/development.html)

<details>
<summary>Robot modules (checked against the extension registrations)</summary>

| Robot | Extension | Success signal | Shared modules |
| --- | --- | --- | --- |
| LIBERO / LIBERO-PRO / LIBERO-plus | `src/robots/libero` | LIBERO `terminated` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, operator, Flash, MotorMind |
| RoboCasa | `src/robots/robocasa` | `env._check_success()` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, MotorMind; MotorMind drives the arm, the base held |
| RoboTwin | `src/robots/robotwin` | `eval_success` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, XPolicyLab, MotorMind; MotorMind drives one arm (`--robotwin-mm-arm`) |
| ManiSkill | `src/robots/maniskill` | ManiSkill `success` | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, MotorMind; `--vla-adapter openvla` (OpenVLA on the bridge WidowX) and its `--vla-alone` baseline; MotorMind drives `--arm panda` and `xarm6_robotiq` on the stock scenes |
| RoboLab | `src/robots/robolab` | RoboLab's task predicate | memory, explore, video, units, VDM, code, `--privileged`, Flash |
| RoboDojo (two ARX X5) | `src/robots/robodojo` | RoboDojo's `is_episode_end` (`score` = partial credit) | memory, explore, video, units, VDM, code, `--privileged`, flywheel, Flash, XPolicyLab |
| Robosuite | `src/robots/robosuite` | robosuite `_check_success` (Restack adds CaP-X's off-table rule), latched | memory, explore, video, units, VDM, code, `--privileged`, flywheel, MotorMind; MotorMind runs on the one-arm tasks |
| Metaworld | `src/robots/metaworld` | Metaworld `info["success"]`, latched | memory, explore, video, units, VDM, code, `--privileged`, flywheel, MotorMind; MotorMind refuses rotations (the Sawyer's hand orientation is fixed) |
| Genesis | `src/robots/genesis` | the task predicate (cube_pick: an 8 cm lift) | memory, explore, video, units, VDM, code, `--privileged`, flywheel |
| HumanCLAW (SMPL-X humanoid, HSSD homes) | `src/robots/humanclaw` | HumanClawBench's NavSR@20cm | video, units, VDM, `--privileged` |
| BEHAVIOR-1K / R1Pro | `src/robots/behavior` | the BDDL activity's `success`, latched (`q_score` = partial credit) | memory, explore, video, units, VDM, code, `--privileged` |
| Franka (real) | `src/robots/franka` | operator verdict (`--operator`) | memory, explore, video, units, VDM, code, operator |
| Dual Franka (real) | `src/robots/dual_franka` | operator verdict (required) | memory, explore, video, units, VDM, code, operator, XPolicyLab |
| Piper / dual Piper (real) | `src/robots/piper` | operator verdict (required) | memory, explore, video, units, VDM, code, operator |
| UR5e (real) | `src/robots/ur5e` | operator verdict (required) | memory, explore, video, units, VDM, code, operator |

</details>
