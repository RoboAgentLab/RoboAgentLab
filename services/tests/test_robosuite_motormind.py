# Copyright 2026 The pi-embodied Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Robosuite's MotorMind mode (env.mm_observe / env.mm_run_step over utils/motormind.py) on a
kinematic OSC_POSE stand-in: robot0's base frame and LIBERO's capability table, one control step a
tick in the controller's base frame, the incremental gripper held at channel 0, success and its
step, home, Wipe's missing fingers and the two-arm tasks' refusal."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.robots.robosuite import env_server as rs
from pi_embodied_services.robots.robosuite import tasks

BASE_P = np.array([-0.56, 0.0, 0.912])
BASE_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # turned 90 deg
DOWN = np.diag([1.0, -1.0, -1.0])
CAMERAS = ["robot0_robotview", "sideview", "robot0_eye_in_hand", "agentview"]


class _Arm:
    """robosuite's env as the facade reads it: OSC_POSE deltas in the base frame (0.05 m / 0.5 rad
    per unit), the Panda gripper's incremental action (fingers move 8 mm a step toward the
    action's sign, an object ``obj`` wide stops them), success once the TCP is ``goal_z`` or
    higher."""

    action_dim = 7
    control_freq = 20

    def __init__(self, obj: float = 0.0005, goal_z: float = 9.0):
        self.obj, self.goal_z = obj, goal_z
        self.table_offset = np.array([0.0, 0.0, 0.8])
        self.table_full_size = np.array([0.8, 0.8, 0.05])
        self.robots = [
            SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"))
        ]
        model = SimpleNamespace(
            body_name2id=lambda name: 0,
            camera_name2id=CAMERAS.index,
            cam_fovy=np.array([45.0, 45.0, 75.0, 45.0]),
        )
        self.sim = SimpleNamespace(model=model, data=SimpleNamespace())
        self.sim.data.body_xpos = np.array([BASE_P])
        self.sim.data.body_xmat = np.array([BASE_R.reshape(-1)])
        self.sim.data.cam_xmat = np.array([np.eye(3).reshape(-1)] * 4)
        self.reset()

    def reset(self):
        self.pos = BASE_P + BASE_R @ np.array([0.45, 0.0, 0.3])
        self.rot = BASE_R @ DOWN
        self.width = 0.078
        self.actions: list[np.ndarray] = []

    @property
    def _cam_xpos(self):
        return np.array(
            [[0.5, 0, 1.4], [0, 1.3, 1.5], self.pos + [0, 0, 0.05], [0.5, 0, 1.4]]
        )

    def _get_observations(self):
        self.sim.data.cam_xpos = self._cam_xpos
        return {
            "robot0_eef_pos": self.pos.copy(),
            "robot0_eef_quat": rig.matrix_quat_xyzw(self.rot),
            "robot0_gripper_qpos": np.array([self.width / 2, -self.width / 2]),
            "robot0_joint_pos": np.zeros(7),
        }

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        self.actions.append(a)
        self.pos = self.pos + BASE_R @ (a[:3] * 0.05)
        self.rot = rig.axis_angle_matrix(BASE_R @ (a[3:6] * 0.5)) @ self.rot
        if len(a) > 6:
            self.width = float(np.clip(self.width - 0.008 * a[6], self.obj, 0.078))

    def _check_success(self):
        return bool(self.pos[2] >= self.goal_z)


def facade(task: str = "Lift", **kw) -> rs.RobosuiteEnvFacade:
    t = tasks.TASKS[task]
    f = object.__new__(rs.RobosuiteEnvFacade)
    f._mm_setup(**{k: kw.pop(k) for k in ("mm_home", "mm_truth") if k in kw})
    f._task_name, f._task = task, t
    f._env = _Arm(**kw)
    f._env.action_dim = 7 if t.gripper else 6
    f._seed, f._settle, f._max_steps, f._video_every = 0, 0, 10_000, 4
    f._size = 8
    f._grip = {arm: rs.OPEN for arm in t.arms}
    f._steps, f._success_step, f._home = 0, None, {}
    f._record, f._motion_frames = None, []
    f._cameras = {"agentview": t.camera, "wrist": rs.WRIST_CAMERA}
    f.ground_truth_poses = lambda names=None: {"poses": {}}

    def render(camera, size, depth=False):
        rgb = np.full((size, size, 3), 80, dtype=np.uint8)
        if not depth:
            return rgb
        cam = f._env._cam_xpos[CAMERAS.index(camera)]
        return rgb, np.full((size, size), cam[2] - 0.8, dtype=np.float32)

    f._render = render
    rs.BaseEnvFacade.__init__(f)
    f.reset()
    f._env.actions.clear()
    return f


def move(f, axis_base, mm: float, speed=0.05):
    p0, r0 = f._mm_pose()
    a = np.asarray(axis_base, dtype=np.float64)
    return f.mm_run_step(
        seg={
            "kind": "translate",
            "goal_position_m": (p0 + a * mm / 1000).tolist(),
            "goal_rotation": r0.tolist(),
            "axis": a.tolist(),
            "distance_m": mm / 1000,
            "speed": speed,
        }
    )


def test_the_base_frame_is_robot0s_and_the_caps_are_liberos_panda():
    f = facade()
    o = f.mm_observe()
    assert o["pose"]["position_m"] == pytest.approx([0.45, 0.0, 0.3])
    assert rig.jaws_down(np.asarray(o["pose"]["rotation"]))
    assert o["base_pose"]["body"] == "robot0_base"
    caps = o["caps"]
    assert (caps["max_translation_m"], caps["max_rotation_rad"]) == (0.05, 0.5)
    assert (
        caps["reach_m"] == 0.78
        and caps["gripper_open_span_m"] == rig.GRIPPER_OPEN_SPAN_M
    )
    assert caps["tilt_limits_deg"]["pitch_up"] == 53.6
    assert o["table_z_m"] == pytest.approx(0.8 - BASE_P[2], abs=0.003)
    assert set(o["cameras"]) == {"scene", "side", "wrist"}
    assert o["cameras"]["scene"]["camera"] == "robot0_robotview"
    assert o["cameras"]["side"]["camera"] == "sideview"
    assert o["settle_ticks"] == 0, "an OSC hand needs no settle"
    assert o["success_step"] is None


def test_a_tick_is_one_osc_step_in_the_base_frame():
    f = facade()
    w0 = f._env.pos.copy()
    out = move(f, [1, 0, 0], 50, speed=0.025)
    assert out["outcome"] == "done"
    assert np.allclose(f._env.pos - w0, BASE_R @ [0.05, 0, 0], atol=rig.ARRIVE_M)
    assert out["env_steps"] == out["ticks"] == len(f._env.actions)
    # 25 mm/s is 1.25 mm a tick: 0.025 of OSC's 0.05 m per unit, along base +X.
    assert f._env.actions[0][:3] == pytest.approx([0.025, 0.0, 0.0])
    # One video frame every fourth step.
    assert len(out["frames"]) == out["ticks"] // 4
    _p, r0 = f._mm_pose()
    goal = rig.axis_angle_matrix([0, 0, math.radians(30)]) @ r0
    turned = f.mm_run_step(
        seg={
            "kind": "rotate",
            "goal_rotation": goal.tolist(),
            "speed": math.radians(60),
        }
    )
    assert turned["turned_rad"] == pytest.approx(math.radians(30), abs=0.03)


def test_the_incremental_gripper_holds_at_channel_zero():
    f = facade(obj=0.03)
    held = f.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert held["grasp"]["holding"] is True and f._env.actions[-1][6] == rs.CLOSE
    part = f.mm_run_step(seg={"kind": "gripper", "state": "open", "open_to_mm": 50})
    assert part["gripper"]["commanded"] == "hold" and f._env.actions[-1][6] == 0.0
    gap = part["detail"]["opening_mm"]
    move(f, [0, 0, 1], 20)
    assert f._mm_opening() * 1000 == pytest.approx(gap)
    assert f.state()["robot0_gripper_command"] == "open"


def test_success_and_its_step_latch_and_home_resets_with_mm_home_reset():
    f = facade(goal_z=BASE_P[2] + 0.319, mm_home="reset")
    out = move(f, [0, 0, 1], 40)
    assert out["terminated"] and out["episode_over"]
    assert out["observation"]["success_step"] == 8
    assert move(f, [0, 0, 1], 10)["outcome"] == "refused"
    home = f.mm_run_step(seg={"kind": "home"})
    assert home["outcome"] == "done" and home["episode_reset"]
    assert (home["env_steps"], home["terminated"]) == (0, False)
    assert home["observation"]["success_step"] is None
    assert facade().mm_run_step(seg={"kind": "home"})["outcome"] == "refused"


def test_wipe_refuses_gripper_segments_and_the_two_arm_tasks_refuse_the_mode():
    w = facade("Wipe")
    assert (
        w.mm_run_step(seg={"kind": "gripper", "state": "close"})["outcome"] == "refused"
    )
    assert move(w, [1, 0, 0], 10)["outcome"] == "done"
    two = object.__new__(rs.RobosuiteEnvFacade)
    two._task_name, two._task = "TwoArmLift", tasks.TASKS["TwoArmLift"]
    with pytest.raises(ValueError, match="drives one arm; TwoArmLift has 2"):
        two._mm_static()


def test_the_mm_rpcs_are_registered_and_declared_internal():
    f = facade()
    assert {"env.mm_observe", "env.mm_run_step"} <= set(f._rpc)
    f._manifest_ready()
