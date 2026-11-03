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

"""RoboCasa's MotorMind mode (env.mm_observe / env.mm_run_step over utils/motormind.py) on a
kinematic PandaOmron stand-in: the arm controller's origin as the base frame (the mobile base
turned), one 12-D step a tick with the base and torso still, the incremental gripper held at
channel 0, the success latch and the episode's steps, home."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.robots.robocasa.env_server import RoboCasaEnvFacade
from pi_embodied_services.utils.rpc import RpcFacade

ORIGIN_P = np.array([1.0, 2.0, 0.9])
ORIGIN_R = rig.axis_angle_matrix(
    [0, 0, math.radians(120)]
)  # the base turned in the kitchen
DOWN = np.diag([1.0, -1.0, -1.0])
CAMERAS = ["robot0_agentview_center", "robot0_agentview_left", "robot0_eye_in_hand"]
COUNTER_Z = 0.92


def xyzw(r) -> np.ndarray:
    return rig.matrix_quat_xyzw(r)


class _Omron:
    """robosuite's kitchen env as the facade reads it: the arm's OSC_POSE deltas in its origin's
    frame (0.05 m / 0.5 rad per unit), never below the counter; the Panda fingers (each joint at
    half the gap) moving 4 mm a step toward the action's sign, stopped by an ``obj`` that wide;
    success once the TCP is ``goal_z`` or higher."""

    action_dim = 12

    def __init__(self, obj: float = 0.0, goal_z: float = 9.0):
        self.obj, self.goal_z = obj, goal_z
        ctl = SimpleNamespace(
            origin_pos=ORIGIN_P,
            origin_ori=ORIGIN_R,
            output_max=np.r_[[0.05] * 3, [0.5] * 3],
        )
        self.robots = [
            SimpleNamespace(
                composite_controller=SimpleNamespace(part_controllers={"right": ctl}),
                robot_model=SimpleNamespace(naming_prefix="robot0_"),
            )
        ]
        model = SimpleNamespace(
            camera_name2id=CAMERAS.index, cam_fovy=np.array([60.0, 60.0, 75.0])
        )
        self.sim = SimpleNamespace(model=model, data=SimpleNamespace())
        self.sim.data.cam_xmat = np.array([np.eye(3).reshape(-1)] * 3)
        self.reset()

    def reset(self):
        self.pos = ORIGIN_P + ORIGIN_R @ np.array([0.5, 0.0, 0.3])
        self.rot = ORIGIN_R @ DOWN
        self.gap = 0.08
        self.actions: list[np.ndarray] = []
        return self.obs()

    def obs(self):
        self.sim.data.cam_xpos = np.array(
            [[1.0, 2.0, 2.4], [1.0, 2.0, 2.4], self.pos + [0, 0, 0.05]]
        )
        return {
            "robot0_eef_pos": self.pos.copy(),
            "robot0_eef_quat": xyzw(self.rot),
            "robot0_gripper_qpos": np.array([self.gap / 2, -self.gap / 2]),
            "robot0_base_pos": ORIGIN_P.copy(),
            "robot0_base_quat": xyzw(ORIGIN_R),
            "robot0_base_to_eef_pos": ORIGIN_R.T @ (self.pos - ORIGIN_P),
        }

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        self.actions.append(a)
        new = self.pos + ORIGIN_R @ (a[:3] * 0.05)
        new[2] = max(new[2], COUNTER_Z + 0.002)
        self.pos = new
        self.rot = rig.axis_angle_matrix(ORIGIN_R @ (a[3:6] * 0.5)) @ self.rot
        self.gap = float(
            np.clip(self.gap - 0.004 * np.clip(a[6], -1, 1), self.obj, 0.08)
        )
        return self.obs(), 0.0, False, {}

    def _check_success(self):
        return bool(self.pos[2] >= self.goal_z)


def facade(**kw) -> RoboCasaEnvFacade:
    f = object.__new__(RoboCasaEnvFacade)
    f._mm_setup(**{k: kw.pop(k) for k in ("mm_home", "mm_truth") if k in kw})
    RpcFacade.__init__(f)
    f.env = _Omron(**kw)
    f.seed, f.scene = 0, None
    f._steps, f._episode_start, f._mm_success = 0, 0, False
    f._run_start, f._run_obs, f._run_frames = 0, None, []
    f._pos_jac = f._fwd_offset = f._home_rel = None
    f._motion_frames, f._policy_frames, f._motion_steps = [], [], 0
    f._recording, f._sam3_url, f._sam3 = False, "", None
    f.ground_truth_poses = lambda names=None: {"poses": {}}

    def render(camera_name, height, width, depth):
        rgb = np.zeros((height, width, 3), np.uint8)
        if not depth:
            return rgb
        z = f.env.sim.data.cam_xpos[CAMERAS.index(camera_name)][2] - COUNTER_Z
        return rgb, np.full((height, width), z, np.float32)

    f.render_camera = render
    RoboCasaEnvFacade._register_rpc(f)
    f.reset()
    f.env.actions.clear()
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


def test_the_base_frame_is_the_arm_controllers_origin():
    f = facade()
    o = f.mm_observe()
    assert o["pose"]["position_m"] == pytest.approx([0.5, 0.0, 0.3])
    assert rig.jaws_down(np.asarray(o["pose"]["rotation"]))
    assert o["base_pose"]["body"] == "robot0_osc_origin"
    assert o["caps"]["max_translation_m"] == 0.05 and o["caps"]["reach_m"] == 0.78
    assert o["table_z_m"] == pytest.approx(COUNTER_Z - ORIGIN_P[2], abs=0.003)
    assert set(o["cameras"]) == {"scene", "side", "wrist"}
    assert o["env_steps"] == 0


def test_a_tick_is_one_arm_step_the_base_and_torso_still():
    f = facade()
    w0 = f.env.pos.copy()
    out = move(f, [1, 0, 0], 40, speed=0.025)
    assert out["outcome"] == "done"
    assert f.env.pos - w0 == pytest.approx(ORIGIN_R @ [0.04, 0, 0], abs=rig.ARRIVE_M)
    acts = np.stack(f.env.actions)
    assert acts[0][:3] == pytest.approx([0.025, 0.0, 0.0])
    assert np.all(acts[:, 7:11] == 0) and np.all(acts[:, 11] == -1), (
        "arm mode, base still"
    )
    assert out["env_steps"] == out["ticks"] == len(acts)
    assert len(out["frames"]) == out["ticks"]


def test_a_rotation_turns_about_the_bases_axis():
    f = facade()
    _p, r0 = f._mm_pose()
    goal = rig.axis_angle_matrix([0, 0, math.radians(30)]) @ r0
    out = f.mm_run_step(
        seg={
            "kind": "rotate",
            "goal_rotation": goal.tolist(),
            "speed": math.radians(60),
        }
    )
    assert out["turned_rad"] == pytest.approx(math.radians(30), abs=0.03)


def test_gripper_channel_zero_holds_the_incremental_gripper():
    f = facade(obj=0.03)
    held = f.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert held["grasp"]["holding"] is True
    part = f.mm_run_step(seg={"kind": "gripper", "state": "open", "open_to_mm": 50})
    gap = part["detail"]["opening_mm"]
    assert part["gripper"]["commanded"] == "hold" and 50 <= gap <= 58
    assert f.env.actions[-1][6] == 0.0
    move(f, [0, 0, 1], 20)
    assert f._mm_opening() * 1000 == pytest.approx(gap, abs=0.5)


def test_success_latches_and_home_resets():
    f = facade(goal_z=ORIGIN_P[2] + 0.319, mm_home="reset")
    out = move(f, [0, 0, 1], 40)
    assert out["terminated"] and out["episode_over"]
    f.env.goal_z = 9.0  # the task undone: the latch holds
    assert f.mm_observe(images=False)["terminated"]
    home = f.mm_run_step(seg={"kind": "home"})
    assert home["episode_reset"] and home["env_steps"] == 0 and not home["terminated"]


def test_the_mm_rpcs_are_registered_and_declared_internal():
    f = facade()
    assert {"env.mm_observe", "env.mm_run_step"} <= set(f._rpc)
    f._manifest_ready()
