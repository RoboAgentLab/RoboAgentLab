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

"""RoboTwin's MotorMind mode (env.mm_observe / env.mm_run_step over utils/motormind.py) on a
kinematic stand-in for RLinf's RoboTwin env: one arm's base frame, ``ee`` ticks that move the TCP
and turn the eef about it (the other arm held), ``qpos`` ticks that ramp the gripper target, the
measured gripper value, the action budget, and non-square camera renders."""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_robotwin_code import env_server

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.utils.rpc import RpcFacade

RoboTwinEnvFacade = env_server.RoboTwinEnvFacade
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
#: The eef's approach (+x) points down, its +y along the world's +y.
EE_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
TCP_OFFSET = np.array([0.12, 0.0, 0.0])  # along the eef's +x, as a gripper's fingertips
TABLE_Z = 0.74
H, W = 240, 320


def wxyz(r) -> list[float]:
    return [float(v) for v in rig.matrix_quat_wxyz(r)]


class _Twin:
    """RLinf's RoboTwin env as the facade reads it: per arm an eef pose that an ``ee`` action
    puts where it asks (unless below the table), a gripper target that a ``qpos`` action sets and
    a measured value that follows it 0.25 a step but stops at ``obj`` closed; the action count and
    budget; a head camera 1.5 m up looking down and a wrist camera on each eef."""

    def __init__(self, obj: float = 0.0, limit: int | None = None):
        self.obj, self.limit = obj, limit
        self.count = 0
        self.ee = {
            "left": np.array([-0.25, 0.0, 0.95]),
            "right": np.array([0.25, 0.0, 0.95]),
        }
        self.rot = {"left": EE_DOWN.copy(), "right": EE_DOWN.copy()}
        self.g = {"left": 1.0, "right": 1.0}
        self.real = {"left": 1.0, "right": 1.0}
        self.actions: list[tuple[str, np.ndarray]] = []

    def tcp(self, arm):
        return self.ee[arm] + self.rot[arm] @ TCP_OFFSET

    def status(self):
        return {
            "eval_success": False,
            "take_action_cnt": self.count,
            "step_lim": self.limit,
            "actual_seed": 0,
        }

    def _state(self):
        out = {
            "qpos_target14": np.r_[
                np.zeros(6), self.g["left"], np.zeros(6), self.g["right"]
            ]
        }
        for arm in ("left", "right"):
            out[f"{arm}_eef_pose"] = np.r_[self.ee[arm], wxyz(self.rot[arm])]
            out[f"{arm}_tcp_pose"] = np.r_[self.tcp(arm), wxyz(self.rot[arm])]
            out[f"{arm}_gripper"] = self.g[arm]
            out[f"{arm}_gripper_real"] = self.real[arm]
        return out

    def robot_state(self, env_id=0):
        return {"robot_state": self._state(), "episode_status": self.status()}

    def arm_base_pose(self, arm, env_id=0):
        x = -0.3 if arm == "left" else 0.3
        return [x, -0.4, 0.75, *wxyz(RZ90)]

    def step(self, action, *, action_type="qpos", env_id=0):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        self.actions.append((action_type, a))
        self.count += 1
        for k, arm in enumerate(("left", "right")):
            if action_type == "ee":
                pose = a[8 * k : 8 * k + 8]
                rot = rig.quat_xyzw_matrix([*pose[4:7], pose[3]])
                if (pose[:3] + rot @ TCP_OFFSET)[2] >= TABLE_Z + 0.002:
                    self.ee[arm], self.rot[arm] = pose[:3].copy(), rot
                self.g[arm] = float(pose[7])
            else:
                self.g[arm] = float(a[7 * k + 6])
            self.real[arm] = float(
                np.clip(self.g[arm], self.obj, 1.0)
                if abs(self.g[arm] - self.real[arm]) <= 0.25
                else self.real[arm] + 0.25 * np.sign(self.g[arm] - self.real[arm])
            )
            self.real[arm] = max(self.real[arm], self.obj)
        one = np.zeros((1,))
        return (
            {"main_images": np.zeros((1, 2, 2, 3), np.uint8)},
            one,
            one.astype(bool),
            one.astype(bool),
            {"robot_state": self._state(), "episode_status": self.status()},
        )

    def _cam_pos(self, name):
        return (
            self.tcp(name.split("_")[0]) + [0, 0, 0.05]
            if "wrist" in name
            else np.array([0.0, 0.0, 1.5])
        )

    def render_camera(self, camera_name, depth=False, env_id=0):
        rgb = np.full((H, W, 3), 70, np.uint8)
        if not depth:
            return rgb
        return rgb, np.full((H, W), self._cam_pos(camera_name)[2] - TABLE_Z, np.float32)

    def get_camera_meta(self, camera_name, env_id=0):
        c2w = rig.cam2world(self._cam_pos(camera_name), np.eye(3))
        return {
            "intrinsic_K": np.array([[200.0, 0, W / 2], [0, 200.0, H / 2], [0, 0, 1]]),
            "extrinsic_cv": np.linalg.inv(c2w)[:3, :4],
        }

    def object_poses(self, env_id=0):
        return {}


def facade(arm="right", **kw) -> RoboTwinEnvFacade:
    f = object.__new__(RoboTwinEnvFacade)
    f._mm_side, f._mm_g = arm, None
    f._mm_setup(**{k: kw.pop(k) for k in ("mm_home", "mm_truth") if k in kw})
    RpcFacade.__init__(f)
    f._env, f._metadata = _Twin(**kw), {"seed": 0}
    f._run_steps, f._run_info, f._run_ref, f._run_frames = 0, None, None, []
    RoboTwinEnvFacade._register_rpc(f)
    f._mm_new_episode()
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


def test_the_arms_base_frame_its_tcp_and_the_cameras_own_sizes():
    f = facade()
    o = f.mm_observe()
    tcp = f._env.tcp("right")
    assert o["pose"]["position_m"] == pytest.approx(RZ90.T @ (tcp - [0.3, -0.4, 0.75]))
    assert rig.jaws_down(np.asarray(o["pose"]["rotation"]))
    assert o["base_pose"]["body"] == "right_arm_base"
    assert o["table_z_m"] == pytest.approx(TABLE_Z - 0.75, abs=0.003)
    assert set(o["cameras"]) == {"scene", "wrist"}
    assert o["cameras"]["wrist"]["camera"] == "right_wrist"
    assert (o["cameras"]["scene"]["width"], o["cameras"]["scene"]["height"]) == (W, H)
    assert o["caps"]["reach_m"] == 0.6 and o["caps"]["supports_home"] is False
    assert o["episode_status"]["take_action_cnt"] == 0
    assert set(o["robot_state"]) >= {"right_tcp_pose", "left_eef_pose"}


def test_a_tick_is_one_ee_action_of_its_arm_the_other_held():
    f = facade()
    left0 = f._env.ee["left"].copy()
    w0 = f._env.tcp("right")
    out = move(f, [1, 0, 0], 40)  # forward is world +y
    assert out["outcome"] == "done"
    assert f._env.tcp("right") - w0 == pytest.approx([0, 0.04, 0], abs=rig.ARRIVE_M)
    assert out["env_steps"] == out["ticks"] == len(f._env.actions)
    assert {kind for kind, _a in f._env.actions} == {"ee"}
    assert np.allclose(f._env.ee["left"], left0), "the other arm did not move"


def test_a_rotation_turns_the_eef_about_the_tcp():
    f = facade()
    p0, r0 = f._mm_pose()
    goal = rig.axis_angle_matrix([0, 0, math.radians(30)]) @ r0
    out = f.mm_run_step(
        seg={"kind": "rotate", "goal_rotation": goal.tolist(), "speed": 1.0}
    )
    assert out["outcome"] == "done"
    assert out["turned_rad"] == pytest.approx(math.radians(30), abs=0.03)
    assert out["moved_m"] < 0.003, "the tool point stayed"


def test_the_gripper_ramps_its_target_and_reads_the_measured_value():
    f = facade()
    empty = f.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert empty["ticks"] == 32 and {k for k, _a in f._env.actions} == {"qpos"}
    assert [a[13] for _k, a in f._env.actions[:3]] == [0.875, 0.75, 0.625]
    assert empty["grasp"]["holding"] is False
    g = facade(obj=0.5)
    held = g.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert held["grasp"]["holding"] is True
    assert held["detail"]["opening_mm"] == pytest.approx(35.0)
    part = g.mm_run_step(seg={"kind": "gripper", "state": "open", "open_to_mm": 50})
    assert part["gripper"]["commanded"] == "hold"
    target = g._env.g["right"]
    move(g, [0, 0, 1], 20)
    assert g._env.g["right"] == target, "a hold keeps the gripper target"


def test_the_action_budget_ends_the_episode():
    f = facade(limit=10)
    out = move(f, [1, 0, 0], 50)
    assert out["truncated"] and out["episode_over"] and out["ticks"] == 10
    assert move(f, [1, 0, 0], 10)["outcome"] == "refused"


def test_the_left_arm_and_the_registration():
    f = facade("left")
    assert f.mm_observe(images=False)["base_pose"]["body"] == "left_arm_base"
    assert {"env.mm_observe", "env.mm_run_step"} <= set(f._rpc)
    with pytest.raises(ValueError, match="arm must be"):
        RoboTwinEnvFacade(_Twin(), metadata={"seed": 0}, mm_arm="both")
