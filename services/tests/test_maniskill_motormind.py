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

"""ManiSkill's MotorMind mode (env.mm_observe / env.mm_run_step over utils/motormind.py) on a
kinematic stand-in of MM_CONTROL_MODE: the arm's base frame and capability fields, one control step
a tick (a unit realises 0.05 m or 0.5 rad, the rotation negated as ManiSkill 3.0.1 turns it), the
absolute-target fingers ramped open and held at their joint, the scene view padded square, success,
home, and the arms and rigs that refuse the mode."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.robots.maniskill import env_server as ms

BASE_P = np.array([-0.615, 0.0, 0.0])
#: The Panda hand pointing down (x along world y, z down), as ManiSkill resets it.
DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _wxyz(r) -> np.ndarray:
    return np.asarray(rig.matrix_quat_wxyz(r)).reshape(1, 4)


def _down_camera(eye) -> np.ndarray:
    """World-to-camera (3x4, OpenCV) of a camera at ``eye`` looking straight down."""
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    c2w[:3, 3] = eye
    return np.linalg.inv(c2w)[:3]


class _Arm:
    """ManiSkill's env in MM_CONTROL_MODE as the facade reads it: the TCP moves 0.05 m and turns
    -0.5 rad (about the commanded base axis) per action unit; the Panda's fingers chase their
    absolute target (normalised over [-0.01, 0.04] m) up to 10 mm a step and stop on an object
    ``obj`` wide; success once the TCP is ``goal_z`` or higher. The agentview (640x480) and wrist
    (256) look straight down; the render camera answers on demand."""

    def __init__(self, obj: float = 0.0, goal_z: float = 9.0):
        self.obj, self.goal_z = obj, goal_z
        self.actions: list[np.ndarray] = []
        self.renders = 0
        arm = self
        cfg = SimpleNamespace(normalize_action=True, lower=-0.01, upper=0.04)
        # Its qpos follows the fingers (``_sync``).
        gripper = SimpleNamespace(
            config=cfg, control_joint_indices=np.array([0]), qpos=None
        )
        self._gripper = gripper
        robot = SimpleNamespace(
            pose=SimpleNamespace(p=BASE_P.reshape(1, 3), q=np.array([[1.0, 0, 0, 0]])),
            root=SimpleNamespace(name="panda_link0"),
            get_qpos=lambda: np.concatenate([np.zeros(7), [arm.q, arm.q]]).reshape(
                1, 9
            ),
        )
        self.agent = SimpleNamespace(
            robot=robot,
            tcp=SimpleNamespace(pose=None),
            controller=SimpleNamespace(controllers={"gripper": gripper}),
        )
        side = SimpleNamespace(
            get_params=lambda: {
                "extrinsic_cv": _down_camera([0.0, 0.7, 0.5]).reshape(1, 3, 4),
                "intrinsic_cv": np.array(
                    [[[600.0, 0, 384], [0, 600.0, 384], [0, 0, 1]]]
                ),
            }
        )
        self.unwrapped = SimpleNamespace(
            agent=self.agent,
            evaluate=lambda: {"success": np.array([arm.p[2] >= arm.goal_z])},
            render_rgb_array=self._render_side,
            scene=SimpleNamespace(human_render_cameras={ms.MM_SIDE_CAMERA: side}),
        )
        self.reset()

    def _render_side(self, name):
        self.renders += 1
        return np.full((1, 768, 768, 3), 90, dtype=np.uint8)

    def _sync(self):
        self.agent.tcp.pose = SimpleNamespace(p=self.p.reshape(1, 3), q=_wxyz(self.r))
        self._gripper.qpos = np.array([[self.q, self.q]])

    def _obs(self):
        self._sync()
        wrist_eye = self.p + [0.0, 0.0, 0.05]
        k = np.array([[[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]])
        kw = np.array([[[128.0, 0, 128], [0, 128.0, 128], [0, 0, 1]]])
        return {
            "sensor_data": {
                "base_camera": {
                    "rgb": np.full((1, 480, 640, 3), 50, dtype=np.uint8),
                    "depth": np.full((1, 480, 640, 1), 1000, dtype=np.int16),
                },
                "hand_camera": {
                    "rgb": np.full((1, 256, 256, 3), 70, dtype=np.uint8),
                    "depth": np.full(
                        (1, 256, 256, 1), round(wrist_eye[2] * 1000), dtype=np.int16
                    ),
                },
            },
            "sensor_param": {
                "base_camera": {
                    "extrinsic_cv": _down_camera([0.0, 0.0, 1.0]).reshape(1, 3, 4),
                    "intrinsic_cv": k,
                },
                "hand_camera": {
                    "extrinsic_cv": _down_camera(wrist_eye).reshape(1, 3, 4),
                    "intrinsic_cv": kw,
                },
            },
        }

    def reset(self, seed=None):
        self.p = BASE_P + [0.627, 0.038, 0.182]
        self.r = DOWN.copy()
        self.q = 0.04
        self.actions.clear()
        return self._obs(), {"success": np.array([False])}

    def step(self, a):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        self.actions.append(a)
        self.p = self.p + np.clip(a[:3], -1, 1) * ms.MM_POS_M
        self.r = rig.axis_angle_matrix(-np.clip(a[3:6], -1, 1) * ms.MM_ROT_RAD) @ self.r
        target = -0.01 + (np.clip(a[6], -1, 1) + 1) / 2 * 0.05
        self.q = float(np.clip(target, self.q - 0.01, self.q + 0.01))
        self.q = max(self.q, self.obj / 2)
        info = {"success": np.array([self.p[2] >= self.goal_z])}
        return self._obs(), np.array([0.0]), np.array([False]), np.array([False]), info


def facade(robot: str = "panda", **kw) -> ms.ManiskillEnvFacade:
    f = object.__new__(ms.ManiskillEnvFacade)
    ms.BaseEnvFacade.__init__(f)
    f._mm_setup(**{k: kw.pop(k) for k in ("mm_home", "mm_truth") if k in kw})
    f._env = _Arm(**kw)
    f._robot = ms.mm_spec(ms.ROBOTS[robot], robot, False)
    f._mm = True
    f._rig = None
    f._cameras = ms.CAMERAS
    f._obs_mode = ms.OBS_MODE
    f._grip = [1.0]
    f._seed, f._settle_steps = 0, 0
    f._meta = {
        "robot": robot,
        "env_id": "StackCube-v1",
        "wrist": True,
        "table_z": 0.0,
        "control_mode": ms.MM_CONTROL_MODE,
        "mm": True,
    }
    f._rgb = lambda obs, name: np.zeros((4, 4, 3), np.uint8)
    f.check_visible = lambda obs: None
    f._mm_place_side = lambda: None
    f.ground_truth_poses = lambda names=None: {"poses": {}}
    f.stop_requested = lambda: False
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


def test_mm_spec_runs_the_pose_mode_and_keeps_move_deltas_metres():
    spec = ms.mm_spec(ms.ROBOTS["panda"], "panda", False)
    assert spec.control_mode == ms.MM_CONTROL_MODE and spec.rot_dims == 3
    # move_delta's 0.1 m per unit in the mode's 0.05 / 0.38 m bound: the same metres commanded.
    a = spec.action(np.array([0.5, 0.0, -0.5]), 1.0)
    assert a == pytest.approx([0.38, 0.0, -0.38, 0.0, 0.0, 0.0, 1.0])
    assert ms.mm_spec(ms.ROBOTS["xarm6_robotiq"], "xarm6_robotiq", False).gripper == (
        -1.0,
        1.0,
    )
    with pytest.raises(ValueError, match="an RLinf rig keeps the pd_ee_delta_pos"):
        ms.mm_spec(ms.ROBOTS["panda"], "panda", True)
    for robot in ("widowxai", "panda_stick", "panda_pair", "widowx250s"):
        with pytest.raises(ValueError, match=f"{robot} {ms.MM_REFUSED[robot][:12]}"):
            ms.mm_spec(ms.ROBOTS[robot], robot, False)
    assert set(ms.MM_ROBOTS) | set(ms.MM_REFUSED) == set(ms.ROBOTS)


def test_add_mm_mode_copies_the_pose_mode_with_the_scaled_bounds():
    def configs(self):
        return {
            "pd_ee_delta_pose": {
                "arm": SimpleNamespace(
                    pos_lower=-0.1,
                    pos_upper=0.1,
                    rot_lower=-0.1,
                    rot_upper=0.1,
                    use_target=False,
                ),
                "gripper": "g",
            },
            "pd_ee_delta_pos": {"arm": "pos", "gripper": "g"},
        }

    cls = type("Agent", (), {"_controller_configs": property(configs)})
    ms.add_mm_mode(cls)
    wrapped = cls._controller_configs
    ms.add_mm_mode(cls)  # idempotent
    assert cls._controller_configs is wrapped
    cfg = cls()._controller_configs
    arm = cfg[ms.MM_CONTROL_MODE]["arm"]
    assert (arm.pos_lower, arm.pos_upper) == pytest.approx((-0.05 / 0.38, 0.05 / 0.38))
    assert (arm.rot_lower, arm.rot_upper) == pytest.approx((-0.5 / 0.39, 0.5 / 0.39))
    assert arm.use_target is False and cfg[ms.MM_CONTROL_MODE]["gripper"] == "g"
    assert (
        cfg["pd_ee_delta_pose"]["arm"].pos_lower == -0.1
    )  # the stock mode is untouched
    assert cfg["pd_ee_delta_pos"]["arm"] == "pos"


def test_the_observation_is_in_the_arms_base_frame_with_three_square_views():
    f = facade()
    o = f.mm_observe()
    assert o["pose"]["position_m"] == pytest.approx([0.627, 0.038, 0.182])
    assert rig.jaws_down(np.asarray(o["pose"]["rotation"]))
    assert o["base_pose"]["body"] == "panda_link0"
    assert o["base_pose"]["position_m"] == pytest.approx(BASE_P)
    caps = o["caps"]
    assert (caps["max_translation_m"], caps["max_rotation_rad"]) == (0.05, 0.5)
    assert caps["gripper_open_span_m"] == 0.08 and caps["reach_m"] == 0.78
    assert o["table_z_m"] == pytest.approx(0.0, abs=0.003)
    assert {n: c["camera"] for n, c in o["cameras"].items()} == {
        "scene": "base_camera",
        "side": "render_camera",
        "wrist": "hand_camera",
    }
    scene = o["cameras"]["scene"]
    # The 640x480 agentview padded to 640x640: the principal point moves down by the 80 px bar.
    assert (scene["width"], scene["height"]) == (640, 640)
    assert scene["intrinsic"][1][2] == 320.0 and scene["intrinsic"][0][2] == 320.0
    assert o["cameras"]["side"]["width"] == 768 and f._env.renders == 1
    assert o["clearance"]["n_points"] > 0
    assert o["env_steps"] == 0 and o["config"] == {"mm_home": "refuse", "truth": None}


def test_a_tick_is_one_control_step_in_the_base_frame_and_the_rotation_is_negated():
    f = facade()
    w0 = f._env.p.copy()
    out = move(f, [1, 0, 0], 50, speed=0.025)
    assert out["outcome"] == "done"
    assert np.allclose(f._env.p - w0, [0.05, 0, 0], atol=rig.ARRIVE_M)
    # 25 mm/s is 1.25 mm a tick: 0.025 of the mode's 0.05 m per unit.
    assert f._env.actions[0][:6] == pytest.approx([0.025, 0, 0, 0, 0, 0])
    assert out["env_steps"] == out["ticks"] == len(f._env.actions)
    assert len(out["frames"]) == out["ticks"]
    _p, r0 = f._mm_pose()
    goal = rig.axis_angle_matrix([0, 0, math.radians(30)]) @ r0
    turned = f.mm_run_step(
        seg={
            "kind": "rotate",
            "goal_rotation": goal.tolist(),
            "speed": math.radians(60),
        }
    )
    assert turned["outcome"] == "done"
    assert turned["turned_rad"] == pytest.approx(math.radians(30), abs=0.03)
    # +yaw is sent as -z: ManiSkill 3.0.1's pose mode turns the tool the other way.
    first = f._env.actions[out["ticks"]]
    assert first[5] < 0 and first[3:5] == pytest.approx([0, 0])


def test_the_fingers_close_open_in_a_ramp_and_hold_at_their_joint():
    f = facade(obj=0.03)
    held = f.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert held["grasp"]["holding"] is True and f._env.actions[-1][6] == -1.0
    f._env.actions.clear()
    part = f.mm_run_step(seg={"kind": "gripper", "state": "open", "open_to_mm": 50})
    # Each opening tick commands the fingers' own target plus open_rate.
    first = f._env.actions[0][6]
    assert first == pytest.approx(2 * (0.015 + 0.01) / 0.05 - 1 + 0.2)
    assert part["gripper"]["commanded"] == "hold"
    gap = part["detail"]["opening_mm"]
    assert 50 - 10 <= gap <= 50
    move(f, [0, 0, 1], 20)
    assert f._mm_opening() * 1000 == pytest.approx(gap)
    full = f.mm_run_step(seg={"kind": "gripper", "state": "open"})
    assert full["gripper"]["opening_mm"] == pytest.approx(80.0)


def test_success_latches_and_home_resets_with_mm_home_reset():
    f = facade(goal_z=0.182 + 0.03, mm_home="reset")
    out = move(f, [0, 0, 1], 40)
    assert out["terminated"] and out["episode_over"]
    assert move(f, [0, 0, 1], 10)["outcome"] == "refused"
    home = f.mm_run_step(seg={"kind": "home"})
    assert home["outcome"] == "done" and home["episode_reset"]
    assert (home["env_steps"], home["terminated"], home["episode_id"]) == (0, False, 2)
    assert facade().mm_run_step(seg={"kind": "home"})["outcome"] == "refused"


def test_a_server_without_mm_refuses_the_mm_calls_and_the_rpcs_are_declared():
    f = facade()
    assert {"env.mm_observe", "env.mm_run_step"} <= set(f._rpc)
    f._manifest_ready()
    plain = facade()
    plain._mm = False
    plain._mm_rig = None
    plain._meta["control_mode"] = "pd_ee_delta_pos"
    with pytest.raises(RuntimeError, match="started with --mm .*runs pd_ee_delta_pos"):
        plain.mm_observe()
