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

"""Metaworld's MotorMind mode (env.mm_observe / env.mm_run_step over utils/motormind.py) on a
Sawyer-like stand-in: a hand welded to its mocap target with lag, position-servo fingers whose
effort 0 opens them. The base frame, the four-step tick, the servo on the mocap target and its
settle, the relaxed target after a block, the refused rotation, the opening from the closed stop
and the held gap, home and the registration."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.robots.metaworld import env_server as mw

HAND_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
OFFSET = np.array([0.004, 0.001, -0.05])


class _Sawyer:
    """Metaworld's env as the facade reads it: the mocap target moves 1 cm per unit action
    (clipped to its box), the hand covers half of the way to it each step and stops on the table
    (z 0.0); r_close / l_close are position servos (targets effort and -effort, clipped to their
    ranges) moving 2 mm a step; success when the TCP is within 2 cm of ``goal``."""

    frame_skip = 5

    def __init__(self, goal=(1.0, 1.0, 1.0)):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.mocap_low = np.array([-0.5, 0.4, -0.05])
        self.mocap_high = np.array([0.5, 1.0, 0.5])
        self.model = SimpleNamespace(
            opt=SimpleNamespace(timestep=0.0025),
            joint=lambda name: SimpleNamespace(
                qposadr=[{"r_close": 0, "l_close": 1}[name]],
                dofadr=[{"r_close": 0, "l_close": 1}[name]],
            ),
        )
        self._target_pos = self.goal
        self.offset = OFFSET.copy()
        self.reset()

    def reset(self):
        self.hand = np.array([0.0, 0.6, 0.15])
        self.mocap = self.hand - OFFSET
        self.qpos = np.zeros(2)
        self.actions: list[np.ndarray] = []
        self.data = SimpleNamespace(
            mocap_pos=np.array([self.mocap]),
            qpos=self.qpos,
            # The fingers stop where they are sent: they never coast.
            qvel=np.zeros(2),
            body=self._body,
        )
        return np.zeros(39), {}

    def _body(self, name):
        if name == "right_arm_base_link":
            return SimpleNamespace(xpos=np.zeros(3), xmat=np.eye(3).reshape(-1))
        if name == "hand":
            return SimpleNamespace(xpos=self.hand, xmat=HAND_DOWN.reshape(-1))
        raise KeyError(name)

    @property
    def tcp_center(self):
        return self.hand.copy()

    def get_body_com(self, name):
        width = 0.0943 - (self.qpos[0] - self.qpos[1])
        side = 1 if name == "leftpad" else -1
        return self.hand + [0, side * width / 2, 0]

    def seed(self, s):
        pass

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        self.actions.append(a)
        m = np.clip(
            self.data.mocap_pos[0] + a[:3] * 0.01, self.mocap_low, self.mocap_high
        )
        self.data.mocap_pos = np.array([m])
        self.hand = self.hand + 0.5 * (m + self.offset - self.hand)
        self.hand[2] = max(self.hand[2], 0.0)
        targets = [np.clip(a[3], 0.0, 0.04), np.clip(-a[3], -0.03, 0.0)]
        for i, t in enumerate(targets):
            self.qpos[i] += np.clip(t - self.qpos[i], -0.002, 0.002)
        d = float(np.linalg.norm(self.hand - self.goal))
        return np.zeros(39), 0.0, False, False, {"success": float(d < 0.02)}

    def close(self):
        pass


def facade(goal=(1.0, 1.0, 1.0), **mm) -> mw.MetaworldEnvFacade:
    f = object.__new__(mw.MetaworldEnvFacade)
    f._mm_setup(**mm)
    mw.BaseEnvFacade.__init__(f)
    f._env = _Sawyer(goal)
    f._seed = 0
    f._view_size = 4
    f._renderers = {}
    f._obs = np.zeros(39)
    f._info = {}
    f._success_once = False
    f._gripper_effort = mw.OPEN
    f._closed = False
    f._box = None
    f._meta = {"task": "reach-v3", "seed": 0}
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    f._render = lambda camera, h, w, depth=False: frame
    f.ground_truth_poses = lambda names=None: {"poses": {}}

    def shots(cameras, size, depth):
        # A camera 1.5 m above the table looking straight down, the wrist on the hand.
        out = {}
        for name in cameras:
            pos = (
                f._env.hand + [0, 0, 0.1]
                if name == "wrist"
                else np.array([0, 0.6, 1.5])
            )
            out[name] = {
                "rgb": np.full((size, size, 3), 90, dtype=np.uint8),
                "K": rig.intrinsic(60.0, size, size),
                "cam2world": rig.cam2world(pos, np.eye(3)),
                **(
                    {"depth": np.full((size, size), pos[2], np.float32)}
                    if name in depth
                    else {}
                ),
            }
        return out

    f._mm_shots = shots
    f.reset()
    # The episode's first mm call lets the hand come to rest (four quiet ticks here).
    f.mm_observe(images=False)
    f._env.actions.clear()
    return f


def move(f, axis_base, mm: float, speed=0.05, push=False):
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
            "push": push,
        }
    )


def test_the_base_frame_faces_the_table_and_the_tool_points_down():
    f = facade()
    o = f.mm_observe(images=False)
    # Base +X is world +y (away from the stand), +Y world -x (the robot's left).
    assert o["pose"]["position_m"] == pytest.approx([0.6, 0.0, 0.15])
    assert np.allclose(np.asarray(o["pose"]["rotation"])[:, 2], [0, 0, -1])
    assert rig.jaws_down(np.asarray(o["pose"]["rotation"]))
    assert o["base_pose"]["body"] == "right_arm_base_link"
    caps = o["caps"]
    assert caps["max_translation_m"] == pytest.approx(0.04)  # four 1 cm steps a tick
    assert caps["tilt_limits_deg"] == {
        "roll_cw": 0.0,
        "roll_ccw": 0.0,
        "pitch_down": 0.0,
        "pitch_up": 0.0,
    }
    assert any("rotate is refused" in n for n in caps["notes"])
    assert caps["gripper_open_span_m"] == pytest.approx(0.0706)
    assert o["table_z_m"] == pytest.approx(0.0, abs=0.003)
    assert o["camera_names"] == ["scene", "side", "wrist"]
    # The hand came to rest before the episode's first observation: four quiet ticks.
    assert o["settle_ticks"] == 4 and o["env_steps"] == 16


def test_a_tick_is_four_control_steps_of_the_mocap_target():
    f = facade()
    w0 = f._env.hand.copy()
    out = move(f, [1, 0, 0], 40)  # forward
    assert out["outcome"] == "done", out["message"]
    assert np.allclose(f._env.hand - w0, [0, 0.04, 0], atol=rig.ARRIVE_M)
    assert out["env_steps"] - 16 == 4 * out["ticks"] == len(f._env.actions)
    # 50 mm/s: 2.5 mm a tick, 0.625 mm per step, as a mocap action of 0.0625 along world +y.
    # A lagging hand first comes to rest (here it is: four quiet zero ticks).
    assert out["rest_ticks"] == 4
    assert f._env.actions[16][:3] == pytest.approx([0.0, 0.0625, 0.0])
    # The servo steers the mocap target; the lagging hand settles under the judge.
    assert out["settle_ticks"] > 0
    assert out["ticks"] == out["rest_ticks"] + out["servo_ticks"]
    left = move(f, [0, 1, 0], 20)
    assert left["outcome"] == "done"
    assert f._env.hand[0] == pytest.approx(w0[0] - 0.02, abs=rig.ARRIVE_M)


def test_the_hands_offset_from_its_target_is_read_at_each_segments_start():
    f = facade()
    # Where the hand rests relative to its mocap target changes (the arm sags differently):
    # once it has settled, a move still arrives along its axis.
    f._env.offset = OFFSET + [0.005, -0.003, 0.0]
    f.mm_run_step(seg={"kind": "wait", "seconds": 1.0})
    out = move(f, [1, 0, 0], 40)
    assert out["along_axis_m"] == pytest.approx(0.04, abs=rig.ARRIVE_M)
    assert out["lateral_m"] < 0.001


def test_a_descent_onto_the_table_is_blocked_and_the_target_comes_back_to_the_hand():
    f = facade()
    out = move(f, [0, 0, -1], 300)
    assert (out["outcome"], out["abort_reason"]) == ("limited", "blocked")
    assert f._env.data.mocap_pos[0] + OFFSET == pytest.approx(f._env.hand)
    assert out["hold_ticks"] == rig.HOLD_TICKS


def test_every_rotation_is_refused_nothing_moved():
    f = facade()
    _p, r0 = f._mm_pose()
    goal = rig.axis_angle_matrix([0, 0, math.radians(20)]) @ r0
    out = f.mm_run_step(
        seg={"kind": "rotate", "goal_rotation": goal.tolist(), "speed": 1.0}
    )
    assert out["outcome"] == "refused" and out["ticks"] == 0
    assert f._env.actions == []


def test_the_opening_counts_from_the_closed_stop_and_a_hold_keeps_the_gap():
    f = facade()
    assert f.mm_observe(images=False)["gripper"]["opening_mm"] == pytest.approx(70.6)
    closed = f.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert closed["detail"]["opening_mm"] == pytest.approx(0.6, abs=0.1)
    assert closed["grasp"]["holding"] is False
    assert all(a[3] == mw.CLOSE for a in f._env.actions)
    part = f.mm_run_step(seg={"kind": "gripper", "state": "open", "open_to_mm": 30})
    gap = part["detail"]["opening_mm"]
    # Checked every control step, not every tick: the fingers stop within a step (4 mm) of it.
    assert 30 <= gap <= 34.5 and part["gripper"]["commanded"] == "hold"
    # Effort 0 would open these fingers: a hold commands half the gap to each.
    hold = f._env.actions[-1][3]
    assert hold == pytest.approx((f._env.qpos[0] - f._env.qpos[1]) / 2) and hold > 0
    assert f._mm_opening() * 1000 == pytest.approx(gap, abs=0.5)
    move(f, [0, 0, 1], 20)
    assert f._mm_opening() * 1000 == pytest.approx(gap, abs=0.5)
    f.mm_run_step(seg={"kind": "gripper", "state": "open"})
    assert f._env.actions[-1][3] == mw.OPEN


def test_success_latches_and_home_resets_with_mm_home_reset():
    f = facade(goal=(0.0, 0.62, 0.15), mm_home="reset")
    out = move(f, [1, 0, 0], 30)
    assert out["terminated"] and out["episode_over"]
    assert move(f, [1, 0, 0], 10)["outcome"] == "refused"
    home = f.mm_run_step(seg={"kind": "home"})
    assert home["outcome"] == "done" and home["episode_reset"]
    # The new episode's steps are its settle ticks only.
    assert home["env_steps"] == 16 and not home["terminated"]
    assert home["observation"]["settle_ticks"] == 4
    assert home["pose_after"]["position_m"] == pytest.approx([0.6, 0.0, 0.15])
    assert facade().mm_run_step(seg={"kind": "home"})["outcome"] == "refused"


def test_the_mm_rpcs_are_registered_and_declared_internal():
    f = facade()
    assert {"env.mm_observe", "env.mm_run_step"} <= set(f._rpc)
    f._manifest_ready()
    assert "env.mm_run_step" in mw.MOTIONS
