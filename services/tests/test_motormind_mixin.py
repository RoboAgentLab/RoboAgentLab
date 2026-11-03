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

"""MotorMind mode's env-server mixin (utils/motormind.py) on a kinematic arm: the base frame, the
segments (speed, blocked, contact, skid, rotation), the gripper (close, empty, held, open-to and
hold), wait, home, the episode's end, the step count and success a reply carries, the video
frames, the truth log, a hand that cannot rotate and a hand that lags its target. The same checks
as LIBERO's (test_motormind_rig.py) where the semantics are the same."""

from __future__ import annotations

import base64
import io
import json
import math

import numpy as np
import pytest
from PIL import Image

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.utils.motormind import MotorMindMixin


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )


DOWN = np.diag([1.0, -1.0, -1.0])
BASE_P = np.array([-0.5, 0.2, 0.8])
BASE_R = rz(90.0)  # turned, so base and world axes differ
TABLE_Z = 0.8


def decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


class Arm(MotorMindMixin):
    """A kinematic arm: each tick moves the tool by the commanded world increments (two env steps
    a tick), never below the table, with an optional sideways drift per metre and an optional
    lag (the tool covers ``1 - lag`` of the way to its commanded target per tick); fingers move
    8 mm a tick toward closed (on an object ``obj`` wide) or open, or stay; success latches at
    ``success_at`` env steps, truncation at ``max_steps``."""

    def __init__(
        self,
        *,
        rotates: bool = True,
        fingers: bool = True,
        lag: float = 0.0,
        obj: float = 0.0005,
        success_at: int = 0,
        max_steps: int = 10_000,
        **kw,
    ):
        self.rotates, self.fingers, self.lag = rotates, fingers, lag
        self.obj, self.success_at, self.max_steps = obj, success_at, max_steps
        self.stop = False
        self.resets = 0
        self._mm_setup(**kw)
        self.reset()

    def stop_requested(self) -> bool:
        return self.stop

    def reset(self):
        self.resets += 1
        self.pos = BASE_P + BASE_R @ np.array([0.45, 0.0, 0.3])
        self.target = self.pos.copy()
        self.rot = BASE_R @ DOWN
        self.width = 0.078
        self.steps = 0
        self.success = False
        self.drift = 0.0
        self._mm_new_episode()

    # -- hooks --

    def _mm_static(self):
        return {
            "body": "base_link",
            "pos": BASE_P.tolist(),
            "xmat": BASE_R.tolist(),
            "max_translation_m": 0.05,
            "max_rotation_rad": 0.5 if self.rotates else 0.0,
            "fingers": self.fingers,
            "caps": {
                "gripper_open_span_m": 0.078,
                "gripper_pad_length_m": 0.02,
                "reach_m": 0.8,
                "comfortable_reach_m": 0.7,
                "tilt_limits_deg": {
                    "roll_cw": 90.0,
                    "roll_ccw": 90.0,
                    "pitch_down": 90.0,
                    "pitch_up": 50.0,
                },
            },
        }

    def _mm_tool(self):
        return self.pos.copy(), self.rot.copy()

    def _mm_opening(self):
        return self.width

    def _mm_tick(self, dp, dr, grip, until=None):
        for _ in range(2):
            self.steps += 1
            self.success |= bool(self.success_at) and self.steps >= self.success_at
        self.target = self.target + np.asarray(dp)
        new = self.pos + (1.0 - self.lag) * (self.target - self.pos)
        new[0] += self.drift * float(np.linalg.norm(dp))
        new[2] = max(new[2], TABLE_Z + 0.002)
        self.pos = new
        if not self.lag:
            self.target = new.copy()
        if self.rotates:
            self.rot = rig.axis_angle_matrix(dr) @ self.rot
        if grip > 0:
            self.width = max(self.obj, self.width - 0.008)
        elif grip < 0:
            self.width = min(0.078, self.width + 0.008)

    def _mm_target(self):
        return self.target.copy() if self.lag else None

    def _mm_set_target(self, p):
        self.target = np.asarray(p, dtype=np.float64).copy()

    def _mm_status(self):
        return self.success, self.steps >= self.max_steps, self.steps

    def _mm_cameras(self):
        return {"scene": "overhead", "wrist": "hand_cam"}

    def _mm_shots(self, cameras, size, depth):
        out = {}
        for name in cameras:
            pos = self.pos if name == "wrist" else np.array([0.0, 0.0, 1.5])
            fovy = 75.0 if name == "wrist" else 45.0
            shot = {
                "rgb": np.full((size, size, 3), 100, dtype=np.uint8),
                "K": rig.intrinsic(fovy, size, size),
                "cam2world": rig.cam2world(pos, np.eye(3)),
            }
            if name in depth:
                shot["depth"] = np.full((size, size), pos[2] - TABLE_Z, np.float32)
            out[name] = shot
        return out

    def _mm_video_frame(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def _mm_reset_scene(self):
        self.reset()

    def _mm_truth_extra(self):
        return {"success_now": self.success}


def base_pose(a: Arm):
    return a._mm_pose()


def translate(a: Arm, axis_base, mm: float, speed=0.05, push=False, **kw):
    p0, r0 = base_pose(a)
    ax = np.asarray(axis_base, dtype=np.float64)
    return a.mm_run_step(
        seg={
            "kind": "translate",
            "goal_position_m": (p0 + ax * mm / 1000).tolist(),
            "goal_rotation": r0.tolist(),
            "axis": ax.tolist(),
            "distance_m": mm / 1000,
            "speed": speed,
            "push": push,
        },
        **kw,
    )


def test_observation_is_in_the_base_frame_with_the_robots_views_and_caps():
    a = Arm()
    out = a.mm_observe()
    assert np.allclose(out["pose"]["position_m"], [0.45, 0.0, 0.3])
    assert np.allclose(out["pose"]["rotation"], DOWN)
    assert out["base_pose"]["body"] == "base_link"
    assert out["table_z_m"] == pytest.approx(0.0, abs=0.001)
    caps = out["caps"]
    assert caps["max_translation_m"] == 0.05 and caps["max_speed_m_s"] == pytest.approx(
        1.0
    )
    assert caps["reach_m"] == 0.8 and caps["gripper_open_span_m"] == 0.078
    assert not caps["supports_home"] and "no home" in caps["notes"][-1]
    assert set(out["cameras"]) == {"scene", "wrist"}
    assert out["cameras"]["wrist"]["aim_px"] == pytest.approx([384, 384], abs=1e-6)
    assert out["aim"]["source"] == "table"
    assert decode(out["mosaic_jpeg"]).shape == (256, 512, 3)
    assert out["config"] == {"mm_home": "refuse", "truth": None}
    assert out["env_steps"] == 0 and out["episode_over"] is False
    assert a.mm_observe(images=False)["obs_id"] == out["obs_id"] + 1


def test_a_translation_moves_along_the_base_axis_at_the_commanded_speed():
    a = Arm()
    w0 = a.pos.copy()
    out = translate(a, [0, 1, 0], 50, speed=0.025)  # base +Y = MotorMind left
    assert out["outcome"] == "done" and out["abort_reason"] == ""
    assert out["along_axis_m"] == pytest.approx(0.05, abs=0.003)
    assert out["axis"] == [0, 1, 0]
    assert np.allclose(a.pos - w0, BASE_R @ [0, 0.05, 0], atol=0.003)
    # 25 mm/s is 1.25 mm a tick (38 of them reach the 3 mm arrival band), two env steps a
    # tick, one video frame a tick.
    assert out["ticks"] == out["servo_ticks"] == 38
    assert out["env_steps"] == 76 == out["observation"]["env_steps"]
    assert len(out["frames"]) == 38
    assert out["observation"]["obs_id"] == out["obs_id"]


def test_a_descent_onto_the_table_is_blocked_and_a_push_reports_contact():
    a = Arm()
    assert translate(a, [0, 0, -1], 50)["outcome"] == "done"
    out = translate(a, [0, 0, -1], 300)
    assert (out["outcome"], out["abort_reason"]) == ("limited", "blocked")
    assert out["hold_ticks"] == 2 and out["ticks"] == out["servo_ticks"] + 2
    translate(a, [0, 0, 1], 10)
    pushed = translate(a, [0, 0, -1], 40, push=True)
    assert pushed["outcome"] == "contact" and "axis" not in pushed


def test_a_translation_pushed_sideways_skids():
    a = Arm()
    a.drift = 1.5
    out = translate(a, [1, 0, 0], 50)
    assert (out["outcome"], out["abort_reason"]) == ("limited", "skidded")


def test_a_rotation_turns_about_the_base_axis_and_a_fixed_hand_refuses_it():
    a = Arm()
    _p0, r0 = base_pose(a)
    seg = {
        "kind": "rotate",
        "goal_rotation": (
            rig.axis_angle_matrix([0, 0, math.radians(30)]) @ r0
        ).tolist(),
        "speed": math.radians(60),
    }
    out = a.mm_run_step(seg=seg)
    assert out["outcome"] == "done"
    assert out["turned_rad"] == pytest.approx(math.radians(30), abs=0.03)
    fixed = Arm(rotates=False)
    refused = fixed.mm_run_step(seg=seg)
    assert refused["outcome"] == "refused" and refused["ticks"] == 0
    assert "cannot rotate" in refused["message"]
    # Its translations ignore the orientation (which it does not command) and arrive.
    fixed.rot = rig.axis_angle_matrix([0, 0, 0.2]) @ fixed.rot
    p0, _r0 = base_pose(fixed)
    moved = fixed.mm_run_step(
        seg={
            "kind": "translate",
            "goal_position_m": (p0 + [0.03, 0, 0]).tolist(),
            "goal_rotation": DOWN.tolist(),
            "axis": [1, 0, 0],
            "speed": 0.05,
        }
    )
    assert moved["outcome"] == "done" and not moved["tick_budget_exhausted"]


def test_the_gripper_closes_judges_the_grasp_and_holds_a_partial_opening():
    a = Arm()
    empty = a.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert empty["ticks"] == 32 and empty["grasp"]["holding"] is False
    assert "nothing held" in empty["message"]
    b = Arm(obj=0.03)
    held = b.mm_run_step(seg={"kind": "gripper", "state": "close"})
    assert held["grasp"]["holding"] is True and held["gripper"]["commanded"] == "close"
    part = b.mm_run_step(seg={"kind": "gripper", "state": "open", "open_to_mm": 40})
    assert 40 <= part["detail"]["opening_mm"] <= 48
    assert part["gripper"]["commanded"] == "hold" and part["gripper"]["channel"] == 0.0
    before = part["detail"]["opening_mm"]
    up = translate(b, [0, 0, 1], 30)
    assert up["gripper"]["opening_mm"] == pytest.approx(before)
    nofingers = Arm(fingers=False).mm_run_step(
        seg={"kind": "gripper", "state": "close"}
    )
    assert nofingers["outcome"] == "refused" and nofingers["ticks"] == 0


def test_wait_ticks_are_seconds_over_the_tick_period():
    a = Arm()
    assert a.mm_run_step(seg={"kind": "wait", "seconds": 0.5})["ticks"] == 10
    assert a.mm_run_step(seg={"kind": "wait", "seconds": 0.02})["ticks"] == 1


def test_home_is_refused_by_default_and_is_the_robots_reset_as_a_variant():
    a = Arm()
    translate(a, [1, 0, 0], 20)
    out = a.mm_run_step(seg={"kind": "home"})
    assert out["outcome"] == "refused" and a.resets == 1
    assert out["observation"]["home"] == {"asked": 1, "refused": 1, "reset": 0}
    b = Arm(mm_home="reset")
    translate(b, [1, 0, 0], 20)
    b.mm_run_step(seg={"kind": "gripper", "state": "close"})
    out = b.mm_run_step(seg={"kind": "home"})
    assert out["outcome"] == "done" and out["episode_reset"] and out["episode_id"] == 2
    assert out["ticks"] == 0 and out["env_steps"] == 0
    assert (
        out["gripper"]["commanded"] == "open"
        and out["observation"]["caps"]["supports_home"]
    )
    assert np.allclose(out["observation"]["pose"]["position_m"], [0.45, 0.0, 0.3])
    with pytest.raises(ValueError, match="mm-home"):
        Arm(mm_home="arm")


def test_success_latches_through_a_segment_and_then_motion_is_refused():
    a = Arm(success_at=5)
    out = translate(a, [1, 0, 0], 50)
    assert out["outcome"] == "done" and out["terminated"] and out["episode_over"]
    again = translate(a, [1, 0, 0], 10)
    assert again["outcome"] == "refused" and again["ticks"] == 0
    b = Arm(max_steps=24)
    out = translate(b, [1, 0, 0], 100)
    assert out["truncated"] and out["ticks"] == 12


def test_a_stop_ends_the_segment_between_ticks():
    a = Arm()
    a.stop = True
    out = translate(a, [1, 0, 0], 50)
    assert (
        out["cancelled"] and out["ticks"] == 0 and out["message"].startswith("stopped")
    )


def test_a_lagging_hand_is_steered_by_its_target_then_settles_under_the_judge():
    a = Arm(lag=0.5)
    out = translate(a, [1, 0, 0], 40)
    assert out["outcome"] == "done", out["message"]
    assert out["settle_ticks"] > 0
    assert out["along_axis_m"] == pytest.approx(0.04, abs=rig.ARRIVE_M)
    # Blocked by the table, it stops pressing: the target comes back to the hand.
    out = translate(a, [0, 0, -1], 400)
    assert out["abort_reason"] == "blocked"
    assert np.allclose(a.target, a.pos, atol=0.01)


def test_bad_segments_are_errors():
    a = Arm()
    with pytest.raises(ValueError, match="kind"):
        a.mm_run_step(seg={"kind": "jump"})
    with pytest.raises(ValueError, match="zero"):
        a.mm_run_step(
            seg={
                "kind": "translate",
                "goal_position_m": [0, 0, 0],
                "axis": [0, 0, 0],
                "speed": 0.05,
            }
        )
    with pytest.raises(ValueError, match="speed"):
        translate(a, [1, 0, 0], 10, speed=0.0)
    with pytest.raises(ValueError, match="open_to_mm"):
        a.mm_run_step(seg={"kind": "gripper", "state": "close", "open_to_mm": 30})
    with pytest.raises(ValueError, match="observe"):
        a.mm_run_step(seg={"kind": "wait", "seconds": 0.1}, observe="images")


def test_truth_lines_are_keyed_by_obs_id(tmp_path):
    path = tmp_path / "mm_truth.jsonl"
    a = Arm(mm_truth=str(path))
    o = a.mm_observe(images=False)
    s = translate(a, [1, 0, 0], 10)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x["obs_id"] for x in lines] == [o["obs_id"], s["obs_id"]]
    assert lines[1]["kind"] == "segment" and lines[1]["segment"]["outcome"] == "done"
    assert "success_now" in lines[0] and "success_now" not in o
    assert o["config"]["truth"] == str(path)
