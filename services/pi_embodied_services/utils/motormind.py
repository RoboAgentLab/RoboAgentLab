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

"""MotorMind mode's env-server side for the robots whose simulator lives in the facade's own
process (Metaworld, robosuite, RoboCasa, RoboTwin): ``env.mm_observe`` and ``env.mm_run_step``
over a few robot hooks, with the semantics of LIBERO's (robots/libero/env_server.py, its MotorMind
section): the same base-frame segments, servo step, per-tick judge (blocked / skidded / creep),
segment results, grasp judgement, wrist clearance, aim marker, table height and three-view mosaic,
all from robots/libero/motormind_rig.py. LIBERO keeps its own implementation: its simulator runs
in a worker process reached through ``env_call``.

A MotorMind tick is rig.CONTROL_DT (0.05 s) of simulation, as on LIBERO: a robot whose control
period is shorter runs several env steps per tick (``_mm_tick``), so speeds, tick budgets and the
stall and creep counts mean the same time everywhere.

What differs from LIBERO, by design:

- No hand-back of a lost reply's steps (LIBERO's ``delivered`` / ``call_id``): every observation
  carries the episode's absolute ``env_steps`` and the latched ``terminated``, so pi's step count
  and success latch catch up at the next mm call after a timed-out segment; only that segment's
  video frames are lost.
- A motion's video frames ride in the reply (``frames``), as the robot's own motions send them.
- A rotation the robot cannot make (``max_rotation_rad`` 0: Metaworld's position-only hand) is
  refused, nothing moved; a robot without fingers refuses gripper segments the same way.
- ``home`` with ``--mm-home reset`` runs the robot's own episode reset (``_mm_reset_scene``). The
  settle after a reset (LIBERO's ``--settle-ticks``) is a lagging hand's coming to rest at the
  episode's first mm call (``settle_ticks``), not a fixed count; no side camera is remounted: the
  robot's ``_mm_cameras`` say which views it has.

The hooks a facade defines (base frame and poses in the world frame, metres, radians):

- ``_mm_static()``: ``{"body", "pos", "xmat"}`` of the base (read once per episode), the per-tick
  limits ``max_translation_m`` / ``max_rotation_rad`` (0: cannot rotate), ``fingers`` (bool),
  ``caps`` (the robot's own capability fields, see :func:`caps`);
- ``_mm_tool()``: the tool point's world position and rotation (approach axis = +Z column);
- ``_mm_opening()``: the finger gap, m;
- ``_mm_tick(dp_world, dr_world, grip, until)``: one tick moving the tool by these increments with
  gripper channel ``grip`` (+1 close, -1 open, 0 hold the fingers where they are); a robot with
  several env steps a tick ends it early once ``until()`` (None: never) holds;
- ``_mm_status()``: ``(terminated, truncated, env_steps)`` of the episode (terminated = success
  latched);
- ``_mm_shots(cameras, size, depth)``: per name of ``cameras`` (name -> robot camera) an upright
  ``rgb`` uint8, ``depth`` (camera z, m) for the names in ``depth``, ``K`` and ``cam2world``
  (4x4, OpenCV axes);
- ``_mm_cameras()``: MotorMind's view names (``scene``, ``side``, ``wrist``) -> robot cameras;
- ``_mm_reset_scene()``: the robot's episode reset (``home`` with ``--mm-home reset``);
- optional: ``_mm_video_frame()`` (one frame for the episode video, or None; called per tick);
  ``_mm_target()`` / ``_mm_set_target(p_world)`` for a hand that follows a commanded target with
  lag (Metaworld's mocap weld): the servo steers the target plus the hand's offset from it at the
  segment's start, settles under the judge once it arrives, and after an abort puts the target
  back at the hand, as an OSC hold does; ``_mm_opening_ahead()`` (where fingers that coast come
  to rest: an open-to stops on it); ``_mm_truth_extra()`` (privileged truth).

The facade calls ``_mm_setup(mm_home, mm_truth)`` in its constructor, ``_mm_new_episode()`` after
every reset, and registers ``_mm_rpc()``.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from pi_embodied_services.robots.libero import motormind_rig as rig
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("motormind")

#: --mm-home: refuse the model's ``home`` (default), or run it as the robot's episode reset.
MM_HOME = ("refuse", "reset")
#: The env.mm_run_step segment kinds.
MM_KINDS = ("translate", "rotate", "gripper", "wait", "home")
#: A lagging hand's settle after its target arrived: at most this many ticks, until the tool moves
#: less than SETTLED_M (m) in one.
SETTLE_TICKS = 12
SETTLED_M = 0.0005
#: Before a lagging hand's segment: at most REST_TICKS ticks until it moved less than REST_M (m) in
#: each of REST_QUIET ticks in a row (Metaworld's hand rings for about ten ticks after a reset; at
#: each turn of the ringing one tick is still).
REST_TICKS = 40
REST_M = 0.0001
REST_QUIET = 4


def _finite(name: str, value) -> np.ndarray:
    """A wire value as a finite float array; ``name`` names it in the error."""
    a = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite numbers, got {value!r}")
    return a


def caps(
    *,
    max_translation_m: float,
    max_rotation_rad: float,
    supports_home: bool,
    table_z_m: float | None,
    gripper_open_span_m: float,
    gripper_pad_length_m: float,
    reach_m: float,
    comfortable_reach_m: float,
    tilt_limits_deg: dict,
    gripper_gap_measured: bool = True,
    notes: tuple[str, ...] = (),
) -> dict:
    """A robot's capability table (MotorMind's ``Capabilities``): per-tick limits over the tick
    period, the robot's measured gripper and reach, LIBERO's defaults for the rest (the original
    rig's speeds and timing, rig.libero_caps), ``supports_home`` from ``--mm-home``."""
    dt = rig.CONTROL_DT
    rotates = max_rotation_rad > 0
    return {
        "max_translation_m": float(max_translation_m),
        # A robot that cannot rotate keeps a nominal limit: pi cuts rotations by it, the server
        # refuses them.
        "max_rotation_rad": float(max_rotation_rad if rotates else 0.5),
        "max_speed_m_s": float(max_translation_m) / dt,
        "max_rotation_speed_rad_s": float(max_rotation_rad if rotates else 0.5) / dt,
        "default_speed_m_s": 0.05,
        "default_rotation_speed_rad_s": math.radians(60.0),
        "control_point_offset_m": 0.0,
        "supports_contact_stop": True,
        "contact_stop_trusted": True,
        "supports_home": bool(supports_home),
        "gripper_gap_measured": bool(gripper_gap_measured),
        "gripper_open_span_m": float(gripper_open_span_m),
        "gripper_pad_length_m": float(gripper_pad_length_m),
        "table_z_m": table_z_m,
        "comfortable_reach_m": float(comfortable_reach_m),
        "reach_m": float(reach_m),
        "tilt_limits_deg": dict(tilt_limits_deg),
        "tilted_floor_m": 0.100,
        "tilt_reach_deg": 80.0 if rotates else 0.0,
        "max_object_height_m": 0.16,
        "subgoal_median_s": 21.0,
        "subgoal_verdict_s": 4.6,
        "arrival_tolerance_m": rig.ARRIVE_M,
        "mission_budget_s": None,
        "floor_above_table_m": None,
        "depth_min_range_m": rig.DEPTH_MIN_RANGE_M,
        "control_dt_s": dt,
        "notes": [
            "A motion that stops making progress ends on its own and reports how far it went.",
            "The episode ends at success or at its step horizon; after that no action is accepted.",
            *notes,
            "home resets the whole scene (the episode reset)."
            if supports_home
            else "There is no home: a new episode starts only from a fresh scene reset by the harness.",
        ],
    }


class MotorMindMixin:
    """``env.mm_observe`` / ``env.mm_run_step`` over the robot hooks of the module docstring."""

    def _mm_setup(self, mm_home: str = "refuse", mm_truth: str | None = None) -> None:
        """The server's --mm-home and --mm-truth; the mode's state for a first episode."""
        if mm_home not in MM_HOME:
            raise ValueError(f"--mm-home must be one of {MM_HOME}, not {mm_home!r}")
        self._mm_home = mm_home
        self._mm_truth = mm_truth
        self._mm_clock = 0.0
        self._mm_episode = 0
        self._mm_obs_id = 0
        self._mm_homes = {"asked": 0, "refused": 0, "reset": 0}
        self._mm_episode_state()

    def _mm_rpc(self) -> dict:
        """The RPC methods to register."""
        return {"env.mm_observe": self.mm_observe, "env.mm_run_step": self.mm_run_step}

    def _mm_new_episode(self) -> None:
        """After every reset: a new episode id and the frame clock one tick on (it never goes
        back), then a fresh episode state."""
        self._mm_episode += 1
        self._mm_clock += rig.CONTROL_DT
        self._mm_episode_state()

    def _mm_episode_state(self) -> None:
        """Ticks back to 0, the gripper channel open, the base and table read again at the next mm
        call."""
        self._mm_ticks = 0
        self._mm_settled = 0
        self._mm_grip = -1.0
        self._mm_opening_prev: float | None = None
        self._mm_opening_step = 0.0
        self._mm_rig: dict | None = None
        self._mm_table: dict | None = None

    # ---- hooks with defaults ----

    def _mm_video_frame(self):
        return None

    def _mm_opening_ahead(self) -> float:
        """The finger gap the fingers come to rest at (fingers that do not coast: the gap now)."""
        return self._mm_opening()

    def _mm_target(self) -> np.ndarray | None:
        """The commanded target a lagging hand follows (world), or None: the hand is servoed."""
        return None

    def _mm_set_target(self, p_world: np.ndarray) -> None:
        return None

    def _mm_truth_extra(self) -> dict:
        return {}

    # ---- the rig ----

    def _mm_commission(self) -> dict:
        """Read once per episode: the base frame, the limits and the robot's caps fields; a lagging
        hand comes to rest (MotorMind's settle after a reset, LIBERO's --settle-ticks: the first
        observation must not show it moving); then the table height from the scene camera's depth."""
        if self._mm_rig is None:
            st = self._mm_static()
            if st.get("body") is None:
                raise RuntimeError(
                    "MotorMind needs the robot base pose: the robot has none"
                )
            self._mm_rig = {
                "frame": rig.BaseFrame(
                    np.asarray(st["pos"], dtype=np.float64).reshape(3),
                    np.asarray(st["xmat"], dtype=np.float64).reshape(3, 3),
                ),
                "body": st["body"],
                "max_t": float(st["max_translation_m"]),
                "max_r": float(st["max_rotation_rad"]),
                "fingers": bool(st.get("fingers", True)),
                "caps": dict(st.get("caps", {})),
            }
            if self._mm_target() is not None:
                self._mm_settled = self._mm_come_to_rest([])
            self._mm_measure_table()
        return self._mm_rig

    def _mm_measure_table(self) -> dict:
        """The table height (base frame) from every second pixel of the scene depth."""
        frame = self._mm_rig["frame"]
        px = rig.CAMERA_PX
        cam = self._mm_cameras()["scene"]
        shot = self._mm_shots({"scene": cam}, px, ["scene"])["scene"]
        c2b = frame.transform() @ np.asarray(shot["cam2world"], dtype=np.float64)
        pts, _ = rig.back_project(
            np.asarray(shot["depth"]), shot["K"], c2b, stride=rig.TABLE_STRIDE
        )
        self._mm_table = {"z_m": rig.table_height(pts[:, 2]), "n_points": len(pts)}
        return self._mm_table

    def _mm_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The tool pose in the base frame."""
        frame = self._mm_commission()["frame"]
        p, r = self._mm_tool()
        return frame.point(p), frame.rotation(r)

    def _mm_live(self) -> bool:
        terminated, truncated, _ = self._mm_status()
        return not (terminated or truncated)

    def _mm_running(self) -> bool:
        """Whether a segment in progress may take another tick: until the step horizon (a success
        latch does not cut it short) or a stop."""
        return not self._mm_status()[1] and not self.stop_requested()

    def _mm_step(self, dp_base, dr_base, frames: list, until=None) -> None:
        """One tick (base-frame increments) with the held gripper channel; the frame clock, the
        finger gap's change per tick and the video."""
        frame = self._mm_rig["frame"]
        self._mm_tick(
            frame.vec_to_world(dp_base),
            frame.vec_to_world(dr_base),
            self._mm_grip,
            until,
        )
        self._mm_clock += rig.CONTROL_DT
        self._mm_ticks += 1
        gap = self._mm_opening()
        if self._mm_opening_prev is not None:
            self._mm_opening_step = abs(gap - self._mm_opening_prev)
        self._mm_opening_prev = gap
        shot = self._mm_video_frame()
        if shot is not None:
            frames.append(shot)

    def _mm_hold(self, n: int, frames: list) -> int:
        """``n`` zero ticks with the held gripper channel (fewer at the horizon or a stop)."""
        k = 0
        while k < int(n) and self._mm_running():
            self._mm_step(np.zeros(3), np.zeros(3), frames)
            k += 1
        return k

    def _mm_grasp(self) -> dict:
        moving = self._mm_opening_step > rig.MOVING_PER_TICK_M
        span = self._mm_rig["caps"].get("gripper_open_span_m", rig.GRIPPER_OPEN_SPAN_M)
        return rig.grasp_state(
            rig.channel_name(self._mm_grip), self._mm_opening(), moving, span
        )

    def _mm_static_fields(self) -> dict:
        """The fields every observation repeats: base pose, caps, table, degraded flags, the home
        mode and its counts, the configuration pi checks when it attaches."""
        r = self._mm_rig
        table = (self._mm_table or {}).get("z_m")
        return {
            "base_pose": {
                "body": r["body"],
                **rig.pose_dict(r["frame"].p, r["frame"].R),
            },
            "caps": caps(
                max_translation_m=r["max_t"],
                max_rotation_rad=r["max_r"],
                supports_home=self._mm_home == "reset",
                table_z_m=table,
                **r["caps"],
            ),
            "control_dt_s": rig.CONTROL_DT,
            "table_z_m": table,
            "degraded": {
                "table_z": table is None,
                # No robot here mounts MotorMind's side camera (LIBERO's --side-camera).
                "side_mount": False,
                "side_mount_reason": None,
            },
            "camera_names": list(self._mm_cameras()),
            "home_mode": self._mm_home,
            "home": dict(self._mm_homes),
            "config": {"mm_home": self._mm_home, "truth": self._mm_truth or None},
        }

    # ---- images ----

    def _mm_aim(self, p, r, aim, clearance_mm, tracked) -> dict:
        if aim is not None:
            return {"point": _finite("aim_base_m", aim).reshape(3), "source": "given"}
        t = None if tracked is None else float(_finite("aim_tracked_m", tracked))
        point, depth, source = rig.aim_point(
            p, r, (self._mm_table or {}).get("z_m"), clearance_mm, t
        )
        return {"point": point, "depth_m": depth, "source": source}

    def _mm_images(self, p, r, aim, cameras, tracked=None) -> dict:
        views = self._mm_cameras()
        names = list(views)
        if cameras:
            unknown = [c for c in cameras if c not in names]
            if unknown:
                raise ValueError(f"cameras must be among {names}, got {unknown}")
            names = [n for n in names if n in cameras]
        px = rig.CAMERA_PX
        shots = self._mm_shots(
            {n: views[n] for n in names}, px, ["wrist"] if "wrist" in names else []
        )
        to_base = self._mm_rig["frame"].transform()
        out: dict[str, Any] = {"cameras": {}}
        if "wrist" in names:
            # The clearance first: with no table it sets the aim marker's depth.
            s = shots["wrist"]
            c2b = to_base @ np.asarray(s["cam2world"], dtype=np.float64)
            depth = rig.quantize_depth(np.asarray(s["depth"]))
            cam_pts, _z = rig.back_project(depth, s["K"], np.eye(4))
            out["clearance"] = rig.clearance(
                cam_pts @ c2b[:3, :3].T + c2b[:3, 3], cam_pts, p, r
            )
        aim_at = self._mm_aim(
            p, r, aim, (out.get("clearance") or {}).get("mm"), tracked
        )
        marked = []
        for n in names:
            s = shots[n]
            k = np.asarray(s["K"], dtype=np.float64)
            c2b = to_base @ np.asarray(s["cam2world"], dtype=np.float64)
            uv = rig.project(k, c2b, aim_at["point"])
            imgs = rig.camera_images(s["rgb"], None if uv is None else uv[:2])
            marked.append(imgs.pop("_marked"))
            cam = {
                "camera": views[n],
                "width": px,
                "height": px,
                "intrinsic": k.tolist(),
                "cam2base": c2b.tolist(),
                "timestamp": self._mm_clock,
                "aim_px": None if uv is None else [uv[0], uv[1]],
                **imgs,
            }
            if n == "wrist":
                cam["depth_min_range_m"] = rig.DEPTH_MIN_RANGE_M
            out["cameras"][n] = cam
        out["aim"] = {
            "source": aim_at["source"],
            "base_m": [float(v) for v in aim_at["point"]],
            **({"depth_m": aim_at["depth_m"]} if "depth_m" in aim_at else {}),
        }
        out["mosaic_jpeg"] = rig.mosaic(marked)
        return out

    # ---- observation ----

    def _mm_observation(
        self, images: bool, aim=None, cameras=None, segment=None, tracked=None
    ) -> dict:
        """One MotorMind observation (a new ``obs_id``); a line of privileged truth goes to
        --mm-truth under the same id."""
        self._mm_obs_id += 1
        p, r = self._mm_pose()
        opening = self._mm_opening()
        terminated, truncated, steps = self._mm_status()
        out: dict[str, Any] = {
            "obs_id": self._mm_obs_id,
            "frame_clock_s": self._mm_clock,
            "episode_id": self._mm_episode,
            "ticks": self._mm_ticks,
            "settle_ticks": self._mm_settled,
            "env_steps": int(steps),
            "pose": rig.pose_dict(p, r),
            "gripper": {
                "opening_mm": opening * 1000.0,
                "commanded": rig.channel_name(self._mm_grip),
                "channel": float(self._mm_grip),
                "moving": self._mm_opening_step > rig.MOVING_PER_TICK_M,
            },
            "grasp": self._mm_grasp(),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_over": bool(terminated or truncated),
            **self._mm_static_fields(),
        }
        if images:
            out.update(self._mm_images(p, r, aim, cameras, tracked))
        self._mm_write_truth(out, segment)
        return out

    def _mm_write_truth(self, obs: dict, segment: dict | None) -> None:
        """Append one line of privileged truth (--mm-truth), keyed by ``obs_id``: the tool's world
        pose, the latch and the robot's ``_mm_truth_extra`` (``success_now``, objects). It never
        enters an observation; a failure is logged and the call goes on."""
        if not self._mm_truth:
            return
        try:
            p, r = self._mm_tool()
            line = {
                "obs_id": obs["obs_id"],
                "kind": "segment" if segment else "observe",
                "episode_id": obs["episode_id"],
                "frame_clock_s": obs["frame_clock_s"],
                "ticks": obs["ticks"],
                "wall_time": time.time(),
                "tool_pos_world": [float(v) for v in np.asarray(p).reshape(3)],
                "tool_rotation_world": np.asarray(r, dtype=np.float64).tolist(),
                "success_latched": obs["terminated"],
                **({"segment": segment} if segment else {}),
                **self._mm_truth_extra(),
            }
            with open(self._mm_truth, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=float) + "\n")
        except Exception as e:  # noqa: BLE001  the truth log never breaks the episode
            logger.warning("mm_truth.jsonl: %s", e)

    def mm_observe(
        self, images: bool = True, aim_base_m=None, cameras=None, aim_tracked_m=None
    ) -> dict:
        """MotorMind's observation, base frame: ``obs_id``, ``frame_clock_s``, ``episode_id``,
        ``ticks``, ``env_steps``, ``pose``, ``gripper``, ``grasp``, ``terminated`` /
        ``truncated`` / ``episode_over`` and the static fields (``base_pose``, ``caps``,
        ``table_z_m``, ``degraded``, ``config``, ...). ``images``: each view's 768 render as
        ``marked768`` / ``marked512`` / ``raw512`` JPEGs with ``intrinsic`` and ``cam2base``, the
        wrist ``clearance``, the aim marker (``aim_base_m``, else rig.aim_point with the caller's
        tracked clearance ``aim_tracked_m``) and the ``mosaic_jpeg``; ``cameras`` limits the
        renders to some of the views."""
        self._mm_commission()
        return self._mm_observation(
            bool(images), aim_base_m, cameras, tracked=aim_tracked_m
        )

    # ---- one segment ----

    def mm_run_step(
        self, seg: dict, observe: str = "state", aim_base_m=None, aim_tracked_m=None
    ) -> dict:
        """Run one segment of a MotorMind action (base frame, metres, radians); the segment kinds
        and their fields are LIBERO's (``env.mm_run_step`` there): ``translate`` (goal_position_m,
        axis, speed, optional goal_rotation, distance_m, push), ``rotate`` (goal_rotation, speed,
        optional goal_position_m), ``gripper`` (state, optional open_to_mm), ``wait`` (seconds),
        ``home`` (refused, or the robot's episode reset with --mm-home reset).

        Returns the measurement (``outcome`` done / contact / limited / refused,
        ``abort_reason``, ``moved_m``, ``along_axis_m``, ``lateral_m``, ``turned_rad``, ``ticks``,
        ``pose_before`` / ``pose_after``, ...), ``frames`` (the video frames the segment produced)
        and ``observation`` (``observe``: ``state`` without images, ``full`` as env.mm_observe)."""
        self._mm_commission()
        if observe not in ("state", "full"):
            raise ValueError("observe must be 'state' or 'full'")
        kind = seg.get("kind") if isinstance(seg, dict) else None
        if kind not in MM_KINDS:
            raise ValueError(f"seg.kind must be one of {MM_KINDS}, got {kind!r}")
        t0 = time.monotonic()
        start_ticks = self._mm_ticks
        episode = self._mm_episode
        p0, r0 = self._mm_pose()
        frames: list = []
        if kind == "home":
            out = self._mm_home_step()
        elif not self._mm_live():
            out = {
                "outcome": "refused",
                "abort_reason": "",
                "message": "the episode is over (success latched or the step limit reached): nothing moved",
            }
        elif kind in ("translate", "rotate"):
            out = self._mm_motion(seg, p0, r0, frames)
        elif kind == "gripper":
            out = self._mm_gripper(seg, frames)
        else:
            out = self._mm_wait(seg, frames)
        same = self._mm_episode == episode
        p1, r1 = self._mm_pose()
        if "moved_m" not in out:
            d = float(np.linalg.norm(p1 - p0)) if same else 0.0
            turned = rig.angle_between(r0, r1) if same else 0.0
            out.update(moved_m=d, along_axis_m=d, lateral_m=0.0, turned_rad=turned)
        ticks = self._mm_ticks - start_ticks if same else 0
        out.update(
            kind=kind,
            ticks=ticks,
            elapsed_s=time.monotonic() - t0,
            frame_clock_s=self._mm_clock,
            pose_before=rig.pose_dict(p0, r0),
            pose_after=rig.pose_dict(p1, r1),
            episode_id=self._mm_episode,
            frames=frames,
        )
        observation = self._mm_observation(
            observe == "full",
            aim_base_m,
            segment={
                "kind": kind,
                "outcome": out["outcome"],
                "abort_reason": out.get("abort_reason", ""),
                "ticks": ticks,
            },
            tracked=aim_tracked_m,
        )
        out.update(
            obs_id=observation["obs_id"],
            grasp=observation["grasp"],
            gripper=observation["gripper"],
            episode_over=observation["episode_over"],
            terminated=observation["terminated"],
            truncated=observation["truncated"],
            env_steps=observation["env_steps"],
            observation=observation,
        )
        return out

    @staticmethod
    def _mm_refused(message: str) -> dict:
        return {"outcome": "refused", "abort_reason": "", "message": message}

    def _mm_motion(
        self, seg: dict, p0: np.ndarray, r0: np.ndarray, frames: list
    ) -> dict:
        r = self._mm_rig
        kind = seg["kind"]
        speed = float(_finite("speed", seg.get("speed", float("nan"))))
        if speed <= 0:
            raise ValueError("speed must be > 0")
        given_r = seg.get("goal_rotation")
        if kind == "rotate" and given_r is None:
            raise ValueError("a rotate segment needs goal_rotation")
        goal_r = (
            None if given_r is None else _finite("goal_rotation", given_r).reshape(3, 3)
        )
        axis, goal_p, given_d = None, None, seg.get("distance_m")
        if kind == "translate":
            goal_p = _finite("goal_position_m", seg.get("goal_position_m")).reshape(3)
            a = _finite("axis", seg.get("axis")).reshape(3)
            if np.linalg.norm(a) < 1e-9:
                raise ValueError("axis must not be the zero vector")
            axis = a / np.linalg.norm(a)
            if given_d is not None:
                given_d = float(_finite("distance_m", given_d))
        elif seg.get("goal_position_m") is not None:
            goal_p = _finite("goal_position_m", seg["goal_position_m"]).reshape(3)
        if kind == "rotate" and r["max_r"] <= 0:
            return self._mm_refused(
                "this robot's hand cannot rotate (its orientation is fixed): nothing moved"
            )
        frame = r["frame"]
        # A lagging hand: steer its target, offset by where the hand rests relative to it, once it
        # has come to rest (after a reset it is still finding its place on the weld).
        offset, rest = None, 0
        if self._mm_target() is not None:
            rest = self._mm_come_to_rest(frames)
            p0, r0 = self._mm_pose()
            offset = self._mm_tool()[0] - self._mm_target()
        if goal_p is None:
            goal_p = p0
        # A hand that cannot rotate keeps whatever orientation it has.
        if goal_r is None or r["max_r"] <= 0:
            goal_r = r0
        distance = 0.0
        if kind == "translate":
            distance = (
                float(np.linalg.norm(goal_p - p0)) if given_d is None else given_d
            )
        push = bool(seg.get("push", False))
        segment = rig.Segment(
            kind,
            goal_p,
            goal_r,
            speed,
            axis,
            distance,
            push,
            holding=self._mm_grasp()["holding"] is True,
        )
        judge = rig.SegmentJudge(segment, p0, r0)
        ticks, settle, cancelled, budget = 0, 0, False, False
        p, rot, step = p0, r0, math.inf
        while not self._mm_status()[1]:
            if self.stop_requested():
                cancelled = True
                break
            q = p if offset is None else frame.point(self._mm_target() + offset)
            # A hand that cannot rotate is steered by its position alone.
            now_r = rot if r["max_r"] > 0 else goal_r
            dp, dr, arrived = rig.servo_step(
                q, now_r, segment, r["max_t"], max(r["max_r"], 0.0), rig.CONTROL_DT
            )
            if arrived:
                # A lagging hand settles under the judge once its target has arrived.
                if offset is None or settle >= SETTLE_TICKS or step < SETTLED_M:
                    break
                settle += 1
            elif ticks >= rig.SEGMENT_MAX_TICKS:
                budget = True
                break
            self._mm_step(dp, dr, frames)
            ticks += 1
            prev, (p, rot) = p, self._mm_pose()
            step = float(np.linalg.norm(p - prev))
            if judge.tick(p, rot):
                break
        hold = 0
        if judge.abort and not cancelled:
            if offset is not None:
                # Stop pressing: the target back at the hand.
                self._mm_set_target(self._mm_tool()[0] - offset)
            hold = self._mm_hold(rig.HOLD_TICKS, frames)
        p1, r1 = self._mm_pose()
        out = rig.segment_result(segment, p0, r0, p1, r1, judge.abort)
        out.update(
            rest_ticks=rest,
            servo_ticks=ticks,
            settle_ticks=settle,
            hold_ticks=hold,
            lateral_limit_m=judge.limit,
            crept=judge.crept,
            tick_budget_exhausted=budget,
            cancelled=cancelled,
            speed=speed,
            push=push,
        )
        if cancelled:
            out["message"] = "stopped: " + out["message"]
        if kind == "translate" and not push:
            out["axis"] = [float(v) for v in axis]
        return out

    def _mm_come_to_rest(self, frames: list) -> int:
        """Zero ticks until the tool moved less than REST_M in each of REST_QUIET ticks in a row
        (at most REST_TICKS)."""
        n, quiet, last = 0, 0, self._mm_tool()[0]
        while n < REST_TICKS and quiet < REST_QUIET and self._mm_running():
            self._mm_step(np.zeros(3), np.zeros(3), frames)
            n += 1
            now = self._mm_tool()[0]
            quiet = quiet + 1 if float(np.linalg.norm(now - last)) < REST_M else 0
            last = now
        return n

    def _mm_gripper(self, seg: dict, frames: list) -> dict:
        state = seg.get("state")
        if state not in ("open", "close"):
            raise ValueError("a gripper segment's state must be 'open' or 'close'")
        asked = seg.get("open_to_mm")
        if asked is not None and state != "open":
            raise ValueError("open_to_mm goes with state 'open'")
        if not self._mm_rig["fingers"]:
            return self._mm_refused("this robot's tool has no fingers: nothing moved")
        detail: dict[str, Any] = {}
        if asked is None:
            self._mm_grip = 1.0 if state == "close" else -1.0
            self._mm_hold(rig.GRIPPER_SETTLE_TICKS, frames)
        else:
            want = float(_finite("open_to_mm", asked)) / 1000.0
            self._mm_grip = -1.0
            n = 0
            while n < rig.OPEN_TO_MAX_TICKS and self._mm_running():
                if self._mm_opening_ahead() >= want:
                    break
                self._mm_step(
                    np.zeros(3),
                    np.zeros(3),
                    frames,
                    until=lambda: self._mm_opening_ahead() >= want,
                )
                n += 1
            # Hold: the fingers stay at this gap until the next gripper command.
            self._mm_grip = 0.0
            self._mm_hold(rig.GRIPPER_SETTLE_TICKS, frames)
            detail["asked_mm"] = want * 1000.0
            detail["opening_ticks"] = n
        grasp = self._mm_grasp()
        opening = self._mm_opening() * 1000.0
        detail["opening_mm"] = opening
        if asked is not None:
            msg = f"opened to {opening:.1f} mm (asked {detail['asked_mm']:.1f} mm), holding the gap"
        elif state == "close":
            word = {
                True: "holding something",
                False: "nothing held",
                None: "grasp unknown",
            }
            msg = f"{word[grasp['holding']]} -- {grasp['reason']}"
        else:
            msg = f"opened to {opening:.1f} mm"
        return {"outcome": "done", "abort_reason": "", "message": msg, "detail": detail}

    def _mm_wait(self, seg: dict, frames: list) -> dict:
        seconds = float(_finite("seconds", seg.get("seconds", float("nan"))))
        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        # The original's count: at least one tick, the seconds over the period truncated.
        n = self._mm_hold(max(1, int(seconds / rig.CONTROL_DT)), frames)
        return {"outcome": "done", "abort_reason": "", "message": f"waited {n} ticks"}

    def _mm_home_step(self) -> dict:
        """The model's ``home``: refused (--mm-home=refuse), or the robot's episode reset
        (``reset``), after which the gripper is commanded open."""
        self._mm_homes["asked"] += 1
        if self._mm_home != "reset":
            self._mm_homes["refused"] += 1
            return self._mm_refused(
                "home is not available (--mm-home=refuse): nothing moved"
            )
        self._mm_reset_scene()
        self._mm_homes["reset"] += 1
        self._mm_commission()
        return {
            "outcome": "done",
            "abort_reason": "",
            "message": "env.reset()",
            "episode_reset": True,
        }
