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

"""The ManiSkill env server's VLA surface without a simulator: Bridge V2 actions translated into
the bridge twins' WidowX control mode (a recorded OpenVLA chunk), ``env.vla_step`` and
``env.vla_observation`` on a fake env, and the server's self-check with the two methods."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pi_embodied_services.components import manifest as M
from pi_embodied_services.robots.maniskill import env_server as ms

#: OpenVLA (openvla/openvla-7b@47a0ec7, --suite bridge_orig, greedy) on PutCarrotOnPlateInScene-v1
#: seed 0: its first five actions from the reset, one per call (the raw Bridge V2 output: [dx, dy,
#: dz] m, [droll, dpitch, dyaw] rad, the open probability), recorded on l40s-0003 through
#: env.vla_observation / env.vla_step with "put carrot on plate".
RECORDED = np.array(
    [
        [
            -0.00266920,
            -0.00074788,
            -0.00985478,
            -0.00205466,
            0.02735372,
            -0.00651390,
            0.99607843,
        ],
        [
            -0.00199818,
            -0.00171916,
            -0.00985478,
            -0.00716342,
            0.03405237,
            -0.00651390,
            0.99607843,
        ],
        [
            -0.00311655,
            0.00151844,
            -0.00803600,
            -0.00013888,
            0.02735372,
            0.00799228,
            0.99607843,
        ],
        [
            -0.00266920,
            0.00022340,
            -0.00881548,
            -0.00013888,
            0.02735372,
            0.00799228,
            0.99607843,
        ],
        [
            -0.00266920,
            0.00022340,
            -0.00881548,
            -0.00013888,
            0.02936331,
            -0.00651390,
            0.99607843,
        ],
    ],
    dtype=np.float32,
)
#: The same chunk through SimplerEnv's widowx_bridge conversion, computed with transforms3d 0.4.2
#: (``euler2axangle(roll, pitch, yaw)``, axes sxyz; ``2 * (open > 0.5) - 1``), not with the scipy
#: code under test. The last row is the native action the box's env server applied.
EXPECTED = np.array(
    [
        [
            -0.00266920,
            -0.00074788,
            -0.00985478,
            -0.00196544,
            0.02736031,
            -0.00648539,
            1.0,
        ],
        [
            -0.00199818,
            -0.00171916,
            -0.00985478,
            -0.00705179,
            0.03407543,
            -0.00639128,
            1.0,
        ],
        [
            -0.00311655,
            0.00151844,
            -0.00803600,
            -0.00024818,
            0.02735302,
            0.00799368,
            1.0,
        ],
        [
            -0.00266920,
            0.00022340,
            -0.00881548,
            -0.00024818,
            0.02735302,
            0.00799368,
            1.0,
        ],
        [
            -0.00266920,
            0.00022340,
            -0.00881548,
            -0.00004324,
            0.02936366,
            -0.00651140,
            1.0,
        ],
    ],
    dtype=np.float32,
)


def _rot(axis: int, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    i, j = [k for k in range(3) if k != axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    if axis == 1:  # about y the off-diagonal signs swap
        r[i, j], r[j, i] = s, -s
    return r


def _rotvec(r: np.ndarray) -> np.ndarray:
    """A rotation matrix's axis-angle vector (Rodrigues), for angles below pi."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return axis / (2 * math.sin(angle)) * angle


def test_the_recorded_openvla_chunk_becomes_the_widowx_action():
    out = ms.vla_actions("bridge_ee_delta_euler", "widowx250s", RECORDED)
    assert out.dtype == np.float32 and out.shape == (5, 7)
    np.testing.assert_allclose(out, EXPECTED, atol=2e-5)


def test_the_euler_delta_is_static_xyz_and_the_gripper_is_binarised():
    """sxyz: R = Rz(yaw) Ry(pitch) Rx(roll), rotated about the fixed axes in that order."""
    roll, pitch, yaw = 0.3, -0.2, 0.5
    r = _rot(2, yaw) @ _rot(1, pitch) @ _rot(0, roll)
    a = np.array([[0.01, -0.02, 0.03, roll, pitch, yaw, 0.51]])
    out = ms.bridge_to_widowx(a)[0]
    np.testing.assert_allclose(out[3:6], _rotvec(r), atol=1e-6)
    assert out[:3].tolist() == pytest.approx([0.01, -0.02, 0.03]), (
        "the translation stays in metres: the controller takes it unnormalised"
    )
    assert out[6] == 1.0
    # A single-axis delta is that axis's rotation vector; 0.5 is not open.
    np.testing.assert_allclose(
        ms.bridge_to_widowx([[0, 0, 0, 0, 0, 0.1, 0.5]])[0],
        [0, 0, 0, 0, 0, 0.1, -1.0],
        atol=1e-7,
    )


def test_an_action_space_drives_only_the_robots_it_was_made_for():
    with pytest.raises(
        ValueError, match="no translation for the VLA action space 'libero_osc'"
    ):
        ms.vla_actions("libero_osc", "widowx250s", RECORDED)
    with pytest.raises(ValueError, match="drive --robot widowx250s, not panda"):
        ms.vla_actions("bridge_ee_delta_euler", "panda", RECORDED)
    # The robot it drives runs the scene's own 7-D pose controller (3 translation, 3 rotation, gripper).
    spec = ms.ROBOTS["widowx250s"]
    assert (spec.rot_dims, spec.gripper) == (3, (1.0, -1.0))


class _Env:
    """A bridge-twin-like env: the TCP moves by each action's xyz; success once it is above z=0.92."""

    def __init__(self):
        self.tcp = np.array([0.0, 0.0, 0.9])
        self.actions: list[np.ndarray] = []
        env = self
        tcp = SimpleNamespace(
            pose=type("P", (), {"p": property(lambda _: env.tcp.reshape(1, 3))})()
        )
        self.unwrapped = SimpleNamespace(agent=SimpleNamespace(tcp=tcp))

    def step(self, a):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        self.actions.append(a)
        self.tcp = self.tcp + a[:3]
        info = {"success": np.array([self.tcp[2] > 0.92])}
        return {"step": 1}, np.array([0.0]), np.array([False]), np.array([False]), info


def facade() -> ms.ManiskillEnvFacade:
    f = object.__new__(ms.ManiskillEnvFacade)
    ms.BaseEnvFacade.__init__(f)
    f._env = _Env()
    f._robot = ms.ROBOTS["widowx250s"]
    f._rig = None
    f._cameras = {"agentview": "3rd_view_camera"}
    f._grip = [1.0]
    f._last_info = {}
    f._success_once = False
    f._meta = {
        "robot": "widowx250s",
        "wrist": False,
        "env_id": "PutCarrotOnPlateInScene-v1",
    }
    raw = (
        np.arange(480 * 640 * 3, dtype=np.uint32)
        .astype(np.uint8)
        .reshape(1, 480, 640, 3)
    )
    f._obs = {"sensor_data": {"3rd_view_camera": {"rgb": raw}}}
    f._rgb = lambda obs, name: np.zeros((4, 4, 3), np.uint8)
    state = lambda _agent=None: {  # noqa: E731
        "tcp_pos": f._env.tcp.astype(np.float32),
        "tcp_quat_wxyz": np.array([1, 0, 0, 0], np.float32),
        "gripper_width": 0.074,
        "qpos": np.zeros(8, np.float32),
    }
    f._state = state
    f._arm_state = state
    return f


def test_vla_step_runs_the_translated_chunk_until_success():
    f = facade()
    chunk = np.zeros((4, 7), np.float32)
    chunk[:, 2] = 0.01  # up 1 cm per action
    chunk[:, 6] = [0.9, 0.9, 0.1, 0.9]
    out = f._rpc["env.vla_step"](chunk, action_space="bridge_ee_delta_euler")
    # Success at z > 0.92: the third action, so the fourth never runs.
    assert out["steps"] == len(out["frames"]) == 3 == len(f._env.actions)
    assert [fr["success"] for fr in out["frames"]] == [False, False, True]
    np.testing.assert_allclose(out["frames"][0]["action"], [0, 0, 0.01, 0, 0, 0, 1])
    assert f._env.actions[2][6] == -1.0, "0.1 open probability: closed"
    assert out["gripper"] == -1.0, "the command the last step held"
    assert out["obs"] is out["frames"][-1]
    assert f._success_once
    again = f._rpc["env.vla_step"](chunk, action_space="bridge_ee_delta_euler")
    assert again["error"] == "the task is already solved; call finish"
    assert again["steps"] == 0 and len(f._env.actions) == 3
    with pytest.raises(ValueError, match="no translation"):
        f._rpc["env.vla_step"](chunk, action_space="libero_osc")


def test_vla_observation_hands_over_the_raw_bridge_frame_and_the_state():
    f = facade()
    out = f._rpc["env.vla_observation"](main="agentview")
    assert out["main"].shape == (480, 640, 3) and out["main"].dtype == np.uint8
    assert np.array_equal(
        out["main"], f._obs["sensor_data"]["3rd_view_camera"]["rgb"][0]
    )
    assert out["wrist"] is None
    assert out["state"].dtype == np.float32
    assert out["state"].tolist() == pytest.approx([0, 0, 0.9, 1, 0, 0, 0, 0.074])
    with pytest.raises(ValueError, match="no wrist camera"):
        f._rpc["env.vla_observation"](main="agentview", wrist="wrist")


def test_the_server_checks_itself_with_the_vla_methods_internal():
    f = facade()
    m = M.load_manifest("maniskill")
    assert {"env.vla_observation", "env.vla_step"} <= set(m["internal"])
    M.self_check(f, m, lambda _c: False)
    assert "env.vla_observation" in f._readonly_methods
