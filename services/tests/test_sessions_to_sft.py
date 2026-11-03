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

"""Teacher sessions -> SFT pairs (finetuned/sessions_to_sft.py): the node shim runs the TypeScript
converter (packages/embodied/src/modes/finetuned/sessions_to_sft.ts) on a tiny session and reports
its counts; the prompts themselves are checked by packages/embodied/test/planner-surface.test.ts."""

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pi_embodied_services.finetuned import sessions_to_sft as s2s

# A 1x1 black PNG (the converter stores the observation's images as recorded).
PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000009077053e"
        "0000000c4944415408d763f8cfc00000030101009c5a4bd80000000049454e44ae426082"
    )
).decode()


def session(claimed: str = "success") -> list[dict]:
    """A ManiSkill session: an observation, a move_delta, a program, finish; the program's result solves."""
    steps = [
        ("view_env_state", {}, {"state": {}}),
        (
            "move_delta",
            {"delta_xyz": [0.0, 0.0, -0.05]},
            {"state": {"tcp_pos": [0.4, 0.0, 0.25]}},
        ),
        (
            "run_code",
            {"code": "r = move_delta([0, 0, 0.05])\nRESULT = r\n"},
            {"status": "ran", "success": True, "terminated": True},
        ),
        ("finish", {"status": claimed, "summary": "done"}, {}),
    ]
    entries: list[dict] = [
        {
            "type": "custom",
            "customType": "robot_task",
            "data": {"robot": "maniskill", "task": "PickCube-v1", "seed": "0"},
        },
        {
            "type": "custom",
            "customType": "code_api",
            "data": {
                "tier": None,
                "available": ["move_delta", "state"],
                "manifest_digest": "",
                "digest": "",
            },
        },
    ]
    for i, (name, args, details) in enumerate(steps):
        entries.append(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": f"c{i}",
                            "name": name,
                            "arguments": args,
                        }
                    ],
                },
            }
        )
        content = [{"type": "text", "text": json.dumps(details)}]
        if name != "finish":
            content.append({"type": "image", "data": PNG, "mimeType": "image/png"})
        entries.append(
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": f"c{i}",
                    "toolName": name,
                    "content": content,
                    "details": details,
                    "isError": False,
                },
            }
        )
    entries.append(
        {
            "type": "custom",
            "customType": "robot_result",
            "data": {
                "robot": "maniskill",
                "claimed": claimed,
                "env_error": False,
                "planner_error": None,
                "code_api": "low",
            },
        }
    )
    return entries


def test_shim_points_at_the_converter():
    assert s2s.SCRIPT.name == "sessions_to_sft.ts"
    assert s2s.SCRIPT.is_file(), s2s.SCRIPT


@pytest.mark.parametrize("surface", ["tools", "code"])
def test_dry_run_counts_the_teacher_steps(tmp_path: Path, surface: str):
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")
    good = tmp_path / "good.jsonl"
    good.write_text("".join(json.dumps(e) + "\n" for e in session()), encoding="utf-8")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "".join(json.dumps(e) + "\n" for e in session("failure")), encoding="utf-8"
    )
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pi_embodied_services.finetuned.sessions_to_sft",
            "--surface",
            surface,
            "--dry-run",
            "--vars",
            "arms=;cameras=agentview,wrist;max_move=0.2;move_subject=the TCP;move_frame=[dx, dy, dz] m",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    report = json.loads(out.stdout)
    # tools: move_delta and finish after an observation (view_env_state has none before it; run_code is not a tool
    # of the manifest's tool list); code: the one program that ran. The failed run is skipped.
    assert report["out"] is None
    assert report["surface"] == surface
    assert report["samples"] == (2 if surface == "tools" else 1)
    by_name = {Path(s["session"]).name: s for s in report["sessions"]}
    assert by_name["bad.jsonl"]["skipped"] == "not a successful run"
    assert by_name["good.jsonl"]["samples"] == report["samples"]
    assert not (tmp_path / "sft.json").exists()
