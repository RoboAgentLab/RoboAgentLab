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

"""Teacher sessions -> SFT pairs for a student speaking the tools or code surface.

    python -m pi_embodied_services.finetuned.sessions_to_sft --surface tools|code --out <dir> \\
        [--robot R] [--vars "arms=left,right;max_move=0.2"] [--tools a,b] [--code-api tier] \\
        [--task T] [--include-failures] [--dry-run] <session.jsonl | dir>...

Runs ``packages/embodied/src/modes/finetuned/sessions_to_sft.ts`` under node (as
``lerobot_to_rollouts.py`` runs ``transform.ts``): the prompt of every sample is rendered by the
same TypeScript the provider renders it with at inference (``surfaces.ts``: the tools list from
the robot's manifest, the code API from code mode's renderer), so a training prompt is byte for
byte the inference one. Writes ``<out>/sft.json`` (LLaMA-Factory's Alpaca layout, ``<image>``
markers plus ``images``, as Show-Harness's ``rollouts_to_alpaca.py`` writes the units samples) and
``<out>/images/<session>/<step>_<i>.png``. ``--dry-run`` prints the counts and writes nothing.
Needs node 22.19+ on PATH (``NODE`` overrides it).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PI_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PI_ROOT / "packages/embodied/src/modes/finetuned/sessions_to_sft.ts"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not SCRIPT.is_file():
        print(f"{SCRIPT} is missing: run from a RoboAgentLab checkout", file=sys.stderr)
        return 2
    node = os.environ.get("NODE", "node")
    cmd = [node, "--experimental-strip-types", "--no-warnings", str(SCRIPT), *args]
    return subprocess.call(cmd, cwd=str(PI_ROOT))


if __name__ == "__main__":
    sys.exit(main())
