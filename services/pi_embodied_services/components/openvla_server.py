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

"""RPC server wrapping OpenVLA (openvla/openvla) for LIBERO and for Bridge V2 rigs.

One 7-D action per call from the main camera only (OpenVLA has no wrist camera
or proprioception): the frame is JPEG round-tripped and Lanczos-resized to 224 like its
``resize_image``, prompted with ``In: What action should the robot take to {task}?\\nOut:`` and
decoded greedily. ``--suite`` picks the checkpoint and what follows (``OPENVLA_CHECKPOINTS``):

- a LIBERO suite: that suite's published fine-tune. The frame comes already rotated 180 degrees
  by RLinf's LiberoEnv (as OpenVLA's LIBERO evaluation rotates it); it is center-cropped to 90%
  of its area and resized back like its ``crop_and_resize`` (the fine-tunes were evaluated with
  ``center_crop=True``; ``--no-center-crop`` skips it), and the gripper is binarised and inverted
  as in its LIBERO evaluation: the answer is in LIBERO's OSC space (``libero_osc``).
- ``bridge_orig``: the base ``openvla/openvla-7b`` (trained on Open X-Embodiment, Bridge V2
  included) un-normalised with its Bridge V2 statistics, as OpenVLA's own Bridge evaluation and
  SimplerEnv's WidowX setup run it: no center crop (the base model was evaluated without it), and
  the raw action, Bridge's end-effector delta [dx, dy, dz, droll, dpitch, dyaw, open in [0, 1]]
  (``bridge_ee_delta_euler``), which the robot's env server translates into its control mode.

The checkpoints are pinned by commit hash in ``OPENVLA_CHECKPOINTS`` and fetched through
``HF_ENDPOINT`` (hf-mirror) when ``--model-path`` is not a directory. Run it in the ``openvla``
venv (``services[openvla]``, see README).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pi_embodied_services.components.vla_adapter_base import (
    BRIDGE_EE_DELTA_EULER,
    DEFAULT_SUITE,
    LIBERO_OSC,
    ChunkVLAFacade,
    Frame,
    add_server_args,
    apply_cuda_device,
    center_crop_resize,
    libero_gripper,
    reported_suite,
    resize_jpeg_lanczos,
    snapshot,
)
from pi_embodied_services.utils.logging import get_logger

logger = get_logger("openvla_server")


@dataclass(frozen=True)
class Checkpoint:
    """A published checkpoint and how it runs: its repo and pinned commit, the ``norm_stats``
    key it un-normalises with (None: the checkpoint's only one), whether its evaluation center
    crops, and the action space it answers in."""

    repo: str
    revision: str
    unnorm_key: str | None
    center_crop: bool
    action_space: str


#: The checkpoints published by the OpenVLA authors, by the suite ``--suite`` names: the four
#: LIBERO fine-tunes, and the base model with its Bridge V2 statistics (``bridge_orig``, one of
#: its 24 Open X-Embodiment datasets).
OPENVLA_CHECKPOINTS: dict[str, Checkpoint] = {
    "libero_spatial": Checkpoint(
        "openvla/openvla-7b-finetuned-libero-spatial",
        "962318cec55ac10993ff0f5f43eda9a270b4c873",
        None,
        True,
        LIBERO_OSC,
    ),
    "libero_object": Checkpoint(
        "openvla/openvla-7b-finetuned-libero-object",
        "287d6cfdf12d07b1449505f66d9bf3550257e9b3",
        None,
        True,
        LIBERO_OSC,
    ),
    "libero_goal": Checkpoint(
        "openvla/openvla-7b-finetuned-libero-goal",
        "fa5ae1e7509348889295bba8e08621d8b55e9baf",
        None,
        True,
        LIBERO_OSC,
    ),
    "libero_10": Checkpoint(
        "openvla/openvla-7b-finetuned-libero-10",
        "80970322773f81baa2e22fe495d0487b93a05cfa",
        None,
        True,
        LIBERO_OSC,
    ),
    "bridge_orig": Checkpoint(
        "openvla/openvla-7b",
        "47a0ec7fc4ec123775a391911046cf33cf9ed83f",
        "bridge_orig",
        False,
        BRIDGE_EE_DELTA_EULER,
    ),
}
IMAGE_SIZE = 224
#: OpenVLA's LIBERO evaluation crops the centered 90% of the image area (``center_crop=True``).
CROP_SCALE = 0.9

Policy = Callable[[np.ndarray, str], np.ndarray]


def prompt_for(instruction: str) -> str:
    """OpenVLA's LIBERO prompt (``get_vla_action``)."""
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class OpenVLAFacade(ChunkVLAFacade):
    """``policy(image_224_rgb, instruction) -> raw 7-D action`` (denormalised, gripper in [0, 1]),
    answered in ``action_space``: LIBERO's (the gripper converted) or the raw Bridge action."""

    SERVICE_NAME = "openvla"
    horizon = 1
    uses_wrist = False

    def __init__(
        self,
        *,
        policy: Policy,
        model: str,
        revision: str | None,
        suite: str | None = None,
        center_crop: bool = True,
        action_space: str = LIBERO_OSC,
    ):
        self._policy = policy
        self._center_crop = center_crop
        super().__init__(
            model=model, revision=revision, suite=suite, action_space=action_space
        )

    def _act(self, frame: Frame) -> np.ndarray:
        image = resize_jpeg_lanczos(frame.main, IMAGE_SIZE)
        if self._center_crop:
            image = center_crop_resize(image, CROP_SCALE)
        raw = np.asarray(self._policy(image, frame.instruction), np.float32)
        return libero_gripper(raw) if self._action_space == LIBERO_OSC else raw


def load_policy(path: str, unnorm_key: str | None, attn: str) -> tuple[Policy, str]:
    """The checkpoint's processor + ``OpenVLAForActionPrediction`` on the GPU, as a :data:`Policy`."""
    import torch
    from PIL import Image
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    # The LIBERO fine-tunes ship no remote code: register the repo's classes (its get_vla does).
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    keys = list(config.norm_stats)
    if unnorm_key is None:
        if len(keys) != 1:
            raise ValueError(f"--unnorm-key is required; the checkpoint has {keys}")
        unnorm_key = keys[0]
    elif unnorm_key not in keys:
        raise ValueError(
            f"unknown --unnorm-key {unnorm_key!r}; the checkpoint has {keys}"
        )
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        path,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).cuda()
    model.eval()

    def policy(image: np.ndarray, instruction: str) -> np.ndarray:
        inputs = processor(prompt_for(instruction), Image.fromarray(image)).to(
            "cuda", dtype=torch.bfloat16
        )
        with torch.inference_mode():
            action = model.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False
            )
        return np.asarray(action, dtype=np.float32)

    return policy, unnorm_key


def main() -> None:
    p = argparse.ArgumentParser()
    add_server_args(p)
    p.add_argument(
        "--model-path",
        default=os.environ.get("OPENVLA_CHECKPOINT_PATH"),
        help="checkpoint directory or Hugging Face repo id (default OPENVLA_CHECKPOINT_PATH env)",
    )
    p.add_argument(
        "--suite",
        choices=sorted(OPENVLA_CHECKPOINTS),
        default=None,
        help="the published checkpoint to fetch when --model-path is unset (default "
        "libero_spatial): a LIBERO fine-tune, or bridge_orig (openvla-7b with its Bridge V2 "
        "statistics, for a Bridge rig such as ManiSkill's widowx250s); with --model-path, the suite "
        "that checkpoint's actions are un-normalised with, reported by vla.info so the robot "
        "refuses a tool it cannot run",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="commit hash to fetch (default: the pin in OPENVLA_CHECKPOINTS for --suite)",
    )
    p.add_argument(
        "--unnorm-key",
        default=None,
        help="norm_stats key for action denormalisation (default: the suite's, else the "
        "checkpoint's only key)",
    )
    p.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="the 0.9 center crop of OpenVLA's LIBERO evaluation (default: on for the LIBERO "
        "fine-tunes, which were evaluated with it; off for bridge_orig)",
    )
    p.add_argument(
        "--attn", default="sdpa", help="attention implementation (sdpa | eager)"
    )
    args = p.parse_args()
    apply_cuda_device(args.cuda_device)

    ckpt = OPENVLA_CHECKPOINTS[args.suite or DEFAULT_SUITE]
    model = args.model_path or ckpt.repo
    revision = args.revision or (ckpt.revision if model == ckpt.repo else None)
    suite = reported_suite(args.suite, model, ckpt.repo)
    center_crop = ckpt.center_crop if args.center_crop is None else args.center_crop
    t0 = time.time()
    path = snapshot(model, revision)
    logger.info("loading OpenVLA from %s (revision=%s) ...", path, revision)
    policy, unnorm_key = load_policy(
        path, args.unnorm_key or ckpt.unnorm_key, args.attn
    )
    logger.info(
        "model ready in %.1fs (unnorm_key=%s, action_space=%s, center_crop=%s)",
        time.time() - t0,
        unnorm_key,
        ckpt.action_space,
        center_crop,
    )
    OpenVLAFacade(
        policy=policy,
        model=model,
        revision=revision,
        suite=suite,
        center_crop=center_crop,
        action_space=ckpt.action_space,
    ).serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
