"""Shared plumbing for the *real* PushT / Cube / Reacher / TwoRoom experiments.

We do NOT modify or import training entry points from the official
`le-wm` repo or `stable_worldmodel` -- we only reuse them as read-only
libraries (official checkpoints, official HDF5Dataset, official pixel
normalization) exactly the way `fit_probe.py` in that repo already does,
so numbers are directly comparable to what the official codebase would
produce. This mirrors the internal project spec (not included in this release) §12's "zero-touch to
training" rule: everything here is eval-only.
"""

from __future__ import annotations

import os

import sys
from pathlib import Path

import numpy as np
import torch

LEWM_REPO = Path(os.environ.get("LEWM_REPO", "./le-wm"))
if str(LEWM_REPO) not in sys.path:
    sys.path.insert(0, str(LEWM_REPO))

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset  # noqa: E402

from pretrained import load_pretrained_model  # noqa: E402

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

EP_KEY = {
    "pusht": "episode_idx",
    "tworoom": "ep_idx",
    "cube": "ep_idx",
    "reacher": "ep_idx",
}

DATASET_NAME = {
    "pusht": "pusht_expert_train",
    "tworoom": "tworoom",
    "cube": "ogbench/cube_single_expert",
    "reacher": "dmc/reacher_random",
}

# quantity_key + dim used by each env's h5 (see config/explore/*.yaml)
STATE_KEY = {
    "pusht": "state",
    "tworoom": "proprio",
    "cube": "proprio",
    "reacher": "proprio",
}

# action_dim the official checkpoint's action_encoder was built with
# (action_block * raw action dim, see config/explore/<env>.yaml)
ACTION_DIM = {
    "pusht": 10,
    "tworoom": 10,
    "cube": 25,
    "reacher": 10,
}


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """(...,H,W,C) uint8 -> (...,C,H,W) float32, ImageNet-normalized.
    Byte-for-byte the same as explore_train.py's `_normalize_pixels`."""
    pixels = pixels.float() / 255.0
    pixels = pixels.movedim(-1, -3)
    extra_dims = pixels.ndim - 3
    mean = _IMAGENET_MEAN.to(pixels.device).view(*([1] * extra_dims), 3, 1, 1)
    std = _IMAGENET_STD.to(pixels.device).view(*([1] * extra_dims), 3, 1, 1)
    return (pixels - mean) / std


def align_pixels(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.shape[-1] not in (1, 3):
        pixels = pixels.movedim(-3, -1)
    return pixels


def load_model(env: str, device: str = "cuda"):
    model = load_pretrained_model(env, ACTION_DIM[env], device)
    model.eval()
    return model


def load_clips_dataset(env: str, num_steps: int, frameskip: int, extra_keys: list[str] | None = None):
    keys = ["pixels", "action", EP_KEY[env]]
    for k in extra_keys or []:
        if k not in keys:
            keys.append(k)
    return HDF5Dataset(
        DATASET_NAME[env],
        keys_to_load=keys,
        frameskip=frameskip,
        num_steps=num_steps,
        cache_dir=swm.data.utils.get_cache_dir(),
    )


@torch.no_grad()
def encode_clip_batch(model, pixels: torch.Tensor, device: str) -> torch.Tensor:
    """pixels: (B,T,H,W,C) or (B,T,C,H,W) uint8 -> emb (B,T,D) float32 cpu."""
    pixels = align_pixels(pixels).to(device)
    pixels = normalize_pixels(pixels)
    out = model.encode({"pixels": pixels})
    return out["emb"].cpu()


def pearsonr_per_dim(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    out = np.zeros(pred.shape[-1])
    for i in range(pred.shape[-1]):
        p, t = pred[:, i], true[:, i]
        if p.std() < 1e-8 or t.std() < 1e-8:
            out[i] = 0.0
        else:
            out[i] = np.corrcoef(p, t)[0, 1]
    return out
