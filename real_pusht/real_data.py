"""Direct h5py windowing over the REAL pusht_expert_train.h5 dataset --
independent of stable_worldmodel's Dataset class so this training pipeline
has no dependency on (and cannot accidentally touch) the official training
entry points. Purely reads pixels/state/action.

Window = k+1 frames spaced `frameskip` raw env-steps apart (matches the
official model's temporal resolution: frameskip=5, history_size=3). Action
fed to the predictor for the transition frame_i -> frame_{i+1} is the MEAN
of the `frameskip` raw actions in between (a simplification vs. the official
10-dim concatenation -- fine here since we are training our OWN
matched-budget baseline/TI-JEPA pair from scratch, not trying to load official
weights into this architecture).
"""

from __future__ import annotations

import os

import hdf5plugin  # noqa: F401  -- registers the HDF5 compression filter plugins pusht_expert_train.h5 needs
import numpy as np
import torch
from torch.utils.data import Dataset

H5_PATH = os.environ.get("PUSHT_H5", os.path.expanduser(
    "~/.stable_worldmodel/datasets/pusht_expert_train.h5"))


class RealPushTWindows(Dataset):
    def __init__(self, k: int = 3, frameskip: int = 5, image_size: int = 84,
                 split: str = "train", val_frac: float = 0.1, seed: int = 0,
                 max_episodes: int | None = None):
        import h5py

        self.k = k
        self.frameskip = frameskip
        self.image_size = image_size
        f = h5py.File(H5_PATH, "r")
        self.ep_len = f["ep_len"][:]
        self.ep_offset = f["ep_offset"][:]
        self.f = f  # keep open; h5py supports concurrent reads, workers=0

        n_ep = len(self.ep_len) if max_episodes is None else min(max_episodes, len(self.ep_len))
        g = np.random.default_rng(seed)
        perm = g.permutation(len(self.ep_len))[:n_ep]
        n_val = max(1, int(val_frac * n_ep))
        self.episodes = perm[n_val:] if split == "train" else perm[:n_val]

        span = k * frameskip
        self.index = []  # (episode_id, start_within_ep)
        for ep in self.episodes:
            L = self.ep_len[ep]
            if L <= span + 1:
                continue
            n_starts = L - span - 1
            for s in range(0, n_starts, 3):  # subsample starts to keep index size sane
                self.index.append((int(ep), int(s)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, s = self.index[idx]
        off = self.ep_offset[ep]
        frame_idx = [off + s + i * self.frameskip for i in range(self.k + 1)]
        pixels = self.f["pixels"][frame_idx[0] : frame_idx[-1] + 1 : self.frameskip]  # (k+1,224,224,3)
        pixels = pixels.astype(np.float32) / 255.0
        pixels = torch.from_numpy(pixels).permute(0, 3, 1, 2)  # (k+1,3,224,224)
        pixels = torch.nn.functional.interpolate(
            pixels, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

        actions = []
        for i in range(self.k + 1 - 1):
            a0, a1 = off + s + i * self.frameskip, off + s + (i + 1) * self.frameskip
            actions.append(self.f["action"][a0:a1].mean(axis=0))
        actions = torch.from_numpy(np.stack(actions).astype(np.float32))  # (k, 2)

        state = self.f["state"][frame_idx[0] : frame_idx[-1] + 1 : self.frameskip].astype(np.float32)  # (k+1,7)
        return {"pixels": pixels, "actions": actions, "state": torch.from_numpy(state)}


def collate(batch):
    return {
        "pixels": torch.stack([b["pixels"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "state": torch.stack([b["state"] for b in batch]),
    }


class RealPushTWindowsFromCache(Dataset):
    """Fast in-RAM dataset over a `build_cache.py`-produced .npz: k+1
    CONSECUTIVE decimated frames (already frameskip-spaced) per window.
    Episode-level train/val split (held out for both training the
    encoder/predictor AND for any later probe fitting on this checkpoint)."""

    def __init__(self, cache_path: str, k: int = 3, split: str = "train",
                 val_frac: float = 0.1, seed: int = 0, window_stride: int = 2):
        d = np.load(cache_path)
        self.pixels = d["pixels"]       # (N,S,S,3) uint8, full cache in RAM
        self.actions = d["actions"]     # (N,2)
        self.state = d["state"]         # (N,7)
        self.ep_starts = d["ep_starts"]
        self.ep_lens = d["ep_lens"]
        self.image_size = int(d["image_size"])
        self.k = k

        n_ep = len(self.ep_starts)
        g = np.random.default_rng(seed)
        perm = g.permutation(n_ep)
        n_val = max(1, int(val_frac * n_ep))
        eps = perm[n_val:] if split == "train" else perm[:n_val]

        self.index = []
        self.window_episode_ids = []
        for ep in eps:
            start, L = int(self.ep_starts[ep]), int(self.ep_lens[ep])
            for s in range(0, max(0, L - k - 1), window_stride):
                self.index.append(start + s)
                self.window_episode_ids.append(int(ep))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        g_start = self.index[idx]
        pix = self.pixels[g_start : g_start + self.k + 1].astype(np.float32) / 255.0
        pix = torch.from_numpy(pix).permute(0, 3, 1, 2)  # (k+1,3,S,S)
        act = torch.from_numpy(self.actions[g_start : g_start + self.k].copy())  # (k,2)
        st = torch.from_numpy(self.state[g_start : g_start + self.k + 1].copy())  # (k+1,7)
        return {"pixels": pix, "actions": act, "state": st}
