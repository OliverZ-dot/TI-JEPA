"""In-RAM windowed dataset over a `reacher_build_cache.py`-produced .npz for
the REAL dm_control Reacher benchmark. Schema for `__getitem__` mirrors
`ti_jepa/data.py::WindowDataset` exactly (`o_ctx`/`o_next`/`action` keys) so
`ti_jepa.train.train_baseline` / `train_tijepa` can be called UNMODIFIED,
just swapping the data source (synthetic renderer -> real MuJoCo pixels).

Pose ground truth for eval-only probes: dm_control Reacher's shoulder joint
(`qpos[:,0]`) is an *unlimited* hinge (angle unwraps/accumulates across full
rotations), so a linear probe on raw `qpos[:,0]` is meaningless by
construction (this bit us on the first from-scratch attempt at this
environment, see `the results log`'s "Attempted extension" section -- fixed here
from the start). We instead expose `pose4 = [sin(q0), cos(q0), sin(q1),
cos(q1)]` (4-dim, periodic-safe) as the probe target; `qvel` (2-dim, angular
velocity, not periodic) is used as-is.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RealReacherWindowsFromCache(Dataset):
    def __init__(self, cache_path: str, k: int = 3, split: str = "train",
                 val_frac: float = 0.1, seed: int = 0, window_stride: int = 2):
        d = np.load(cache_path)
        self.pixels = d["pixels"]        # (N,224,224,3) uint8, full cache in RAM
        self.actions = d["actions"]      # (N,2) float32
        self.qpos = d["qpos"]            # (N,2) float32, raw (shoulder unwrapped)
        self.qvel = d["qvel"]            # (N,2) float32
        self.finger_pos = d["finger_pos"]  # (N,2) float32
        self.target_pos = d["target_pos"]  # (N,2) float32
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

    def pose4(self, g_start: int, n: int) -> np.ndarray:
        """[sin(q0),cos(q0),sin(q1),cos(q1)] for n consecutive frames starting
        at g_start -- periodic-safe pose target, eval-only."""
        qp = self.qpos[g_start : g_start + n]  # (n,2)
        return np.stack([np.sin(qp[:, 0]), np.cos(qp[:, 0]), np.sin(qp[:, 1]), np.cos(qp[:, 1])], axis=-1)

    def __getitem__(self, idx):
        g_start = self.index[idx]
        k = self.k
        pix = self.pixels[g_start : g_start + k + 1].astype(np.float32) / 255.0
        pix = torch.from_numpy(pix).permute(0, 3, 1, 2)  # (k+1,3,224,224)
        o_ctx = pix[:k]
        o_next = pix[k]
        act = torch.from_numpy(self.actions[g_start + k - 1].copy())  # (2,) action at last context step
        pose4 = torch.from_numpy(self.pose4(g_start, k + 1))          # (k+1,4)
        qvel = torch.from_numpy(self.qvel[g_start : g_start + k + 1].copy())  # (k+1,2)
        return {"o_ctx": o_ctx, "o_next": o_next, "action": act, "pose4": pose4, "qvel": qvel}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
