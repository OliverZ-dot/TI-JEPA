"""Fit linear (Ridge) decode-only probes: current-frame representation ->
true qpos / qvel, on our OWN trained (baseline_k1 / TI-JEPA) checkpoints,
using held-out windows from the Reacher cache. Episode-level split (mirrors
Protocol A's anti-leakage fix). Purely a read-out tool for turning latent
rollouts into physical trajectories for the CEM cost function / reporting --
never touches training, exactly like `ti_jepa/eval/kill_experiment.py`'s
`fit_position_probe` for InertiaBall or `real_pusht/kill_experiment.py`'s for
real PushT.

IMPORTANT: the shoulder joint is an *unlimited* hinge -- the dataset's raw
`qpos[0]` is an unwrapped angle that accumulates across full rotations
(observed range in our cache: [-8.2, 6.9] rad, i.e. several full turns), so
it is fundamentally not linearly recoverable from a single image (two frames
that look identical, e.g. angle and angle+2*pi, have wildly different raw
values). We therefore probe/decode/plan in `[sin(shoulder), cos(shoulder),
wrist]` space (3-dim, `pose3` below) instead of raw `[shoulder, wrist]` --
smooth, bounded, and periodicity-free, exactly analogous to using fingertip
Cartesian position instead of raw joint angle. `qpos_to_pose3`/`pose3_to_qpos`
convert between the two; all CEM costs and success checks operate in `pose3`
space, i.e. minimizing `||pose3(q) - pose3(goal)||^2` is a monotonic,
unwrapped function of the true angular distance to the goal.
"""

from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(_REPO_ROOT, "real_pusht"))
from real_data import RealPushTWindowsFromCache  # noqa: E402


def qpos_to_pose3(qpos: np.ndarray) -> np.ndarray:
    """(...,2) [shoulder, wrist] -> (...,3) [sin(shoulder), cos(shoulder), wrist]."""
    shoulder, wrist = qpos[..., 0], qpos[..., 1]
    return np.stack([np.sin(shoulder), np.cos(shoulder), wrist], axis=-1)


@torch.no_grad()
def _collect(model_type, encoder, k, device, cache_path, split, seed, max_windows=6000):
    ds = RealPushTWindowsFromCache(cache_path, k=k, split=split, seed=seed, window_stride=3)
    n = min(len(ds), max_windows)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()

    reps, pose3_true, qvel_true, v_model = [], [], [], []
    bs = 256
    for start in range(0, n, bs):
        batch_idx = idx[start : start + bs]
        pix = torch.stack([ds[i]["pixels"] for i in batch_idx]).to(device)  # (B,k+1,3,S,S)
        st = torch.stack([ds[i]["state"] for i in batch_idx])                # (B,k+1,4) qpos+qvel
        o_ctx = pix[:, :k]
        if model_type == "baseline":
            b = o_ctx.shape[0]
            z_ctx = encoder(o_ctx.reshape(b * k, *o_ctx.shape[2:])).reshape(b, k, -1)
            reps.append(z_ctx[:, -1, :].cpu().numpy())
        else:
            q_window, v_t, _ = encoder.forward_window(o_ctx)
            reps.append(q_window[:, -1, :].cpu().numpy())
            v_model.append(v_t.cpu().numpy())
        qpos_np = st[:, k - 1, :2].numpy()
        pose3_true.append(qpos_to_pose3(qpos_np))
        qvel_true.append(st[:, k - 1, 2:].numpy())

    reps = np.concatenate(reps, axis=0)
    pose3_true = np.concatenate(pose3_true, axis=0)
    qvel_true = np.concatenate(qvel_true, axis=0)
    v_model = np.concatenate(v_model, axis=0) if v_model else None
    return reps, pose3_true, qvel_true, v_model


def _r2(pred, y):
    return float(1 - np.mean((pred - y) ** 2) / np.mean((y - y.mean(0)) ** 2))


def fit_pose3_probe(model_type, encoder, k, device, cache_path, seed=0):
    """rep -> [sin(shoulder), cos(shoulder), wrist]. Used for both reporting
    and the CEM cost function's position decode."""
    X_tr, y_tr, _, _ = _collect(model_type, encoder, k, device, cache_path, "train", seed)
    X_te, y_te, _, _ = _collect(model_type, encoder, k, device, cache_path, "val", seed)
    reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
    r2 = _r2(reg.predict(X_te), y_te)
    print(f"  [{model_type}] pose3 probe held-out R^2 = {r2:.3f} (n_train={len(X_tr)}, n_val={len(X_te)})",
          flush=True)
    return reg


def fit_qvel_probe_tijepa(encoder, k, device, cache_path, seed=0):
    _, _, y_tr, v_tr = _collect("tijepa", encoder, k, device, cache_path, "train", seed)
    _, _, y_te, v_te = _collect("tijepa", encoder, k, device, cache_path, "val", seed)
    reg = Ridge(alpha=1.0).fit(v_tr, y_tr)
    r2 = _r2(reg.predict(v_te), y_te)
    print(f"  [tijepa] v_t -> true qvel probe held-out R^2 = {r2:.3f}", flush=True)
    return reg
