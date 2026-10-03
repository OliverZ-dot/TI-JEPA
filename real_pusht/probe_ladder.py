"""Protocol A (probe ladder) on the REAL official LeWM checkpoint + REAL
benchmark data (PushT / Reacher / Cube), per the internal project spec (not included in this release) §4.1.

For each environment we ask: can a *linear* probe recover the ground-truth
instantaneous velocity of (a) the agent/robot itself and (b) -- the more
important, novel case -- an unactuated pushed/manipulated OBJECT, from:

  - single frame embedding z_t
  - pair (z_{t-1}, z_t)
  - window (z_{t-2}, z_{t-1}, z_t), k=3 = official LeWM predictor's history_size

This is entirely eval-only: the official pretrained encoder is frozen, we
only fit a Ridge probe on top. No training code/weights/configs in the
official repo are touched.

Methodology fixes vs. a naive version (important -- see §4.1's own warning
"按 episode 划分 train/test，不要随机打乱时间步（防泄漏）"):
  1. train/val split is by EPISODE, not by frame -- otherwise adjacent frames
     of the same continuous trajectory leak across the split and the probe
     can partially memorize per-episode trajectory shape instead of
     genuinely reading instantaneous velocity out of z.
  2. RidgeCV over an alpha grid (a single fixed alpha silently starved
     several rows of signal in an earlier draft of this script).
  3. Every row is reported alongside a SHUFFLED-LABEL control (same features,
     Y permuted across validation clips) to show the empirical chance floor
     for that exact sample size / feature dimensionality.
  4. A ground-truth-POSITION-only baseline (no embedding at all) is included
     for the velocity targets, to quantify how much of any apparent success
     could come from a pose->velocity confound in the data distribution
     (expert/near-expert policies don't move at i.i.d. random velocities from
     every position) rather than from genuinely reading velocity out of z.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from common import EP_KEY, encode_clip_batch, load_clips_dataset, load_model, pearsonr_per_dim

FRAMESKIP = {"pusht": 5, "reacher": 4, "cube": 4, "tworoom": 4}
NUM_STEPS = 6
ALPHAS = np.logspace(-3, 4, 15)


def velocity_targets(env: str, batch: dict) -> dict[str, np.ndarray]:
    out = {}
    if env == "pusht":
        state = batch["state"].float().numpy()
        out["agent_vel(direct,in-state)"] = (state[:, :-1, 5:7], state[:, :-1, :2])
        block_pos = state[:, :, 2:4]
        out["block_vel(finite-diff,GT)"] = (block_pos[:, 1:] - block_pos[:, :-1], block_pos[:, :-1])
    elif env == "reacher":
        qvel = batch["qvel"].float().numpy()
        qpos = batch["qpos"].float().numpy()
        out["joint_vel(direct,GT)"] = (qvel[:, :-1], qpos[:, :-1])
        fpos = batch["finger_pos"].float().numpy()
        out["finger_vel(finite-diff,GT)"] = (fpos[:, 1:] - fpos[:, :-1], fpos[:, :-1])
    elif env == "cube":
        qvel = batch["proprio_joint_vel"].float().numpy()
        qpos_pos = batch["privileged_block_0_pos"].float().numpy()
        out["robot_joint_vel(direct,GT)"] = (qvel[:, :-1], qpos_pos[:, :-1])
        out["block_vel(finite-diff,GT)"] = (qpos_pos[:, 1:] - qpos_pos[:, :-1], qpos_pos[:, :-1])
    elif env == "tworoom":
        pos = batch["pos_agent"].float().numpy()
        out["agent_vel(finite-diff,GT)"] = (pos[:, 1:] - pos[:, :-1], pos[:, :-1])
    else:
        raise ValueError(env)
    return out


EXTRA_KEYS = {
    "pusht": ["state"],
    "reacher": ["qvel", "qpos", "finger_pos"],
    "cube": ["proprio_joint_vel", "privileged_block_0_pos"],
    "tworoom": ["pos_agent"],
}


def build_feature_sets(emb: np.ndarray):
    N, T, D = emb.shape
    feats = {"single(z_i)": [], "pair(z_{i-1},z_i)": [], "window_k3(z_{i-2..i})": []}
    idxs = []
    for i in range(1, T - 1):
        feats["single(z_i)"].append(emb[:, i])
        feats["pair(z_{i-1},z_i)"].append(np.concatenate([emb[:, i - 1], emb[:, i]], axis=-1))
        lo = max(0, i - 2)
        window = emb[:, lo : i + 1]
        if window.shape[1] < 3:
            pad = np.repeat(window[:, :1], 3 - window.shape[1], axis=1)
            window = np.concatenate([pad, window], axis=1)
        feats["window_k3(z_{i-2..i})"].append(window.reshape(window.shape[0], -1))
        idxs.append(i)
    for k in feats:
        feats[k] = np.concatenate(feats[k], axis=0)
    return feats, idxs


def episode_split(ep_ids: np.ndarray, seed: int, val_frac: float = 0.2):
    uniq = np.unique(ep_ids)
    g = np.random.default_rng(seed)
    perm = g.permutation(uniq)
    n_val_ep = max(1, int(val_frac * len(uniq)))
    val_eps = set(perm[:n_val_ep].tolist())
    is_val = np.array([e in val_eps for e in ep_ids])
    return ~is_val, is_val


def fit_eval_probe(X: np.ndarray, Y: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, seed: int):
    Xtr, Ytr, Xval, Yval = X[train_mask], Y[train_mask], X[val_mask], Y[val_mask]
    reg = RidgeCV(alphas=ALPHAS)
    if Y.ndim == 1 or Y.shape[-1] == 1:
        reg.fit(Xtr, Ytr.ravel())
        pred = reg.predict(Xval).reshape(-1, 1)
        Yval2 = Yval.reshape(-1, 1)
    else:
        # RidgeCV doesn't natively support multi-output CV scoring well for
        # picking one alpha per column jointly; fit one RidgeCV per output dim.
        preds = []
        for d in range(Y.shape[-1]):
            r = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr[:, d])
            preds.append(r.predict(Xval))
        pred = np.stack(preds, axis=-1)
        Yval2 = Yval
    r2 = r2_score(Yval2, pred, multioutput="raw_values")
    r = pearsonr_per_dim(pred, Yval2)

    g = np.random.default_rng(seed + 999)
    shuf_idx = g.permutation(len(Yval2))
    r2_shuf = r2_score(Yval2[shuf_idx], pred, multioutput="raw_values")

    return {
        "r2_per_dim": r2.tolist(),
        "r2_mean": float(np.mean(r2)),
        "pearson_r_per_dim": r.tolist(),
        "pearson_r_mean": float(np.mean(np.abs(r))),
        "r2_mean_SHUFFLED_CONTROL": float(np.mean(r2_shuf)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["pusht", "reacher", "cube", "tworoom"], required=True)
    ap.add_argument("--num_clips", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    extra_keys = EXTRA_KEYS[args.env] + [EP_KEY[args.env]]
    model = load_model(args.env, args.device)
    dataset = load_clips_dataset(args.env, num_steps=NUM_STEPS, frameskip=FRAMESKIP[args.env], extra_keys=extra_keys)
    print(f"[{args.env}] dataset has {len(dataset)} clips of length {NUM_STEPS}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    n_clips = min(args.num_clips, len(dataset))
    sub_idx = torch.randperm(len(dataset), generator=g)[:n_clips]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, sub_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_emb, all_extra = [], {k: [] for k in extra_keys}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            emb = encode_clip_batch(model, batch["pixels"], args.device).numpy()
            all_emb.append(emb)
            for k in extra_keys:
                all_extra[k].append(batch[k])
            if bi % 20 == 0:
                print(f"  encoded batch {bi}/{len(loader)}", flush=True)
    emb = np.concatenate(all_emb, axis=0)
    batch_full = {k: torch.cat(v, dim=0) for k, v in all_extra.items()}
    print(f"[{args.env}] total clips encoded: {emb.shape[0]}, emb_dim={emb.shape[-1]}", flush=True)

    ep_ids_per_clip = batch_full[EP_KEY[args.env]][:, 0].numpy()  # (N,) episode id, one per clip
    targets = velocity_targets(args.env, batch_full)
    feats, idxs = build_feature_sets(emb)

    results = {"env": args.env, "num_clips": int(emb.shape[0]), "num_steps": NUM_STEPS,
               "frameskip": FRAMESKIP[args.env], "rows": {}}

    for tname, (tvals, posvals) in targets.items():
        tvals_aligned = np.stack([tvals[:, i - 1] for i in idxs], axis=1)  # align: idx i uses vel computed at frame i-1->i
        pos_aligned = np.stack([posvals[:, i - 1] for i in idxs], axis=1)
        results["rows"][tname] = {}

        Y_flat = tvals_aligned.reshape(-1, tvals_aligned.shape[-1])
        pos_flat = pos_aligned.reshape(-1, pos_aligned.shape[-1])
        ep_flat = np.repeat(ep_ids_per_clip, len(idxs))
        nan_mask = ~np.isnan(Y_flat).any(axis=-1)

        train_mask, val_mask = episode_split(ep_flat, seed=args.seed)
        train_mask &= nan_mask
        val_mask &= nan_mask

        # ground-truth-position-only baseline (no embedding) — quantifies the
        # pose->velocity confound available in this dataset's own action distribution
        res_pos = fit_eval_probe(pos_flat, Y_flat, train_mask, val_mask, seed=args.seed)
        results["rows"][tname]["GT_position_only(confound_baseline)"] = res_pos
        print(f"[{args.env}] target={tname:30s} feat={'GT_position_only':24s} "
              f"r2_mean={res_pos['r2_mean']:+.4f} shuf={res_pos['r2_mean_SHUFFLED_CONTROL']:+.4f}", flush=True)

        for fname, X in feats.items():
            Xc = X[nan_mask]
            res = fit_eval_probe(X, Y_flat, train_mask, val_mask, seed=args.seed)
            results["rows"][tname][fname] = res
            print(f"[{args.env}] target={tname:30s} feat={fname:24s} "
                  f"r2_mean={res['r2_mean']:+.4f} |r|_mean={res['pearson_r_mean']:.4f} "
                  f"shuf={res['r2_mean_SHUFFLED_CONTROL']:+.4f} (n_tr={res['n_train']},n_val={res['n_val']})", flush=True)

    out_path = args.out or f"results/probe_ladder_{args.env}.json"
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.env}] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
