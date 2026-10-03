"""Protocol A (probe ladder) for the official-scale (ViT-Tiny/14 + AdaLN)
baseline_k1/baseline_k3/TI-JEPA checkpoints trained on REAL dm_control
Reacher pixels by `reacher_train_official.py`.

Regresses held-out embeddings -> `pose4=[sin(q0),cos(q0),sin(q1),cos(q1)]`
(periodic-safe pose) and -> `qvel` (2-dim angular velocity, the physical
quantity the whole paper is about). Mirrors `real_eval_ours.py::protocol_a`
+ the official-scale `ti_jepa/eval/probe_ladder.py`'s "single/pair/window/
explicit-v" ladder structure.
"""

from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

sys.path.insert(0, _REPO_ROOT)
from ti_jepa.models import LeWMStyleEncoder, TIJEPAEncoder  # noqa: E402

from reacher_data import RealReacherWindowsFromCache, collate  # noqa: E402

ALPHAS = np.logspace(-3, 4, 15)


def load_encoder(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    args = state["args"]
    if args.get("model") == "tijepa":
        encoder = TIJEPAEncoder(image_size=args["image_size"], q_dim=args["q_dim"], v_dim=args["v_dim"],
                                 k=args["k"], feat_dim=args["feat_dim"], backbone="vit").to(device)
        model_type = "tijepa"
    else:
        encoder = LeWMStyleEncoder(image_size=args["image_size"], z_dim=args["z_dim"],
                                    feat_dim=args["feat_dim"], backbone="vit").to(device)
        model_type = "baseline"
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    return encoder, args, model_type


def fit_probe_cv(X, Y, train_mask, val_mask, seed):
    Xtr, Ytr, Xval, Yval = X[train_mask], Y[train_mask], X[val_mask], Y[val_mask]
    preds = []
    for d in range(Y.shape[-1]):
        r = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr[:, d])
        preds.append(r.predict(Xval))
    pred = np.stack(preds, axis=-1)
    r2 = r2_score(Yval, pred, multioutput="raw_values")
    g = np.random.default_rng(seed + 999)
    shuf = g.permutation(len(Yval))
    r2_shuf = r2_score(Yval[shuf], pred, multioutput="raw_values")
    return {"r2_mean": float(np.mean(r2)), "r2_mean_SHUFFLED_CONTROL": float(np.mean(r2_shuf)), "n": int(len(Yval))}


@torch.no_grad()
def run_protocol_a(encoder, model_type, args, device, cache_path, seed=0):
    k = args["k"]
    ds = RealReacherWindowsFromCache(cache_path, k=k, split="val", seed=0)
    print(f"protocol A: {len(ds)} held-out windows", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)

    z_single, z_pair, z_window, v_explicit = [], [], [], []
    pose_next, vel_next = [], []
    for batch in loader:
        pixels = batch["o_ctx"].to(device)  # (B,k,3,224,224)
        b = pixels.shape[0]
        if model_type == "baseline":
            z_ctx = encoder(pixels.reshape(b * k, 3, args["image_size"], args["image_size"])).reshape(b, k, -1)
            z_single.append(z_ctx[:, -1].cpu().numpy())
            z_pair.append(z_ctx[:, -2:].reshape(b, -1).cpu().numpy())
            z_window.append(z_ctx.reshape(b, -1).cpu().numpy())
        else:
            q_window, v_t, _ = encoder.forward_window(pixels)
            z_single.append(q_window[:, -1].cpu().numpy())
            z_pair.append(q_window[:, -2:].reshape(b, -1).cpu().numpy())
            z_window.append(q_window.reshape(b, -1).cpu().numpy())
            v_explicit.append(v_t.cpu().numpy())
        # "next" (t) ground truth = last context frame (index k-1), matching what
        # the single-frame embedding at that same frame should be able to decode.
        pose_next.append(batch["pose4"][:, k - 1].numpy())
        vel_next.append(batch["qvel"][:, k - 1].numpy())

    z_single, z_pair, z_window = map(lambda x: np.concatenate(x, 0), (z_single, z_pair, z_window))
    pose_next = np.concatenate(pose_next, 0)
    vel_next = np.concatenate(vel_next, 0)

    ep_ids = np.array(ds.window_episode_ids)
    uniq_eps = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    perm_eps = rng.permutation(uniq_eps)
    n_val_eps = max(1, int(0.3 * len(uniq_eps)))
    val_eps = set(perm_eps[:n_val_eps].tolist())
    val_mask = np.isin(ep_ids, list(val_eps))
    train_mask = ~val_mask

    rows = {"single": z_single, "pair": z_pair, "window_k3": z_window}
    if model_type == "tijepa":
        rows["v_explicit"] = np.concatenate(v_explicit, 0)

    results = {}
    for name, X in rows.items():
        r_pose = fit_probe_cv(X, pose_next, train_mask, val_mask, seed)
        r_vel = fit_probe_cv(X, vel_next, train_mask, val_mask, seed)
        results[name] = {"pose4": r_pose, "qvel": r_vel}
        print(f"  feat={name:12s} pose4 r2={r_pose['r2_mean']:+.4f}(shuf={r_pose['r2_mean_SHUFFLED_CONTROL']:+.4f}) "
              f"qvel r2={r_vel['r2_mean']:+.4f}(shuf={r_vel['r2_mean_SHUFFLED_CONTROL']:+.4f})", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_k3_ckpt", required=True)
    ap.add_argument("--baseline_k1_ckpt", required=True)
    ap.add_argument("--tijepa_ckpt", required=True)
    ap.add_argument("--cache", default="cache/reacher_cache.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args_cli = ap.parse_args()

    out = {}
    for name, ckpt in [("baseline_k3", args_cli.baseline_k3_ckpt), ("baseline_k1", args_cli.baseline_k1_ckpt),
                        ("tijepa", args_cli.tijepa_ckpt)]:
        print(f"=== {name} ===", flush=True)
        encoder, args, model_type = load_encoder(ckpt, args_cli.device)
        out[name] = run_protocol_a(encoder, model_type, args, args_cli.device, args_cli.cache)

    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved ->", args_cli.out, flush=True)


if __name__ == "__main__":
    main()
