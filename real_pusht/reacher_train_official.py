"""Train baseline_k1 / baseline_k3 / TI-JEPA on REAL dm_control Reacher
pixels, at OFFICIAL LeWM scale (ViT-Tiny/14 encoder + AdaLN transformer
predictor -- same architecture + hyperparameter recipe validated on the
custom CartPole/Pendulum official-scale runs, see the results log r14), reusing
`ti_jepa.train.train_baseline` / `train_tijepa` UNMODIFIED -- only the data
source changes (real MuJoCo pixels via `RealReacherWindowsFromCache`
instead of the synthetic renderer's `WindowDataset`).

Motivation: the FIRST attempt at real Reacher (see the results log's "Attempted
extension" section) used the small-CNN recipe and failed with a
representation-learning capacity ceiling (pose R^2 <= 0.19, v->qvel R^2 ~=
0.00, i.e. exactly chance) -- diagnosed there as likely fixable by "a bigger
backbone (closer to the official ViT-Tiny scale)". Official-scale CartPole
already demonstrated exactly this kind of fix works (explicit v probe
r: 0.30 -> 0.91 switching small-CNN -> ViT-Tiny). This script is that same
fix applied to the one real, official LeWM benchmark whose physics
(torque-controlled, near-frictionless dof_damping=0.01, real coasting
motion) make it a legitimate additional test of the identifiability claim
on REAL pixels (not just custom-toy or frozen-official-checkpoint data).
"""

from __future__ import annotations

import argparse
import json
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

import torch

sys.path.insert(0, _REPO_ROOT)
from ti_jepa.train import train_baseline, train_tijepa  # noqa: E402

from reacher_data import RealReacherWindowsFromCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "tijepa"], required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_reg", type=float, default=5.0)
    ap.add_argument("--lambda_v", type=float, default=1.0)
    ap.add_argument("--feat_dim", type=int, default=192)
    ap.add_argument("--q_dim", type=int, default=96)
    ap.add_argument("--v_dim", type=int, default=96)
    ap.add_argument("--z_dim", type=int, default=192)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--cache", type=str, default="cache/reacher_cache.npz")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    train_ds = RealReacherWindowsFromCache(args.cache, k=args.k, split="train", seed=args.seed)
    print(f"train windows: {len(train_ds)}", flush=True)

    if args.model == "baseline":
        modules, history = train_baseline(
            train_ds, device, args.steps, args.k, args.z_dim, args.lambda_reg, args.batch_size, args.lr,
            image_size=args.image_size, action_dim=2, feat_dim=args.feat_dim,
            backbone="vit", predictor_type="adaln",
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        )
    else:
        modules, history = train_tijepa(
            train_ds, device, args.steps, args.k, args.q_dim, args.v_dim,
            args.lambda_reg, args.lambda_v, args.batch_size, args.lr,
            image_size=args.image_size, action_dim=2, feat_dim=args.feat_dim,
            backbone="vit", predictor_type="adaln",
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    state = {name: m.state_dict() for name, m in modules.items()}
    args_dict = vars(args)
    args_dict["action_dim"] = 2
    state["args"] = args_dict
    torch.save(state, args.out)
    print("saved checkpoint to", args.out, flush=True)

    hist_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print("saved history to", hist_path, flush=True)


if __name__ == "__main__":
    main()
