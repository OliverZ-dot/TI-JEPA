"""Train our OWN matched-budget baseline (LeWM-style, single-frame target,
history-conditioned predictor) and TI-JEPA (pose/motion split) on REAL
pusht_expert_train.h5 pixels+actions. Reuses the exact same architectures
(ConvBackbone/SIGReg) validated on InertiaBall in ../ti_jepa/models.py --
only the data source changes (real photos of real PushT instead of a
synthetic renderer), and image_size is bumped to accommodate real visual
complexity.

This is intentionally a smaller-than-official architecture (CNN, not
ViT-Tiny) trained for a modest step budget on a single GPU -- the goal is a
FAIR, matched-compute baseline-vs-TIJEPA comparison on real data (per
the internal project spec (not included in this release)'s staged plan: "第二阶段再在同一数据上重训
LeWM / TI-JEPA"), not to reproduce the official checkpoint's absolute
performance. The official checkpoint results (protocol A/B above) remain
the separate "upper bound reference".
"""

from __future__ import annotations

import argparse
import json
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, _REPO_ROOT)
from ti_jepa.models import LeWMStyleEncoder, HistoryPredictor, TIJEPAEncoder, TIJEPAPredictor
from ti_jepa.sigreg import SIGReg

from real_data import RealPushTWindowsFromCache, collate


def train_baseline(train_ds, device, steps, k, z_dim, lambda_reg, batch_size, lr, image_size, log_every=100):
    encoder = LeWMStyleEncoder(image_size=image_size, z_dim=z_dim).to(device)
    predictor = HistoryPredictor(per_step_dim=z_dim, k=k, action_dim=2, out_dim=z_dim).to(device)
    sigreg = SIGReg(num_directions=64).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                         num_workers=0, collate_fn=collate)
    it = iter(loader)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        pixels = batch["pixels"].to(device)  # (B,k+1,3,H,W)
        o_ctx, o_next = pixels[:, :k], pixels[:, k]
        action = batch["actions"][:, -1].to(device)  # (B,2) last-step action

        b = o_ctx.shape[0]
        z_ctx = encoder(o_ctx.reshape(b * k, 3, image_size, image_size)).reshape(b, k, z_dim)
        z_next = encoder(o_next)
        z_pred = predictor(z_ctx, action)

        l_pred = torch.mean((z_pred - z_next) ** 2)
        z_all = torch.cat([z_ctx.reshape(-1, z_dim), z_next], dim=0)
        l_reg = sigreg(z_all)
        loss = l_pred + lambda_reg * l_reg

        opt.zero_grad(); loss.backward(); opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"[baseline] step {step}/{steps} loss={loss.item():.5f} "
                  f"l_pred={l_pred.item():.5f} l_reg={l_reg.item():.5f} "
                  f"z_std={z_next.std().item():.4f} ({elapsed:.1f}s)", flush=True)
            history.append({"step": step, "loss": loss.item(), "l_pred": l_pred.item(), "l_reg": l_reg.item()})
    return {"encoder": encoder, "predictor": predictor}, history


def train_tijepa(train_ds, device, steps, k, q_dim, v_dim, lambda_reg, lambda_v, batch_size, lr, image_size,
                  sigreg_target="q", log_every=100):
    encoder = TIJEPAEncoder(image_size=image_size, q_dim=q_dim, v_dim=v_dim, k=k).to(device)
    predictor = TIJEPAPredictor(q_dim=q_dim, v_dim=v_dim, action_dim=2).to(device)
    sigreg = SIGReg(num_directions=64).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=lr)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                         num_workers=0, collate_fn=collate)
    it = iter(loader)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        pixels = batch["pixels"].to(device)  # (B,k+1,3,H,W)
        o_ctx, o_next = pixels[:, :k], pixels[:, k]
        action = batch["actions"][:, -1].to(device)

        q_window, v_t, z_t = encoder.forward_window(o_ctx)
        o_ctx_next_window = torch.cat([o_ctx[:, 1:], o_next.unsqueeze(1)], dim=1)
        q_window_next, v_next, z_next = encoder.forward_window(o_ctx_next_window)
        q_next = q_window_next[:, -1, :]
        q_t = q_window[:, -1, :]
        q_hat, v_hat = predictor(q_t, v_t, action)

        l_q = torch.mean((q_hat - q_next) ** 2)
        l_v = torch.mean((v_hat - v_next) ** 2)
        l_pred = l_q + l_v

        q_all = torch.cat([q_window.reshape(-1, q_dim), q_next], dim=0)
        v_all = torch.cat([v_t, v_next], dim=0)
        if sigreg_target == "q":
            # §3.2 default: SIGReg (isotropic-Gaussian pressure) hits pose only;
            # v is regularized separately by a soft variance floor so it isn't
            # forced towards the same isotropic-Gaussian target as q (v's natural
            # marginal is "mostly near zero, occasional bursts", not Gaussian).
            l_reg = sigreg(q_all)
            l_var = torch.clamp(1.0 - v_all.std(dim=0).mean(), min=0.0)
        else:
            # ablation: SIGReg jointly on z=(q,v) concatenated, mirroring how the
            # LeWM-style baseline regularizes its single undifferentiated z. Tests
            # whether pushing v towards an isotropic Gaussian (rather than letting
            # it float near zero for a mostly-static object) changes what protocol
            # A/B measure -- i.e. whether the *placement* of the collapse-prevention
            # regularizer, not just the pose/motion split itself, drives the result.
            z_t_full = torch.cat([q_t, v_t], dim=-1)
            z_next_full = torch.cat([q_next, v_next], dim=-1)
            z_all = torch.cat([z_t_full, z_next_full], dim=0)
            l_reg = sigreg(z_all)
            l_var = torch.zeros((), device=device)

        loss = l_pred + lambda_reg * l_reg + lambda_v * l_var
        opt.zero_grad(); loss.backward(); opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"[tijepa]   step {step}/{steps} loss={loss.item():.5f} "
                  f"l_q={l_q.item():.5f} l_v={l_v.item():.5f} l_reg={l_reg.item():.5f} "
                  f"l_var={l_var.item():.5f} v_std={v_all.std().item():.4f} ({elapsed:.1f}s)", flush=True)
            history.append({"step": step, "loss": loss.item(), "l_q": l_q.item(), "l_v": l_v.item(),
                             "l_reg": l_reg.item(), "l_var": l_var.item()})
    return {"encoder": encoder, "predictor": predictor}, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "tijepa"], required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--image_size", type=int, default=84)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--z_dim", type=int, default=8)
    ap.add_argument("--q_dim", type=int, default=4)
    ap.add_argument("--v_dim", type=int, default=4)
    ap.add_argument("--lambda_reg", type=float, default=1.0)
    ap.add_argument("--lambda_v", type=float, default=0.1)
    ap.add_argument("--sigreg_target", choices=["q", "z"], default="q")
    ap.add_argument("--cache", type=str, default="cache/pusht_cache.npz")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    train_ds = RealPushTWindowsFromCache(args.cache, k=args.k, split="train", seed=args.seed)
    print(f"train windows: {len(train_ds)}", flush=True)
    assert train_ds.image_size == args.image_size, (train_ds.image_size, args.image_size)

    if args.model == "baseline":
        modules, history = train_baseline(train_ds, device, args.steps, args.k, args.z_dim,
                                           args.lambda_reg, args.batch_size, args.lr, args.image_size)
    else:
        modules, history = train_tijepa(train_ds, device, args.steps, args.k, args.q_dim, args.v_dim,
                                         args.lambda_reg, args.lambda_v, args.batch_size, args.lr, args.image_size,
                                         sigreg_target=args.sigreg_target)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    state = {name: m.state_dict() for name, m in modules.items()}
    state["args"] = vars(args)
    torch.save(state, args.out)
    print("saved checkpoint to", args.out, flush=True)

    hist_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
