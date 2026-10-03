"""训练脚本：在 InertiaBall 上分别训练

  --model baseline  ->  LeWM-style 单帧 target JEPA
  --model tijepa    ->  TI-JEPA（pose/motion 拆分，跨帧 target）

用法：
  python -m ti_jepa.train --model baseline --steps 4000 --out checkpoints/baseline.pt
  python -m ti_jepa.train --model tijepa   --steps 4000 --out checkpoints/tijepa.pt

数据默认缓存在 data/inertia_ball.npz，第一次跑会生成，之后复用。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ti_jepa.data import EpisodeBatch, WindowDataset, cache_path, generate_episodes, split_episodes
from ti_jepa.envs.registry import get_env_spec, make_scaled_config
from ti_jepa.models import LeWMStyleEncoder, HistoryPredictor, RecurrentPredictor, TIJEPAEncoder, TIJEPAPredictor
from ti_jepa.sigreg import SIGReg

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_or_make_data(env_name="inertia_ball", n_train_ep=2400, n_test_ep=400, n_steps=40, image_size=64, seed=0):
    path = cache_path(env_name, image_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        d = np.load(path)
        batch = EpisodeBatch(d["frames"], d["actions"], d["positions"], d["velocities"])
        n_train = int(d["n_train"])
        return split_episodes(batch, n_train)

    cfg = make_scaled_config(env_name, image_size, seed=seed)
    batch = generate_episodes(n_train_ep + n_test_ep, n_steps, cfg, seed=seed, env_name=env_name)
    np.savez_compressed(
        path,
        frames=batch.frames, actions=batch.actions, positions=batch.positions,
        velocities=batch.velocities, n_train=n_train_ep,
    )
    return split_episodes(batch, n_train_ep)


def train_baseline(train_ds, device, steps, k, z_dim, lambda_reg, batch_size, lr,
                    image_size=64, action_dim=2, feat_dim=128, channels=None, log_every=200,
                    backbone="conv", predictor_type="mlp", weight_decay=0.0, grad_clip=0.0):
    encoder = LeWMStyleEncoder(image_size=image_size, z_dim=z_dim, feat_dim=feat_dim, channels=channels,
                                backbone=backbone).to(device)
    if predictor_type == "mlp":
        predictor = HistoryPredictor(per_step_dim=z_dim, k=k, action_dim=action_dim, out_dim=z_dim).to(device)
    elif predictor_type == "adaln":
        from ti_jepa.official_predictor import OfficialBaselinePredictor
        predictor = OfficialBaselinePredictor(per_step_dim=z_dim, k=k, action_dim=action_dim, out_dim=z_dim).to(device)
    elif predictor_type == "rnn":
        # RSSM/Dreamer 风格递归聚合器对照（见 models.py::RecurrentPredictor 顶部注释）。
        predictor = RecurrentPredictor(per_step_dim=z_dim, k=k, action_dim=action_dim, out_dim=z_dim).to(device)
    else:
        raise ValueError(predictor_type)
    sigreg = SIGReg(num_directions=64).to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    it = iter(loader)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        o_ctx = batch["o_ctx"].to(device)      # (B,k,3,H,W)
        o_next = batch["o_next"].to(device)    # (B,3,H,W)
        action = batch["action"].to(device)

        b = o_ctx.shape[0]
        z_ctx = encoder(o_ctx.reshape(b * k, 3, image_size, image_size)).reshape(b, k, z_dim)
        z_next = encoder(o_next)
        z_pred = predictor(z_ctx, action)

        l_pred = torch.mean((z_pred - z_next) ** 2)
        z_all = torch.cat([z_ctx.reshape(-1, z_dim), z_next], dim=0)
        l_reg = sigreg(z_all)
        loss = l_pred + lambda_reg * l_reg

        opt.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"[baseline] step {step}/{steps} loss={loss.item():.5f} "
                  f"l_pred={l_pred.item():.5f} l_reg={l_reg.item():.5f} "
                  f"z_std={z_next.std().item():.4f} ({elapsed:.1f}s)")
            history.append({"step": step, "loss": loss.item(), "l_pred": l_pred.item(), "l_reg": l_reg.item()})

    return {"encoder": encoder, "predictor": predictor}, history


def train_tijepa(train_ds, device, steps, k, q_dim, v_dim, lambda_reg, lambda_v, batch_size, lr,
                  image_size=64, action_dim=2, feat_dim=128, channels=None, log_every=200,
                  backbone="conv", predictor_type="mlp", weight_decay=0.0, grad_clip=0.0):
    encoder = TIJEPAEncoder(image_size=image_size, q_dim=q_dim, v_dim=v_dim, k=k, feat_dim=feat_dim,
                             channels=channels, backbone=backbone).to(device)
    if predictor_type == "mlp":
        predictor = TIJEPAPredictor(q_dim=q_dim, v_dim=v_dim, action_dim=action_dim).to(device)
    elif predictor_type == "adaln":
        from ti_jepa.official_predictor import OfficialTIJEPAPredictor
        predictor = OfficialTIJEPAPredictor(q_dim=q_dim, v_dim=v_dim, action_dim=action_dim).to(device)
    else:
        raise ValueError(predictor_type)
    sigreg = SIGReg(num_directions=64).to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    it = iter(loader)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        o_ctx = batch["o_ctx"].to(device)      # (B,k,3,H,W) frames t-k+1..t
        o_next = batch["o_next"].to(device)    # (B,3,H,W)   frame t+1
        action = batch["action"].to(device)

        # context window (t-k+1..t) -> q_t, v_t, z_t
        q_window, v_t, z_t = encoder.forward_window(o_ctx)
        # target window (t-k+2..t+1) -> q_{t+1}, v_{t+1}
        o_ctx_next_window = torch.cat([o_ctx[:, 1:], o_next.unsqueeze(1)], dim=1)
        q_window_next, v_next, z_next = encoder.forward_window(o_ctx_next_window)
        q_next = q_window_next[:, -1, :]

        q_t = q_window[:, -1, :]
        q_hat, v_hat = predictor(q_t, v_t, action)

        l_q = torch.mean((q_hat - q_next) ** 2)
        l_v = torch.mean((v_hat - v_next) ** 2)
        l_pred = l_q + l_v

        q_all = torch.cat([q_window.reshape(-1, q_dim), q_next], dim=0)
        l_reg = sigreg(q_all)
        v_all = torch.cat([v_t, v_next], dim=0)
        l_var = torch.clamp(1.0 - v_all.std(dim=0).mean(), min=0.0)

        loss = l_pred + lambda_reg * l_reg + lambda_v * l_var

        opt.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"[tijepa]   step {step}/{steps} loss={loss.item():.5f} "
                  f"l_q={l_q.item():.5f} l_v={l_v.item():.5f} l_reg={l_reg.item():.5f} "
                  f"l_var={l_var.item():.5f} v_std={v_all.std().item():.4f} ({elapsed:.1f}s)")
            history.append({
                "step": step, "loss": loss.item(), "l_q": l_q.item(), "l_v": l_v.item(),
                "l_reg": l_reg.item(), "l_var": l_var.item(),
            })

    return {"encoder": encoder, "predictor": predictor}, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["inertia_ball", "pendulum", "cartpole"], default="inertia_ball")
    ap.add_argument("--model", choices=["baseline", "tijepa"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    # 注：说明书 §3.3 建议 λ_reg=0.1（抄官方 LeWM），但那是官方 ViT-Tiny + AdaLN
    # transformer 规模下的数值。在我们这个小 CNN 上，λ_reg=0.1 / lr=3e-4 会让
    # encoder+predictor 联合坍缩（z_std -> ~0，SIGReg 卡在“单点”鞍面出不来，
    # 见 notes/collapse_debug.md）。经验调参：lr=1e-3、λ_reg=1.0 能稳定跑出
    # 非坍缩解（z_std≈1）。迁回官方 ViT/规模时应重新扫一下这两个超参。
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0,
                     help="AdamW weight decay；小CNN配方默认0（未改变已验证结果），"
                          "官方ViT/AdaLN配方建议 1e-3（抄官方 lewm.yaml optimizer.weight_decay）")
    ap.add_argument("--grad_clip", type=float, default=0.0,
                     help="梯度范数裁剪阈值，0=不裁剪（小CNN配方默认）；官方配方"
                          "trainer.gradient_clip_val=1.0，ViT/AdaLN规模建议开启")
    ap.add_argument("--z_dim", type=int, default=8)
    ap.add_argument("--q_dim", type=int, default=4)
    ap.add_argument("--v_dim", type=int, default=4)
    ap.add_argument("--lambda_reg", type=float, default=1.0)
    ap.add_argument("--lambda_v", type=float, default=0.1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat_dim", type=int, default=128,
                     help="backbone 输出特征维度（投影前）；调大是 scale validation 的一部分")
    ap.add_argument("--channels", type=str, default=None,
                     help="逗号分隔的卷积通道数，如 '64,128,128,128,128'（更宽/更深的backbone）；"
                          "默认 None = 原始 32,64,64,64（4层）。仅在 --backbone conv 时生效。")
    ap.add_argument("--backbone", choices=["conv", "vit"], default="conv",
                     help="conv=自建小CNN（默认）；vit=官方规模 ViT-Tiny/14（见 vit_backbone.py），"
                          "用于'扩大规模，尝试原本LeWM规模'验证实验")
    ap.add_argument("--predictor", choices=["mlp", "adaln", "rnn"], default="mlp",
                     help="mlp=自建小MLP predictor（默认，拼接k帧窗口）；adaln=官方规模 AdaLN "
                          "transformer (depth6/heads16/mlp_dim2048, 见 official_predictor.py)；"
                          "rnn=RSSM/Dreamer风格递归聚合器（GRUCell逐步吸收k帧，只用于 "
                          "--model baseline，见 models.py::RecurrentPredictor）")
    ap.add_argument("--image_size", type=int, default=64,
                     help="渲染分辨率；--backbone vit 时通常配 224（官方规模），像素装饰字段会"
                          "按 image_size/64 等比缩放（见 envs/registry.py::make_scaled_config）")
    args = ap.parse_args()
    channels = [int(c) for c in args.channels.split(",")] if args.channels else None
    if args.backbone == "vit":
        assert args.feat_dim == 192, "backbone=vit 要求 feat_dim=192（ViT-Tiny 的 hidden_size）"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    spec = get_env_spec(args.env)
    train_batch, test_batch = get_or_make_data(env_name=args.env, image_size=args.image_size)
    train_ds = WindowDataset(train_batch, k=args.k)
    print(f"[{args.env}] train windows: {len(train_ds)} (episodes={train_batch.frames.shape[0]})")

    if args.model == "baseline":
        modules, history = train_baseline(
            train_ds, device, args.steps, args.k, args.z_dim, args.lambda_reg, args.batch_size, args.lr,
            image_size=args.image_size, action_dim=spec.action_dim, feat_dim=args.feat_dim, channels=channels,
            backbone=args.backbone, predictor_type=args.predictor,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        )
    else:
        modules, history = train_tijepa(
            train_ds, device, args.steps, args.k, args.q_dim, args.v_dim,
            args.lambda_reg, args.lambda_v, args.batch_size, args.lr,
            image_size=args.image_size, action_dim=spec.action_dim, feat_dim=args.feat_dim, channels=channels,
            backbone=args.backbone, predictor_type=args.predictor,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    state = {name: m.state_dict() for name, m in modules.items()}
    args_dict = vars(args)
    args_dict["action_dim"] = spec.action_dim
    args_dict["image_size"] = args.image_size
    args_dict["channels"] = channels  # overwrite raw CLI string with parsed list (or None)
    state["args"] = args_dict
    torch.save(state, args.out)
    print("saved checkpoint to", args.out)

    hist_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print("saved history to", hist_path)


if __name__ == "__main__":
    main()
