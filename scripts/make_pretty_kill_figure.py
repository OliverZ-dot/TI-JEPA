"""Publication-quality replacement for the plain kill-experiment figures in
results/{env}/kill_experiment*.png. Combines:
  (top)    a real 2-row x N-column frame strip of the actual ground-truth
           rollout for both velocity branches (the visual proof that
           identical-looking frames diverge under the real physics),
  (bottom) a clean line plot of each arm's decoded blind-rollout position,
           reusing the exact loading / probing / rollout code in
           ti_jepa.eval.kill_experiment so the curves are the real model
           behavior, not a mockup.

Usage:
    python3 scripts/make_pretty_kill_figure.py --env pendulum \
        --baseline_ckpt checkpoints/pendulum/baseline.pt \
        --baseline_k1_ckpt checkpoints/pendulum/baseline_k1.pt \
        --tijepa_ckpt checkpoints/pendulum/tijepa.pt \
        --out results/pretty/pendulum_kill.png
"""
from __future__ import annotations

import argparse
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

import numpy as np
import torch

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from plot_style import COLORS, LABELS, setup_style  # noqa: E402

setup_style()
import matplotlib.pyplot as plt  # noqa: E402

from ti_jepa.envs.registry import get_env_spec, make_scaled_config  # noqa: E402
from ti_jepa.eval.env_utils import sample_qv1, build_context_generic  # noqa: E402
from ti_jepa.eval.common import load_baseline, load_tijepa, to_tensor_frames  # noqa: E402
from ti_jepa.eval.kill_experiment import (  # noqa: E402
    fit_position_probe,
    blind_rollout_baseline,
    blind_rollout_tijepa,
    rollout_ground_truth,
)
from ti_jepa.data import EpisodeBatch, cache_path, split_episodes  # noqa: E402

ENV_DISPLAY_NAME = {"inertia_ball": "InertiaBall", "pendulum": "Pendulum", "cartpole": "CartPole"}


def compute_kill_data(env, baseline_ckpt, baseline_k1_ckpt, tijepa_ckpt, baseline_rnn_ckpt=None,
                       horizon=15, n_frames=8, speed_lo=0.05, speed_hi=0.10,
                       display_image_size=200, seed=13):
    """Everything needed to render one kill-experiment panel (frame strips +
    decoded-position curves), factored out of main() so it can be reused by
    scripts that combine several environments into one figure (e.g.
    make_dual_kill_figure.py)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = get_env_spec(env)

    print(f"[{env}] baseline k=3")
    b3_enc, b3_pred, b3_args = load_baseline(baseline_ckpt, device)
    print(f"[{env}] baseline k=1")
    b1_enc, b1_pred, b1_args = load_baseline(baseline_k1_ckpt, device)
    print(f"[{env}] tijepa")
    t_enc, t_pred, t_args = load_tijepa(tijepa_ckpt, device)
    has_rnn = baseline_rnn_ckpt is not None
    if has_rnn:
        print(f"[{env}] baseline rnn")
        br_enc, br_pred, br_args = load_baseline(baseline_rnn_ckpt, device)

    image_size = b3_args.get("image_size", 64)
    d = np.load(cache_path(env, image_size))
    full_batch = EpisodeBatch(d["frames"], d["actions"], d["positions"], d["velocities"])
    _, held_out = split_episodes(full_batch, int(d["n_train"]))

    b3_probe = fit_position_probe("baseline", b3_enc, b3_args["k"], device, held_out)
    b1_probe = fit_position_probe("baseline", b1_enc, b1_args["k"], device, held_out)
    t_probe = fit_position_probe("tijepa", t_enc, t_args["k"], device, held_out)
    if has_rnn:
        br_probe = fit_position_probe("baseline", br_enc, br_args["k"], device, held_out)

    k3, k1, kt = b3_args["k"], b1_args["k"], t_args["k"]
    kr = br_args["k"] if has_rnn else 0
    action_dim = spec.action_dim

    model_cfg = make_scaled_config(env, image_size)
    rng = np.random.default_rng(seed)
    max_k = max(k3, kt, kr)
    q, v1 = pick_example(env, model_cfg, rng, max_k, horizon, (speed_lo, speed_hi))
    v2 = -v1

    model_env = spec.env_cls(model_cfg)
    ctx1_3 = build_context_generic(model_env, q, v1, k3, action_dim)
    ctx2_3 = build_context_generic(model_env, q, v2, k3, action_dim)
    ctx1_1, ctx2_1 = ctx1_3[-1:], ctx2_3[-1:]

    gt_pos1, _ = rollout_ground_truth(model_env, q, v1, horizon, action_dim)
    gt_pos2, _ = rollout_ground_truth(model_env, q, v2, horizon, action_dim)

    z1 = blind_rollout_baseline(b3_enc, b3_pred, ctx1_3, k3, horizon, device, action_dim)
    z2 = blind_rollout_baseline(b3_enc, b3_pred, ctx2_3, k3, horizon, device, action_dim)
    pos1_b3, pos2_b3 = b3_probe.predict(z1), b3_probe.predict(z2)

    z1_1 = blind_rollout_baseline(b1_enc, b1_pred, ctx1_1, 1, horizon, device, action_dim)
    z2_1 = blind_rollout_baseline(b1_enc, b1_pred, ctx2_1, 1, horizon, device, action_dim)
    pos1_b1, pos2_b1 = b1_probe.predict(z1_1), b1_probe.predict(z2_1)

    ctx1_t, ctx2_t = ctx1_3[-kt:], ctx2_3[-kt:]
    q1_t, _ = blind_rollout_tijepa(t_enc, t_pred, ctx1_t, kt, horizon, device, action_dim)
    q2_t, _ = blind_rollout_tijepa(t_enc, t_pred, ctx2_t, kt, horizon, device, action_dim)
    pos1_t, pos2_t = t_probe.predict(q1_t), t_probe.predict(q2_t)

    pos1_br = pos2_br = None
    if has_rnn:
        ctx1_r, ctx2_r = ctx1_3[-kr:], ctx2_3[-kr:]
        z1_r = blind_rollout_baseline(br_enc, br_pred, ctx1_r, kr, horizon, device, action_dim)
        z2_r = blind_rollout_baseline(br_enc, br_pred, ctx2_r, kr, horizon, device, action_dim)
        pos1_br, pos2_br = br_probe.predict(z1_r), br_probe.predict(z2_r)

    disp_cfg = make_scaled_config(env, display_image_size)
    disp_env = spec.env_cls(disp_cfg)
    zero_action = np.zeros(action_dim, dtype=np.float32)

    def render_rollout(v):
        disp_env.set_state(q, v)
        frames = [disp_env.render()]
        for _ in range(n_frames - 1):
            frames.append(disp_env.step(zero_action))
        return np.stack(frames)

    frames_plus = render_rollout(v1)
    frames_minus = render_rollout(v2)

    dim0 = 0
    steps = np.arange(len(gt_pos1))
    series = [
        (f"{LABELS['gt']} ($+v$)", COLORS["gt"], "-", 2.4, gt_pos1[:, dim0]),
        (f"{LABELS['gt']} ($-v$)", COLORS["gt"], "--", 2.4, gt_pos2[:, dim0]),
        (f"{LABELS['baseline_k1']} ($+v$)", COLORS["baseline_k1"], "-", 1.6, pos1_b1[:, dim0]),
        (f"{LABELS['baseline_k1']} ($-v$)", COLORS["baseline_k1"], "--", 1.6, pos2_b1[:, dim0]),
        (f"{LABELS['baseline_k3']} ($+v$)", COLORS["baseline_k3"], "-", 1.6, pos1_b3[:, dim0]),
        (f"{LABELS['baseline_k3']} ($-v$)", COLORS["baseline_k3"], "--", 1.6, pos2_b3[:, dim0]),
    ]
    if has_rnn:
        series += [
            (f"{LABELS['baseline_rnn']} ($+v$)", COLORS["baseline_rnn"], "-", 1.6, pos1_br[:, dim0]),
            (f"{LABELS['baseline_rnn']} ($-v$)", COLORS["baseline_rnn"], "--", 1.6, pos2_br[:, dim0]),
        ]
    series += [
        (f"{LABELS['tijepa']} ($+v$)", COLORS["tijepa"], "-", 2.2, pos1_t[:, dim0]),
        (f"{LABELS['tijepa']} ($-v$)", COLORS["tijepa"], "--", 2.2, pos2_t[:, dim0]),
    ]

    return {
        "env": env,
        "env_display": ENV_DISPLAY_NAME.get(env, env),
        "frames_plus": frames_plus,
        "frames_minus": frames_minus,
        "steps": steps,
        "series": series,
        "has_rnn": has_rnn,
    }


def render_kill_panel(fig, outer_spec, data, n_frames, panel_title=None, show_legend=False,
                       frame_title_fontsize=11, label_fontsize=12.5, curve_fontsize=12,
                       panel_title_fontsize=13.5):
    """Draw one frame-strip(+curve) panel into a subplotspec cell of `fig`
    (which may itself be a matplotlib SubFigure). Returns the curve axis
    (useful for pulling shared legend handles)."""
    inner = outer_spec.subgridspec(3, n_frames, height_ratios=[1, 1, 2.3], hspace=0.28, wspace=0.05)
    top_axes, bottom_axes = [], []
    for i in range(n_frames):
        ax = fig.add_subplot(inner[0, i])
        ax.imshow(data["frames_plus"][i], aspect="auto"); ax.axis("off")
        ax.set_title(f"$t={i}$", fontsize=frame_title_fontsize, color="#444444")
        top_axes.append(ax)
        ax2 = fig.add_subplot(inner[1, i])
        ax2.imshow(data["frames_minus"][i], aspect="auto"); ax2.axis("off")
        bottom_axes.append(ax2)
    top_axes[0].text(-0.32, 0.5, "$v=+v$", transform=top_axes[0].transAxes, fontsize=label_fontsize,
                      color=COLORS["tijepa"], ha="right", va="center")
    bottom_axes[0].text(-0.32, 0.5, "$v=-v$", transform=bottom_axes[0].transAxes, fontsize=label_fontsize,
                         color=COLORS["baseline_k1"], ha="right", va="center")
    if panel_title:
        top_axes[n_frames // 2].text(
            0.0, 1.4, panel_title, transform=top_axes[n_frames // 2].transAxes,
            fontsize=panel_title_fontsize, fontweight="bold", ha="center", va="bottom",
        )

    ax = fig.add_subplot(inner[2, :])
    for label, color, ls, lw, y in data["series"]:
        ax.plot(data["steps"], y, color=color, ls=ls, lw=lw, label=label)
    ax.set_xlabel("model step (blind rollout, $a=0$)", fontsize=curve_fontsize)
    ax.set_ylabel("decoded position (probe)", fontsize=curve_fontsize)
    ax.tick_params(labelsize=curve_fontsize - 1.5)
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(ncol=3 if data["has_rnn"] else 2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), fontsize=curve_fontsize - 1.5)
    return ax


def pick_example(env_name, cfg, rng, k_ctx, horizon, speed_range, n_tries=12):
    """Sample several candidate (q, v1) pairs and keep the one with the
    largest ground-truth branch separation, purely so the illustrative
    figure shows a visually convincing divergence (does not affect any
    reported statistic, which is always an average over many pairs)."""
    spec = get_env_spec(env_name)
    env = spec.env_cls(cfg)
    best = None
    for _ in range(n_tries):
        q, v1 = sample_qv1(env_name, rng, cfg, k_ctx, horizon, speed_range)
        gt1, _ = rollout_ground_truth(env, q, v1, horizon, spec.action_dim)
        gt2, _ = rollout_ground_truth(env, q, -v1, horizon, spec.action_dim)
        sep = float(np.linalg.norm(gt1[-1] - gt2[-1]))
        if best is None or sep > best[0]:
            best = (sep, q, v1)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["inertia_ball", "pendulum", "cartpole"], required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--baseline_k1_ckpt", required=True)
    ap.add_argument("--tijepa_ckpt", required=True)
    ap.add_argument("--baseline_rnn_ckpt", default=None)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--speed_lo", type=float, default=0.05)
    ap.add_argument("--speed_hi", type=float, default=0.10)
    ap.add_argument("--display_image_size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--title_prefix", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = compute_kill_data(
        args.env, args.baseline_ckpt, args.baseline_k1_ckpt, args.tijepa_ckpt,
        baseline_rnn_ckpt=args.baseline_rnn_ckpt, horizon=args.horizon, n_frames=args.n_frames,
        speed_lo=args.speed_lo, speed_hi=args.speed_hi, display_image_size=args.display_image_size,
        seed=args.seed,
    )

    # ================= figure =================
    n = args.n_frames
    fig = plt.figure(figsize=(1.05 * n, 6.9))
    outer = fig.add_gridspec(1, 1, left=0.09, right=0.995, top=0.94, bottom=0.09)[0, 0]
    render_kill_panel(fig, outer, data, n, show_legend=True)

    label = args.title_prefix or ENV_DISPLAY_NAME.get(args.env, args.env)
    fig.suptitle(f"{label}: same configuration, opposite velocity  (Protocol B kill experiment)", fontsize=14.5, y=1.01)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.1)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
