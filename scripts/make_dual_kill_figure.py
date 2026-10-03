"""Main-text kill-experiment figure showing TWO environments side by side
in one row (Pendulum | CartPole), each with its own real frame-strip pair
plus decoded blind-rollout curve, sharing one legend. This packs two of the
three physically grounded environments into the same main-text real estate
that used to hold only Pendulum.

Usage:
    python3 scripts/make_dual_kill_figure.py \
        --out results/pretty/pendulum_cartpole_kill_dual.png
"""
from __future__ import annotations

import argparse
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from plot_style import COLORS, setup_style  # noqa: E402

setup_style()
import matplotlib.pyplot as plt  # noqa: E402

from make_pretty_kill_figure import compute_kill_data, render_kill_panel  # noqa: E402

DEFAULT_ENVS = [
    {
        "env": "pendulum",
        "display": "Pendulum",
        "baseline_ckpt": "checkpoints/pendulum/baseline.pt",
        "baseline_k1_ckpt": "checkpoints/pendulum/baseline_k1.pt",
        "tijepa_ckpt": "checkpoints/pendulum/tijepa.pt",
        "seed": 13,
    },
    {
        "env": "cartpole",
        "display": "CartPole",
        "baseline_ckpt": "checkpoints/cartpole/baseline.pt",
        "baseline_k1_ckpt": "checkpoints/cartpole/baseline_k1.pt",
        "tijepa_ckpt": "checkpoints/cartpole/tijepa.pt",
        "seed": 9,
        "speed_lo": 0.08,
        "speed_hi": 0.14,
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_frames", type=int, default=7)
    ap.add_argument("--out", default="results/pretty/pendulum_cartpole_kill_dual.png")
    args = ap.parse_args()

    n = args.n_frames
    panels = []
    for cfg in DEFAULT_ENVS:
        data = compute_kill_data(
            cfg["env"], cfg["baseline_ckpt"], cfg["baseline_k1_ckpt"], cfg["tijepa_ckpt"],
            n_frames=n, seed=cfg.get("seed", 13),
            speed_lo=cfg.get("speed_lo", 0.05), speed_hi=cfg.get("speed_hi", 0.10),
        )
        panels.append((cfg["display"], data))

    fig = plt.figure(figsize=(2.05 * n, 7.3))
    outer_gs = fig.add_gridspec(2, 2, height_ratios=[10, 0.85], wspace=0.09, hspace=0.55,
                                 left=0.07, right=0.99, top=0.82, bottom=0.03)
    curve_axes = []
    for col, (display, data) in zip(range(2), panels):
        ax = render_kill_panel(fig, outer_gs[0, col], data, n, panel_title=display, show_legend=False)
        curve_axes.append(ax)

    handles, labels = curve_axes[0].get_legend_handles_labels()
    ax_legend = fig.add_subplot(outer_gs[1, :])
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, loc="center", ncol=6, fontsize=12.5, frameon=False,
                      bbox_to_anchor=(0.5, 0.5))

    fig.suptitle(
        "Same recipe, two coupling strengths: same configuration, opposite velocity  (Protocol B kill experiment)",
        fontsize=15.5, y=1.06,
    )
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.15)
    print("saved ->", out)


if __name__ == "__main__":
    main()
