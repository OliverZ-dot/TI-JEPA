"""Render actual frame-by-frame image strips of the "same configuration,
opposite velocity" construction used throughout the kill experiment
(Protocol B). This is the purely-visual companion to the quantitative
branch-separation numbers already in the results log: it shows, frame by frame,
that the two branches start pixel-identical and then visibly diverge under
the real physics, which is the whole empirical premise the kill experiment
depends on.

Two families of helpers:
  - gt_branch_frames(env_name, ...): InertiaBall / Pendulum / CartPole,
    rendered at a high display resolution (independent of the resolution
    any checkpoint was trained at -- this is illustration only, no model
    is involved).
  - reacher_branch_frames(...): real dm_control Reacher, using the same
    live simulator the official-scale experiments trained on.

Usage (examples):
    python3 scripts/make_branch_montages.py --demo
"""
from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

import numpy as np

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from plot_style import COLORS, setup_style  # noqa: E402

setup_style()
import matplotlib.pyplot as plt  # noqa: E402

from ti_jepa.envs.registry import get_env_spec, make_scaled_config  # noqa: E402
from ti_jepa.eval.env_utils import sample_qv1  # noqa: E402


ENV_DISPLAY_NAME = {
    "inertia_ball": "InertiaBall",
    "pendulum": "Pendulum",
    "cartpole": "CartPole",
}


def gt_branch_frames(env_name, seed, n_frames=8, image_size=200, speed_range=(0.05, 0.09)):
    """Returns (frames_plus, frames_minus, q, v1): each frames_* is
    (n_frames, image_size, image_size, 3) uint8, the real open-loop rollout
    under a=0 starting from the same q with velocity +v1 / -v1."""
    spec = get_env_spec(env_name)
    cfg = make_scaled_config(env_name, image_size)
    env = spec.env_cls(cfg)
    rng = np.random.default_rng(seed)
    q, v1 = sample_qv1(env_name, rng, cfg, k=1, horizon=n_frames, speed_range=speed_range)
    action_dim = spec.action_dim
    zero_action = np.zeros(action_dim, dtype=np.float32)

    def rollout(v):
        env.set_state(q, v)
        frames = [env.render()]
        for _ in range(n_frames - 1):
            frames.append(env.step(zero_action))
        return np.stack(frames)

    frames_plus = rollout(v1)
    frames_minus = rollout(-v1)
    return frames_plus, frames_minus, q, v1


def reacher_branch_frames(seed, n_frames=8, model_step_steps=3):
    """Real dm_control Reacher, same construction. Returns
    (frames_plus, frames_minus, q0, v)."""
    sys.path.insert(0, os.path.join(_REPO_ROOT, "real_pusht"))
    if os.environ.get("SWM_SRC"):
        sys.path.insert(0, os.environ["SWM_SRC"])
    os.environ.setdefault("MUJOCO_GL", "egl")
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper
    from reacher_kill_experiment_official import sample_config

    env = ReacherDMControlWrapper(task="hard", seed=seed)
    rng = np.random.default_rng(seed)
    q0, v = sample_config(rng)
    zero_action = np.zeros(2, dtype=np.float32)

    def rollout(v_signed):
        state = np.concatenate([q0, v_signed])
        env.reset(seed=seed, options={"state": state})
        frames = [env.render()]
        for _ in range(n_frames - 1):
            for _ in range(model_step_steps):
                env.step(zero_action)
            frames.append(env.render())
        return np.stack(frames)

    frames_plus = rollout(v)
    frames_minus = rollout(-v)
    return frames_plus, frames_minus, q0, v


def plot_branch_strip(ax_row_plus, ax_row_minus, frames_plus, frames_minus, col_labels=True):
    """Draw two rows of a montage onto pre-created axes lists (same length)."""
    n = len(frames_plus)
    for i in range(n):
        ax_row_plus[i].imshow(frames_plus[i], aspect="auto")
        ax_row_plus[i].axis("off")
        ax_row_minus[i].imshow(frames_minus[i], aspect="auto")
        ax_row_minus[i].axis("off")
        if col_labels:
            ax_row_plus[i].set_title(f"$t={i}$", fontsize=13, color="#444444")


def make_single_montage_figure(env_name, out_path, seed=11, n_frames=8, image_size=220,
                                speed_range=(0.05, 0.09), env_label=None):
    """A clean, publication-quality standalone figure: two rows (v=+v on top,
    v=-v below) of n_frames real rollout frames, for one representative
    example. Pixel-identical first column, visibly diverging afterward."""
    if env_name == "reacher":
        frames_plus, frames_minus, q, v = reacher_branch_frames(seed, n_frames=n_frames)
        title = "Real dm_control Reacher"
    else:
        frames_plus, frames_minus, q, v = gt_branch_frames(
            env_name, seed, n_frames=n_frames, image_size=image_size, speed_range=speed_range)
        title = env_label or ENV_DISPLAY_NAME.get(env_name, env_name)

    fig, axes = plt.subplots(2, n_frames, figsize=(1.42 * n_frames, 3.1))
    plot_branch_strip(axes[0], axes[1], frames_plus, frames_minus)
    axes[0][0].set_ylabel("$v=+v$", fontsize=10, color=COLORS["tijepa"], rotation=0,
                           labelpad=32, va="center")
    axes[1][0].set_ylabel("$v=-v$", fontsize=10, color=COLORS["baseline_k1"], rotation=0,
                           labelpad=32, va="center")
    axes[0][0].axis("on"); axes[0][0].set_xticks([]); axes[0][0].set_yticks([])
    for s in axes[0][0].spines.values():
        s.set_visible(False)
    axes[1][0].axis("on"); axes[1][0].set_xticks([]); axes[1][0].set_yticks([])
    for s in axes[1][0].spines.values():
        s.set_visible(False)

    img_diff = float(np.abs(frames_plus[0].astype(np.float32) - frames_minus[0].astype(np.float32)).mean())
    fig.suptitle(
        f"{title}: same configuration $q_0$, opposite velocity  "
        f"(frame 0 pixel difference $={img_diff:.4f}$)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out_path)


def make_grid_figure(env_name, out_path, seeds, n_frames=8, image_size=170,
                      speed_range=(0.05, 0.09), env_label=None):
    """Appendix-scale figure: stack len(seeds) independent examples, each as
    a (v=+v / v=-v) pair of rows, n_frames columns -> 2*len(seeds) rows x
    n_frames columns total."""
    n_examples = len(seeds)
    fig, axes = plt.subplots(2 * n_examples, n_frames,
                              figsize=(1.12 * n_frames, 1.18 * n_examples + 0.16 * n_examples),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    if n_examples == 1:
        axes = axes.reshape(2, n_frames)

    for ex_i, seed in enumerate(seeds):
        if env_name == "reacher":
            frames_plus, frames_minus, q, v = reacher_branch_frames(seed, n_frames=n_frames)
        else:
            frames_plus, frames_minus, q, v = gt_branch_frames(
                env_name, seed, n_frames=n_frames, image_size=image_size, speed_range=speed_range)
        row_plus = axes[2 * ex_i]
        row_minus = axes[2 * ex_i + 1]
        plot_branch_strip(row_plus, row_minus, frames_plus, frames_minus, col_labels=(ex_i == 0))
        row_plus[0].axis("on"); row_plus[0].set_xticks([]); row_plus[0].set_yticks([])
        for s in row_plus[0].spines.values():
            s.set_visible(False)
        row_plus[0].set_ylabel(f"ex.{ex_i+1}\n$v=+v$", fontsize=9.5, color=COLORS["tijepa"],
                                rotation=0, labelpad=28, va="center")
        row_minus[0].axis("on"); row_minus[0].set_xticks([]); row_minus[0].set_yticks([])
        for s in row_minus[0].spines.values():
            s.set_visible(False)
        row_minus[0].set_ylabel(f"ex.{ex_i+1}\n$v=-v$", fontsize=9.5, color=COLORS["baseline_k1"],
                                 rotation=0, labelpad=28, va="center")

    label = env_label or ENV_DISPLAY_NAME.get(env_name, env_name)
    fig.suptitle(f"{label}: {n_examples} independent same-$q_0$, opposite-velocity examples", fontsize=13.5, y=1.05)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.01,
                         wspace=0.04, hspace=0.08)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved ->", out_path)


def make_multi_env_gallery(out_path, envs, seeds, n_frames=9, image_size=150,
                            speed_range=(0.05, 0.09), cell_size=0.85):
    """Main-text-scale figure: one representative (v=+v / v=-v) example per
    environment, stacked so the whole gallery is len(envs) environments x
    2 rows x n_frames columns -- a compact multi-environment grid rather
    than multiple examples of a single environment (that's make_grid_figure,
    used in the appendix instead).

    All source frames are natively square, so the grid geometry is chosen
    (figsize computed from `cell_size`, in inches) so that every cell is
    also square: with `aspect="auto"` imshow, a non-square cell would
    stretch/squash the square frames, which is exactly the distortion this
    is designed to avoid."""
    n_envs = len(envs)
    n_rows = 2 * n_envs
    left, right, bottom = 0.125, 0.995, 0.012
    top = 0.9 - 0.006 * n_rows
    wspace, hspace = 0.06, 0.08
    avail_w = cell_size * (n_frames + (n_frames - 1) * wspace)
    avail_h = cell_size * (n_rows + (n_rows - 1) * hspace)
    figsize = (avail_w / (right - left), avail_h / (top - bottom))

    fig, axes = plt.subplots(n_rows, n_frames, figsize=figsize,
                              gridspec_kw={"wspace": wspace, "hspace": hspace})
    if n_envs == 1:
        axes = axes.reshape(2, n_frames)

    for e_i, (env_name, seed) in enumerate(zip(envs, seeds)):
        if env_name == "reacher":
            frames_plus, frames_minus, q, v = reacher_branch_frames(seed, n_frames=n_frames)
        else:
            frames_plus, frames_minus, q, v = gt_branch_frames(
                env_name, seed, n_frames=n_frames, image_size=image_size, speed_range=speed_range)
        row_plus = axes[2 * e_i]
        row_minus = axes[2 * e_i + 1]
        plot_branch_strip(row_plus, row_minus, frames_plus, frames_minus, col_labels=(e_i == 0))
        row_plus[0].axis("on"); row_plus[0].set_xticks([]); row_plus[0].set_yticks([])
        for s in row_plus[0].spines.values():
            s.set_visible(False)
        label = ENV_DISPLAY_NAME.get(env_name, "Real Reacher" if env_name == "reacher" else env_name)
        row_plus[0].set_ylabel(f"{label}\n$v{{=}}{{+}}v$", fontsize=12, color=COLORS["tijepa"],
                                rotation=0, labelpad=44, va="center", ha="right")
        row_minus[0].axis("on"); row_minus[0].set_xticks([]); row_minus[0].set_yticks([])
        for s in row_minus[0].spines.values():
            s.set_visible(False)
        row_minus[0].set_ylabel(f"{label}\n$v{{=}}{{-}}v$", fontsize=12, color=COLORS["baseline_k1"],
                                 rotation=0, labelpad=44, va="center", ha="right")

    fig.suptitle(
        "Four environments, same recipe: identical configuration $q_0$, opposite velocity, real rollout frames",
        fontsize=15, y=top + 0.05,
    )
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                         wspace=wspace, hspace=hspace)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved ->", out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        make_single_montage_figure("pendulum", "/tmp/demo_pendulum.png", seed=11, n_frames=8)
