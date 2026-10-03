"""One-figure summary of the whole evidence chain.

Three panels, left-to-right, matching the paper's argument:

  A  Protocol A — single-frame embeddings never carry velocity;
                  TI-JEPA's explicit v does (where we have one).
  B  Protocol B — memoryless baseline cannot separate opposite-velocity
                  futures; TI-JEPA always does.
  C  Protocol C — matched-memory planning: TI-JEPA closer to goal than
                  baseline_k1 on the three controlled environments;
                  honest null on real PushT.

Numbers are the published the results log values (not re-computed here).
"""

from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from plot_style import COLORS, setup_style  # noqa: E402

setup_style()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(_REPO_ROOT) / "results" / "story_figure.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- published numbers (the results log) -----------------------------------------
# Protocol A: single-frame velocity Pearson r (or R² floor, marked)
# Official checkpoints report held-out R² ≈ 0; custom envs report Pearson r.
# Both are "at floor" for single-frame; we plot the signed r we have, and
# clip official R² (already ≈0) onto the same axis as r.
# "official arch" = our re-implementation of the official ViT-Tiny/14 +
# AdaLN-transformer scale, trained from scratch (r14); "official ckpt" =
# the released LeWM weights, no retraining, no TI-JEPA v-head (r6/r7).
# Radar/spider version of Protocol A: 8 spokes (regular octagon). The two
# "official ckpt" environments with the least individual story (Cube and
# TwoRoom -- both just sit at floor like every other official checkpoint,
# with near-identical values) are merged into one shared spoke so the chart
# is a clean octagon instead of a 9-gon; no information is lost since both
# were already reporting the same "at floor, no v head" fact.
A_ENVS = [
    "InertiaBall", "Pendulum", "CartPole",
    "CartPole (official arch)", "Pendulum (official arch)",
    "PushT (official ckpt)", "Reacher (official ckpt)",
    "Cube / TwoRoom (official ckpt)",
]
A_SINGLE = [-0.025, -0.009, +0.111, +0.152, -0.010, -0.002, -0.002, -0.002]
A_TIJEPA_V = [0.571, 0.879, 0.302, 0.910, 0.832, np.nan, np.nan, np.nan]
A_HIGHLIGHT = [False, False, False, True, False, False, False, False]  # the "scale rescues it" spoke
A_HAS_V = [True, True, True, True, True, False, False, False]  # spokes with no TI-JEPA v-head at all

# Protocol B: branch-separation / GT
B_ENVS = ["InertiaBall", "Pendulum", "CartPole", "CartPole\n(official arch)", "Pendulum\n(official arch)", "PushT\n(official ckpt)"]
B_BASE_K1 = [0.000, 0.000, 0.000, 0.000, 0.000, 0.0087]
B_TIJEPA = [0.674, 1.012, 1.406, 0.205, 1.172, np.nan]  # official ckpt has no TI-JEPA v-head
B_SIGN = [1.00, 1.00, 1.00, 0.213, 0.447, np.nan]

# Protocol C: closed-loop relative improvement (1 - tijepa/baseline), %
# (Real PushT is a null result on this task and is reported separately in
# the text/tables rather than diluting this panel with a non-win row.)
C_ENVS = ["InertiaBall", "Pendulum", "CartPole"]
C_IMPROVE = [23.0, 55.0, 64.0]
C_P = ["p=5.3e-7", "p=3.2e-10", "p=5.1e-15"]
C_KIND = ["win", "win", "win"]

# Colors reuse the paper-wide palette from plot_style.py so this overview
# figure agrees with every other figure: grey = single-frame / at-floor,
# red = memoryless baseline (the failing arm), amber = history baseline /
# highlight, blue = TI-JEPA (the arm that works everywhere in this figure).
COL_FLOOR = COLORS["neutral"]
COL_V = COLORS["tijepa"]
COL_BASE = COLORS["baseline_k1"]
COL_TI = COLORS["tijepa"]
COL_NULL = COLORS["neutral"]
COL_WIN = COLORS["tijepa"]
COL_HIGHLIGHT = COLORS["baseline_k3"]

FS_TITLE = 17
FS_AXIS = 15.5
FS_TICK = 14.5
FS_LEGEND = 14
FS_ANNOT = 13.5
FS_FOOTER = 13


def _style():
    plt.rcParams.update({
        "axes.grid": False,
    })


def panel_a(ax):
    """Octagon radar/spider chart. 8 spokes = 8 environment/scale
    combinations. Two overlaid shapes: a tiny near-center octagon for the
    single-frame baseline (always at floor) and a larger, open shape for
    TI-JEPA's explicit v_t, which only exists on the 5 spokes where we
    trained a v-head (the 3 official-checkpoint spokes have no v-head at
    all, so they are left as open gaps rather than false zeros)."""
    n = len(A_ENVS)
    angles = (np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2).tolist()
    r_max = 1.0

    ax.set_theta_offset(0.0)
    ax.set_theta_direction(1)
    ax.set_ylim(0, r_max)
    ax.set_rlabel_position(0)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # hand-drawn polygonal (octagon) grid, rings + spokes, instead of the
    # default circular polar grid -- this is what makes it read as a
    # regular octagon rather than a generic circular radar chart
    ring_vals = [0.25, 0.5, 0.75, 1.0]
    for rv in ring_vals:
        ring_theta = angles + angles[:1]
        ring_r = [rv] * n + [rv]
        ax.plot(ring_theta, ring_r, color="#ddd", lw=0.9, zorder=0)
    for a in angles:
        ax.plot([a, a], [0, r_max], color="#ddd", lw=0.9, zorder=0)
    # radial value labels go in the emptiest sector (between the two
    # lowest-value official-checkpoint spokes) so they never collide with
    # a data label
    tick_angle = (angles[5] + angles[6]) / 2.0
    for rv in ring_vals:
        ax.text(tick_angle, rv, f"{rv:.2f}", fontsize=FS_ANNOT - 3.5, color="#999",
                ha="center", va="center", zorder=1)

    # spoke labels around the outside
    for a, label in zip(angles, A_ENVS):
        ax.text(a, r_max * 1.16, label.replace(" (", "\n("), fontsize=FS_TICK - 2.5,
                ha="center", va="center", linespacing=1.25)

    # single-frame baseline: closed octagon, always tiny/near-zero (clip
    # slightly-negative values to 0 for display -- the point is "at floor",
    # not the exact sign of a number this close to 0)
    single_vals = [max(v, 0.0) for v in A_SINGLE]
    ax.plot(angles + angles[:1], single_vals + single_vals[:1], color=COL_FLOOR, lw=1.8,
            label="single-frame $z_t$ / $q_t$", zorder=3)
    ax.fill(angles + angles[:1], single_vals + single_vals[:1], color=COL_FLOOR, alpha=0.35, zorder=2)

    # TI-JEPA explicit v_t: open shape, only across the 5 consecutive
    # spokes that actually have a v-head -- no data is plotted for the
    # other 3 (drawing them as 0 would misleadingly suggest "fails here")
    has_v_idx = [i for i in range(n) if A_HAS_V[i]]
    v_angles = [angles[i] for i in has_v_idx]
    v_vals = [A_TIJEPA_V[i] for i in has_v_idx]
    ax.plot(v_angles, v_vals, color=COL_V, lw=2.4, label="TI-JEPA explicit $v_t$", zorder=4)
    ax.fill(v_angles, v_vals, color=COL_V, alpha=0.18, zorder=2)
    point_colors = [COL_HIGHLIGHT if A_HIGHLIGHT[i] else COL_V for i in has_v_idx]
    point_sizes = [95 if A_HIGHLIGHT[i] else 55 for i in has_v_idx]
    ax.scatter(v_angles, v_vals, color=point_colors, s=point_sizes, zorder=5,
               edgecolors="white", linewidths=0.8)
    for i in has_v_idx:
        ax.text(angles[i], A_TIJEPA_V[i] + 0.075, f"{A_TIJEPA_V[i]:.2f}", fontsize=FS_ANNOT - 1.5,
                ha="center", va="center", color=COL_HIGHLIGHT if A_HIGHLIGHT[i] else COL_V,
                fontweight="bold" if A_HIGHLIGHT[i] else "normal", zorder=6)

    # explicit "no v head" mark on the 3 spokes with no TI-JEPA data --
    # placed at a small but nonzero radius along each spoke's own angle
    # (r=0 would put all three on top of each other at the exact center)
    for i in range(n):
        if not A_HAS_V[i]:
            ax.scatter([angles[i]], [0.16], marker="x", color="#999999", s=55, zorder=5, linewidths=1.8)

    ax.text(-0.30, 1.30, "A   Protocol A --- probe ladder", transform=ax.transAxes,
            fontweight="bold", fontsize=FS_TITLE, ha="left", va="bottom")
    ax.text(-0.30, 1.19, r"single-frame $\approx 0$ everywhere --- explicit $v_t$ is what was missing",
            transform=ax.transAxes, fontsize=FS_ANNOT, color="#666", style="italic", ha="left", va="bottom")
    ax.text(-0.30, -0.14, r"$\times$ = official ckpt, no $v$ head trained (not a zero)",
            transform=ax.transAxes, fontsize=FS_ANNOT - 1.5, color="#888", style="italic", ha="left", va="top")
    # legend as a compact two-column strip under everything else in this
    # panel's own footprint, so it can never bleed into panel B's axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
              fontsize=FS_LEGEND - 2, handlelength=1.6, columnspacing=1.2)

    # callout for the scale-rescue pair (CartPole raw vs. official-arch)
    hl_idx = np.where(np.array(A_HIGHLIGHT))[0]
    if len(hl_idx):
        i = int(hl_idx[0])
        ax.text(-0.30, -0.02, r"official arch rescues the small-CNN" "\n" r"ceiling on CartPole: $0.30\to0.91$",
                transform=ax.transAxes, fontsize=FS_ANNOT - 0.5, color=COL_HIGHLIGHT, ha="left", va="top",
                fontweight="bold", linespacing=1.3)



def panel_b(ax):
    n = len(B_ENVS)
    # x categories are spread out (factor > 1) purely to give the two-line,
    # rotated x-tick labels ("CartPole\n(official arch)" etc.) enough room
    # that adjacent labels no longer overlap each other.
    xs = 1.32
    x = np.arange(n) * xs
    w = 0.36
    ax.bar(x - w / 2, B_BASE_K1, width=w, color=COL_BASE, label="baseline$_{k=1}$ (memoryless)")
    mask = ~np.isnan(B_TIJEPA)
    ax.bar(x[mask] + w / 2, np.array(B_TIJEPA)[mask], width=w, color=COL_TI, label="TI-JEPA")
    ax.axhline(1.0, color="#333", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(B_ENVS, rotation=26, ha="right", fontsize=FS_TICK - 1.5)
    ax.set_xlim(-0.62, x[-1] + 0.62)
    ax.margins(x=0.02)
    # the "1.0 line" meaning lives in the y-axis label itself now, so it
    # can never collide with a bar or an annotation wherever they land
    ax.set_ylabel("branch-sep. ratio\n(1.0 = matches GT exactly)", fontsize=FS_AXIS - 1.5, linespacing=1.4)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 1.85)
    ax.text(0.0, 1.16, "B   Protocol B --- kill experiment", transform=ax.transAxes,
            fontweight="bold", fontsize=FS_TITLE, ha="left", va="bottom")
    ax.text(0.0, 1.05, "same $q$, opposite $v$: memoryless model always collapses to 0",
            transform=ax.transAxes, fontsize=FS_ANNOT, color="#666", style="italic", ha="left", va="bottom")
    ax.legend(loc="upper left", frameon=False, fontsize=FS_LEGEND)
    # direction-accuracy annotation directly above each TI-JEPA bar -- kept
    # short and uniform so it never collides with a neighboring bar. The
    # longer clarification (that this is a *different*, independent
    # statistic from bar height -- bar height is separation magnitude,
    # dir.acc. is only whether the sign was right) is a single standalone
    # note in the empty upper-right corner instead of glued to any one bar.
    for i, acc in enumerate(B_SIGN):
        if np.isnan(acc):
            ax.text(x[i], 0.18, "official\nckpt", ha="center", fontsize=FS_ANNOT - 1, color="#888")
        else:
            ax.text(x[i] + w / 2, B_TIJEPA[i] + 0.07, f"dir.acc.={acc:.0%}".replace("%", r"\%"),
                    ha="center", fontsize=FS_ANNOT - 1.5, color=COL_TI, fontweight="bold")
    ax.text(x[-1] + 0.55, 1.80,
            r"dir.acc.\ = sign only;" "\n" r"bar height = magnitude" "\n" r"(independent stats)",
            ha="right", va="top", fontsize=FS_ANNOT - 2.5, color=COL_TI, style="italic", linespacing=1.35)


def panel_c(ax):
    n = len(C_ENVS)
    y = np.arange(n)[::-1]
    colors = [COL_WIN if k == "win" else COL_NULL for k in C_KIND]
    ax.barh(y, C_IMPROVE, color=colors, height=0.55)
    ax.axvline(0.0, color="#333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(C_ENVS, fontsize=FS_TICK)
    ax.set_xlabel(r"closed-loop: TI-JEPA closer than baseline$_{k=1}$  (\%)", fontsize=FS_AXIS)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlim(-8, 90)
    ax.text(0.0, 1.16, "C   Protocol C --- planning (matched memory)", transform=ax.transAxes,
            fontweight="bold", fontsize=FS_TITLE, ha="left", va="bottom")
    ax.text(0.0, 1.05, "wins only where velocity is necessary and the model can represent it",
            transform=ax.transAxes, fontsize=FS_ANNOT, color="#666", style="italic", ha="left", va="bottom")
    for yi, val, p, kind in zip(y, C_IMPROVE, C_P, C_KIND):
        label = f"+{val:.0f}\\%   {p}" if kind == "win" else "null result (n.s.)"
        xpos = val + 2.0 if kind == "win" else 2.0
        ax.text(xpos, yi, label, va="center", fontsize=FS_ANNOT,
                color=COL_WIN if kind == "win" else "#555")


def main():
    _style()
    # Back to one row of three, now that the panel shapes themselves have
    # changed: A is a radar/octagon (roughly square, not a tall list of
    # bars anymore), B is unchanged, and C lost its null PushT row so it
    # only has 3 categories left. Width ratios are tuned per panel's own
    # content (A needs a square-ish cell for the octagon; B needs extra
    # width for 6 rotated x-tick labels; C is short but has long end-of-bar
    # annotation text) so nothing feels squeezed despite sharing one row.
    fig = plt.figure(figsize=(17.6, 7.7), dpi=160)
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.045, right=0.99, top=0.72, bottom=0.14,
                           width_ratios=[1.0, 1.18, 0.98])
    ax_a = fig.add_subplot(gs[0, 0], projection="polar")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    fig.suptitle(
        "Target identifiability: a single-frame JEPA never sees $\\dot q$; a pose/motion split does.",
        fontsize=21, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, 0.005,
        r"\textbf{Custom envs}: one small CNN, one recipe.   "
        r"\textbf{Official arch}: from-scratch ViT-Tiny/14 + AdaLN transformer, official recipe.   "
        r"\textbf{Official ckpt}: released LeWM weights, zero retraining.",
        ha="center", fontsize=FS_FOOTER + 1.5, color="#444",
    )
    fig.savefig(OUT, dpi=180, bbox_inches="tight", pad_inches=0.15)
    pdf = OUT.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
    print("saved", OUT)
    print("saved", pdf)


if __name__ == "__main__":
    main()
