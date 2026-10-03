"""Two extra paper figures built purely from numbers already reported in
the results log / the paper draft (no new experiments, pure plotting):

  1. results/planning_bar.png   -- Protocol C stop-at-goal distance-to-goal,
     oracle vs. matched-memory baseline_k1 vs. TI-JEPA, open- and closed-loop,
     across the three physically-grounded environments.
  2. results/scale_bar.png      -- explicit v_t probe r: small-CNN vs.
     official ViT-Tiny/AdaLN scale, CartPole and Pendulum, showing the
     0.30 -> 0.91 jump.

Uses the shared paper style (SciencePlots + our own color palette) from
scripts/plot_style.py so these two bar charts match every other figure in
the paper instead of using matplotlib's raw tab10 defaults.
"""

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
from plot_style import COLORS, setup_style  # noqa: E402

setup_style()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------- figure 1
envs = ["InertiaBall", "Pendulum", "CartPole"]
oracle_ol = [0.009, 0.112, 0.046]
base_ol = [0.321, 1.390, 0.904]
tijepa_ol = [0.271, 0.434, 0.624]
oracle_cl = [0.068, 0.211, 0.116]
base_cl = [0.230, 0.647, 0.490]
tijepa_cl = [0.178, 0.292, 0.178]
pct_cl = [23, 55, 64]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=False)
x = np.arange(len(envs))
w = 0.26
for ax, oracle, base, tij, title, pct in [
    (axes[0], oracle_ol, base_ol, tijepa_ol, "Open-loop", None),
    (axes[1], oracle_cl, base_cl, tijepa_cl, "Closed-loop MPC", pct_cl),
]:
    ax.bar(x - w, oracle, w, label="oracle (true velocity)", color=COLORS["oracle"])
    ax.bar(x, base, w, label=r"baseline$_{k=1}$ (memoryless)", color=COLORS["baseline_k1"])
    ax.bar(x + w, tij, w, label="TI-JEPA", color=COLORS["tijepa"])
    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("final distance to goal (lower better)", fontsize=8.5)
    if pct is not None:
        for xi, p, bv, tv in zip(x, pct, base, tij):
            ax.annotate(rf"$-${p}\%", xy=(xi + w / 2, (bv + tv) / 2), xytext=(xi + w + 0.32, (bv + tv) / 2 + 0.02),
                        fontsize=8.5, color=COLORS["tijepa"], fontweight="bold")
axes[0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Protocol C: stop-at-goal planning, matched-memory baseline vs. TI-JEPA", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(_REPO_ROOT, "results", "planning_bar.png"), dpi=170, bbox_inches="tight")
fig.savefig(os.path.join(_REPO_ROOT, "results", "planning_bar.pdf"), bbox_inches="tight")
print("saved planning_bar.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# small-CNN bars in neutral grey ("before"), official ViT-Tiny/AdaLN bars
# in TI-JEPA blue ("after"), so the color itself tells the scale-rescue
# story without needing to read the x-tick labels.
fig2, ax = plt.subplots(figsize=(5.6, 3.6))
labels = ["CartPole\n(small CNN)", "CartPole\n(official ViT/AdaLN)", "Pendulum\n(small CNN)",
          "Pendulum\n(official ViT/AdaLN)"]
vals = [0.302, 0.910, 0.879, 0.832]
colors = [COLORS["neutral"], COLORS["tijepa"], COLORS["neutral"], COLORS["tijepa"]]
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v + 0.02), ha="center", fontsize=9.5)
ax.set_ylabel(r"explicit $v_t$ linear-probe Pearson $r$", fontsize=9.5)
ax.set_ylim(0, 1.05)
ax.set_title("Switching encoder family closes CartPole's small-CNN gap", fontsize=10.5)
fig2.tight_layout()
fig2.savefig(os.path.join(_REPO_ROOT, "results", "scale_bar.png"), dpi=170, bbox_inches="tight")
fig2.savefig(os.path.join(_REPO_ROOT, "results", "scale_bar.pdf"), bbox_inches="tight")
print("saved scale_bar.png")
