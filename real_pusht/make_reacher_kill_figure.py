"""Build a paper-ready qualitative figure for the real dm_control Reacher
kill experiment (Protocol B, official ViT-Tiny/AdaLN scale, r16 in
the results log). Reuses the exact same context-construction / ground-truth
utilities as reacher_kill_experiment_official.py so the rendered frames are
a genuine example from the same distribution the reported numbers came
from, then overlays the branch-separation curves already computed and
cached in results/reacher_official/kill_experiment.json (no re-training,
no re-scoring, purely a visualization pass).

Usage:
    MUJOCO_GL=egl python3 make_reacher_kill_figure.py
"""
import json
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "real_pusht"))
if os.environ.get("SWM_SRC"):
    sys.path.insert(0, os.environ["SWM_SRC"])
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper  # noqa: E402
from reacher_kill_experiment_official import (  # noqa: E402
    build_context_frames, sample_config, MODEL_STEP_DT,
)

OUT_JSON = os.path.join(_REPO_ROOT, "real_pusht", "results", "reacher_official", "kill_experiment.json")
OUT_FIG = os.path.join(_REPO_ROOT, "results", "reacher_official", "kill_experiment_reacher.png")

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

with open(OUT_JSON) as f:
    data = json.load(f)

env = ReacherDMControlWrapper(task="hard", seed=7)
rng = np.random.default_rng(7)
q0, v = sample_config(rng)
ctx_plus = build_context_frames(env, q0, v, k=3)
ctx_minus = build_context_frames(env, q0, -v, k=3)
img_diff = float(np.abs(ctx_plus[-1].astype(np.float32) - ctx_minus[-1].astype(np.float32)).mean())

fig = plt.figure(figsize=(11.5, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.1], wspace=0.35)

ax0 = fig.add_subplot(gs[0])
ax0.imshow(ctx_plus[-1])
ax0.set_title(r"$o_t$  ($v=+v$)", fontsize=10)
ax0.axis("off")

ax1 = fig.add_subplot(gs[1])
ax1.imshow(ctx_minus[-1])
ax1.set_title(r"$o_t$  ($v=-v$)" + "\n(pixel-identical to left)", fontsize=10)
ax1.axis("off")

ax2 = fig.add_subplot(gs[2])
diff_img = np.abs(ctx_plus[-1].astype(np.float32) - ctx_minus[-1].astype(np.float32)).sum(-1)
ax2.imshow(diff_img, cmap="hot", vmin=0, vmax=1)
ax2.set_title(f"|pixel diff|\n(mean={img_diff:.4f})", fontsize=10)
ax2.axis("off")

ax3 = fig.add_subplot(gs[3])
steps = np.arange(len(data["baseline_k3"]["gt_curve"]))
colors = {"baseline_k1": "tab:red", "baseline_k3": "tab:orange", "tijepa": "tab:blue"}
labels = {"baseline_k1": r"baseline$_{k=1}$ (memoryless)", "baseline_k3": r"baseline$_{k=3}$ (history)",
          "tijepa": "TI-JEPA"}
ax3.plot(steps, data["baseline_k3"]["gt_curve"], "k--", lw=2, label="ground truth")
for name in ["baseline_k1", "baseline_k3", "tijepa"]:
    ax3.plot(steps, data[name]["pred_curve"], color=colors[name], lw=2,
              label=f"{labels[name]}  (ratio={data[name]['ratio_mean']:.3f})")
ax3.set_xlabel("model step (blind rollout, a=0)", fontsize=9)
ax3.set_ylabel(r"$\|$decoded finger-pos$(+v)$ $-$ decoded finger-pos$(-v)\|$", fontsize=8)
ax3.set_title("Same $q_0$, opposite velocity: does the blind rollout tell them apart?", fontsize=9)
ax3.legend(fontsize=7.5, loc="upper left")
ax3.grid(alpha=0.3)

fig.suptitle("Real dm_control Reacher, official ViT-Tiny/14 + AdaLN scale, Protocol B kill experiment",
             fontsize=11, y=1.03)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=170, bbox_inches="tight")
fig.savefig(OUT_FIG.replace(".png", ".pdf"), bbox_inches="tight")
print("saved ->", OUT_FIG)
print("img_diff (should be ~0, confirming contexts are pixel-identical):", img_diff)
