"""Compact main-text version: just the real frame strip (no curve panel,
which would duplicate Table 5's numbers), six columns, sized to fit inside
the ICLR page budget. The full frame-strip + curve version lives in
results/pretty/reacher_kill.png for the appendix.

    MUJOCO_GL=egl python3 make_reacher_compact_figure.py
"""
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "real_pusht"))
if os.environ.get("SWM_SRC"):
    sys.path.insert(0, os.environ["SWM_SRC"])

from plot_style import COLORS, setup_style  # noqa: E402

setup_style()
import matplotlib.pyplot as plt  # noqa: E402

from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper  # noqa: E402
from reacher_kill_experiment_official import sample_config, MODEL_STEP_STEPS  # noqa: E402

OUT_FIG = os.path.join(_REPO_ROOT, "results", "pretty", "reacher_kill_compact.png")
N_FRAMES = 6
SEED = 11

env = ReacherDMControlWrapper(task="hard", seed=SEED)
rng = np.random.default_rng(SEED)
q0, v = sample_config(rng)
zero_action = np.zeros(2, dtype=np.float32)


def rollout(v_signed):
    state = np.concatenate([q0, v_signed])
    env.reset(seed=SEED, options={"state": state})
    frames = [env.render()]
    for _ in range(N_FRAMES - 1):
        for _ in range(MODEL_STEP_STEPS):
            env.step(zero_action)
        frames.append(env.render())
    return np.stack(frames)


frames_plus = rollout(v)
frames_minus = rollout(-v)
img_diff = float(np.abs(frames_plus[0].astype(np.float32) - frames_minus[0].astype(np.float32)).mean())

fig, axes = plt.subplots(2, N_FRAMES, figsize=(1.05 * N_FRAMES, 2.6),
                          gridspec_kw={"wspace": 0.05, "hspace": 0.08})
for i in range(N_FRAMES):
    axes[0][i].imshow(frames_plus[i], aspect="auto"); axes[0][i].axis("off")
    axes[0][i].set_title(f"$t={i}$", fontsize=11, color="#444444")
    axes[1][i].imshow(frames_minus[i], aspect="auto"); axes[1][i].axis("off")
axes[0][0].text(-0.35, 0.5, "$v=+v$", transform=axes[0][0].transAxes, fontsize=12,
                 color=COLORS["tijepa"], ha="right", va="center")
axes[1][0].text(-0.35, 0.5, "$v=-v$", transform=axes[1][0].transAxes, fontsize=12,
                 color=COLORS["baseline_k1"], ha="right", va="center")
fig.suptitle(
    f"Real dm\\_control Reacher: same $q_0$, opposite $v$  (frame 0 pixel diff $={img_diff:.4f}$)",
    fontsize=13, y=1.04,
)
fig.subplots_adjust(left=0.09, right=0.995, top=0.82, bottom=0.01, wspace=0.05, hspace=0.08)
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
fig.savefig(OUT_FIG, bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT_FIG.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
print("saved ->", OUT_FIG)
