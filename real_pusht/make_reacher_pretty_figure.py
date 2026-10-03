"""Publication-quality replacement for kill_experiment_reacher.png: adds a
real 2-row x N-column frame strip of the actual dm_control Reacher rollout
(same qpos, opposite velocity) on top of the existing branch-separation
curve (loaded from the already-computed results/reacher_official/
kill_experiment.json, no re-scoring). Uses the shared paper color palette.

    MUJOCO_GL=egl python3 make_reacher_pretty_figure.py
"""
import json
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

from plot_style import COLORS, LABELS, setup_style  # noqa: E402

setup_style()
import matplotlib.pyplot as plt  # noqa: E402

from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper  # noqa: E402
from reacher_kill_experiment_official import sample_config, MODEL_STEP_STEPS  # noqa: E402

OUT_JSON = os.path.join(_REPO_ROOT, "real_pusht", "results", "reacher_official", "kill_experiment.json")
OUT_FIG = os.path.join(_REPO_ROOT, "results", "pretty", "reacher_kill.png")

N_FRAMES = 8
SEED = 11

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

with open(OUT_JSON) as f:
    data = json.load(f)

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

fig = plt.figure(figsize=(1.15 * N_FRAMES, 6.4))
gs = fig.add_gridspec(3, N_FRAMES, height_ratios=[1, 1, 1.9], hspace=0.30, wspace=0.06)

top_axes, bottom_axes = [], []
for i in range(N_FRAMES):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(frames_plus[i], aspect="auto"); ax.axis("off")
    ax.set_title(f"$t={i}$", fontsize=11, color="#444444")
    top_axes.append(ax)
    ax2 = fig.add_subplot(gs[1, i])
    ax2.imshow(frames_minus[i], aspect="auto"); ax2.axis("off")
    bottom_axes.append(ax2)
top_axes[0].text(-0.35, 0.5, "$v=+v$", transform=top_axes[0].transAxes, fontsize=12.5,
                  color=COLORS["tijepa"], ha="right", va="center")
bottom_axes[0].text(-0.35, 0.5, "$v=-v$", transform=bottom_axes[0].transAxes, fontsize=12.5,
                     color=COLORS["baseline_k1"], ha="right", va="center")

ax = fig.add_subplot(gs[2, :])
steps = np.arange(len(data["baseline_k3"]["gt_curve"]))
color_key = {"baseline_k1": "baseline_k1", "baseline_k3": "baseline_k3", "tijepa": "tijepa"}
ax.plot(steps, data["baseline_k3"]["gt_curve"], color=COLORS["gt"], ls="--", lw=2.3, label=LABELS["gt"])
for name in ["baseline_k1", "baseline_k3", "tijepa"]:
    ax.plot(steps, data[name]["pred_curve"], color=COLORS[color_key[name]], lw=2.3,
             label=f"{LABELS[color_key[name]]}  (ratio={data[name]['ratio_mean']:.2f})")
ax.set_xlabel("model step (blind rollout, $a=0$)", fontsize=11.5)
ax.set_ylabel(r"$\|$decoded finger-pos$(+v)-$decoded finger-pos$(-v)\|$", fontsize=11)
ax.set_title("Blind rollout: does the model tell the two branches apart?", fontsize=12.5)
ax.tick_params(labelsize=10)
ax.legend(loc="upper left", fontsize=10.5)
ax.grid(alpha=0.25)

fig.suptitle(
    f"Real dm\\_control Reacher, official ViT-Tiny/14 + AdaLN scale, Protocol B kill experiment  "
    f"(frame 0 pixel difference $={img_diff:.4f}$)".replace("dm\\_control", "dm_control"),
    fontsize=13.5, y=1.02,
)
fig.subplots_adjust(left=0.075, right=0.995, top=0.95, bottom=0.06, hspace=0.30, wspace=0.06)
fig.savefig(OUT_FIG, bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT_FIG.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
print("saved ->", OUT_FIG)
