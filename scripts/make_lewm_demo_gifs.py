"""Demo GIFs from the four REAL official LeWM benchmarks (PushT, Reacher,
Cube, TwoRoom), using the real simulators directly (no model in the loop) --
same spirit as make_demo_gifs.py's toy-env branching demos, extended to the
real benchmarks used in the paper's probe-ladder and kill-experiment tables.

PushT and Reacher support the same "identical frame at t=0, opposite initial
velocity, real physics forward" construction already used for the official
PushT/Reacher kill experiments (see real_pusht/kill_experiment.py and
real_pusht/reacher_kill_experiment_official.py) -- a free rigid body (PushT's
block) and a torque-controlled 2-link arm (Reacher) both have real coasting
dynamics, so the branch split is physically meaningful.

Cube (a robot-arm manipulation task) and TwoRoom (a kinematic,
velocity-is-the-action navigation task) don't have that same inertial-coasting
structure available cheaply/safely to script here, so for those two this
file instead renders a single representative real rollout: smoothed random
actions for Cube, a simple goal-directed controller for TwoRoom. These are
genuine real-simulator rollouts, just not a branching construction.

Needs a Python environment with mujoco/dm_control/ogbench/pymunk installed
(matching the official LeWM/stable_worldmodel repo's requirements), and the
SWM_SRC env var pointing at that repo's source root:
  MUJOCO_GL=osmesa SWM_SRC=/path/to/stable_worldmodel/src python3 scripts/make_lewm_demo_gifs.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
if os.environ.get("SWM_SRC"):
    sys.path.insert(0, os.environ["SWM_SRC"])

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "demo_gifs")
os.makedirs(OUT_DIR, exist_ok=True)
PAD = 8
LABEL_H = 20
IMG = 224


def _font():
    return ImageFont.load_default()


def make_dual_panel(frame_a, frame_b, step_idx, n_total, label_a, label_b):
    canvas = Image.new("RGB", (IMG * 2 + PAD * 3, IMG + LABEL_H + PAD * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((PAD, PAD // 2), label_a, fill="black", font=font)
    draw.text((IMG + PAD * 2, PAD // 2), label_b, fill="black", font=font)
    canvas.paste(Image.fromarray(frame_a), (PAD, LABEL_H))
    canvas.paste(Image.fromarray(frame_b), (IMG + PAD * 2, LABEL_H))
    draw.text((PAD, canvas.height - 14), f"t = {step_idx}/{n_total - 1}", fill="black", font=font)
    return canvas


def make_single_panel(frame, step_idx, n_total, label):
    canvas = Image.new("RGB", (IMG + PAD * 2, IMG + LABEL_H + PAD * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((PAD, PAD // 2), label, fill="black", font=font)
    canvas.paste(Image.fromarray(frame), (PAD, LABEL_H))
    draw.text((PAD, canvas.height - 14), f"t = {step_idx}/{n_total - 1}", fill="black", font=font)
    return canvas


def save_gif(panels, out_path, fps=10, hold_frames=5):
    panels = panels + [panels[-1]] * hold_frames
    panels[0].save(out_path, save_all=True, append_images=panels[1:], duration=int(1000 / fps), loop=0)
    print("saved ->", out_path)


# ---------------------------------------------------------------- PushT ----
def make_pusht_gif(seed=11, n_frames=24, horizon_steps_per_frame=5):
    from stable_worldmodel.envs.pusht.env import PushT

    rng = np.random.default_rng(seed)
    bx = rng.uniform(90, 512 - 90)
    by = rng.uniform(90, 512 - 90)
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(60.0, 100.0)
    theta = rng.uniform(0, 2 * np.pi)
    v = np.array([speed * np.cos(theta), speed * np.sin(theta)])
    agent_park = (40.0, 40.0)

    def rollout(v_signed):
        env = PushT(render_mode="rgb_array", resolution=IMG, damping=1.0)
        state = [agent_park[0], agent_park[1], bx, by, angle, 0.0, 0.0]
        env.reset(seed=0, options={"state": state, "goal_state": state})
        env.block.velocity = tuple(v_signed.tolist())
        frames = [env.render()]
        zero_action = np.zeros(2, dtype=np.float32)
        for _ in range(n_frames - 1):
            for _ in range(horizon_steps_per_frame):
                env.step(zero_action)
            frames.append(env.render())
        return np.stack(frames)

    frames_plus = rollout(v)
    frames_minus = rollout(-v)
    panels = [
        make_dual_panel(frames_plus[i], frames_minus[i], i, n_frames, "real PushT: +v", "real PushT: -v")
        for i in range(n_frames)
    ]
    save_gif(panels, os.path.join(OUT_DIR, "pusht_real_kill_demo.gif"))


# ---------------------------------------------------------------- Reacher --
def make_reacher_gif(seed=3, n_frames=24, model_step_steps=3):
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

    rng = np.random.default_rng(seed)
    q0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)])
    speed = rng.uniform(1.5, 2.5, size=2)
    sign = rng.choice([-1.0, 1.0], size=2)
    v = speed * sign

    def rollout(v_signed):
        env = ReacherDMControlWrapper(task="hard", seed=seed)
        state = np.concatenate([q0, v_signed])
        env.reset(seed=seed, options={"state": state})
        frames = [env.render()]
        zero_action = np.zeros(2, dtype=np.float32)
        for _ in range(n_frames - 1):
            for _ in range(model_step_steps):
                env.step(zero_action)
            frames.append(env.render())
        return np.stack(frames)

    frames_plus = rollout(v)
    frames_minus = rollout(-v)
    panels = [
        make_dual_panel(frames_plus[i], frames_minus[i], i, n_frames, "real Reacher: +v", "real Reacher: -v")
        for i in range(n_frames)
    ]
    save_gif(panels, os.path.join(OUT_DIR, "reacher_real_kill_demo.gif"))


# ------------------------------------------------------------------ Cube ---
def make_cube_gif(seed=5, n_frames=30):
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    env = CubeEnv(env_type="single", ob_type="pixels")
    env.reset(seed=seed, options={"variation": ["all"]})
    rng = np.random.default_rng(seed)
    action_dim = env.action_space.shape[0]
    frames = [env.render()]
    cur = np.zeros(action_dim, dtype=np.float32)
    for _ in range(n_frames - 1):
        # smoothed random walk: blend toward a new random target action
        target = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        target[-1] = rng.choice([-1.0, 1.0])  # gripper open/close less jittery
        cur = 0.75 * cur + 0.25 * target
        cur = np.clip(cur, -1.0, 1.0)
        env.step(cur)
        frames.append(env.render())
    frames = np.stack(frames)
    panels = [make_single_panel(frames[i], i, n_frames, "real CubeEnv (official OGBench/MuJoCo manipulation)")
              for i in range(n_frames)]
    save_gif(panels, os.path.join(OUT_DIR, "cube_real_demo.gif"))


# --------------------------------------------------------------- TwoRoom ---
def make_tworoom_gif(seed=7, n_frames=30):
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv()
    env.reset(seed=seed)
    frames = [env.render()]
    for _ in range(n_frames - 1):
        delta = (env.target_position - env.agent_position).numpy()
        dist = np.linalg.norm(delta) + 1e-6
        action = np.clip(delta / dist, -1.0, 1.0).astype(np.float32)
        env.step(action)
        frames.append(env.render())
    frames = np.stack(frames)
    panels = [make_single_panel(frames[i], i, n_frames, "real TwoRoom (official navigation benchmark)")
              for i in range(n_frames)]
    save_gif(panels, os.path.join(OUT_DIR, "tworoom_real_demo.gif"))


if __name__ == "__main__":
    make_pusht_gif()
    make_reacher_gif()
    make_cube_gif()
    make_tworoom_gif()
