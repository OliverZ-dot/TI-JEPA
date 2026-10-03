"""Generate looping GIF demos for the README: for each custom environment,
render two rollouts that start from the exact same frame and differ only in
initial velocity (one of them negated), running the real physics forward
with zero action. The two branches are pixel-identical at t=0 and visibly
diverge afterward -- this is the live, animated version of the kill
experiment used throughout the paper.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ti_jepa.envs.registry import get_env_spec, make_scaled_config  # noqa: E402
from ti_jepa.eval.env_utils import sample_qv1  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "demo_gifs")
os.makedirs(OUT_DIR, exist_ok=True)

ENV_DISPLAY = {"inertia_ball": "InertiaBall", "pendulum": "Pendulum", "cartpole": "CartPole"}
PAD = 10
LABEL_H = 28


def gt_branch_frames(env_name, seed, n_frames, image_size, speed_range):
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

    return rollout(v1), rollout(-v1)


def make_panel(frame_plus, frame_minus, step_idx, n_total, image_size):
    canvas = Image.new("RGB", (image_size * 2 + PAD * 3, image_size + LABEL_H + PAD * 2), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((PAD, PAD // 2), "branch A: +v", fill="black", font=font)
    draw.text((image_size + PAD * 2, PAD // 2), "branch B: -v  (identical frame at t=0)", fill="black", font=font)
    canvas.paste(Image.fromarray(frame_plus), (PAD, LABEL_H))
    canvas.paste(Image.fromarray(frame_minus), (image_size + PAD * 2, LABEL_H))
    draw.text((PAD, canvas.height - 14), f"t = {step_idx}/{n_total - 1}", fill="black", font=font)
    return canvas


def make_gif(env_name, seed, n_frames=24, image_size=160, speed_range=(0.05, 0.09), fps=10):
    frames_plus, frames_minus = gt_branch_frames(env_name, seed, n_frames, image_size, speed_range)
    panels = [
        make_panel(frames_plus[i], frames_minus[i], i, n_frames, image_size)
        for i in range(n_frames)
    ]
    # hold the last frame a bit longer so the divergence is easy to see on loop
    panels = panels + [panels[-1]] * (fps // 2)
    out_path = os.path.join(OUT_DIR, f"{env_name}_kill_demo.gif")
    panels[0].save(
        out_path, save_all=True, append_images=panels[1:], duration=int(1000 / fps), loop=0,
    )
    print("saved", out_path, f"({len(panels)} frames)")


if __name__ == "__main__":
    make_gif("inertia_ball", seed=3, n_frames=22, image_size=160, speed_range=(0.6, 0.9))
    make_gif("pendulum", seed=5, n_frames=26, image_size=160, speed_range=(1.2, 2.0))
    make_gif("cartpole", seed=7, n_frames=26, image_size=160, speed_range=(1.0, 1.8))
