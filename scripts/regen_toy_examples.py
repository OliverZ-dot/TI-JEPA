"""Regenerate the homepage's interactive-player data (data/examples.json +
data/{env}/plus_*.png, minus_*.png) for the three toy environments, fixing a
real bug: the previously-saved data had 16-point model/ground-truth curves
but only 8 rendered frame images per branch, so the on-page player's image
panel visually froze at frame 7 while the chart kept going to frame 15 (user
report: "demo videos go static after ~frame 10").

This script regenerates BOTH the images and the curves from a single,
self-consistent (q, v) rollout per environment, at the full horizon, so
images and curves always have matching length. No numbers are invented:
ground truth is real physics rollout + render(), baseline/TI-JEPA curves are
real blind latent rollouts decoded with a linear probe fit on held-out data
-- exactly the same procedure as ti_jepa/eval/kill_experiment.py, just also
keeping the per-step rendered frames (which kill_experiment.py throws away)
and extending them to the same length as the curves.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ti_jepa.data import EpisodeBatch, cache_path, split_episodes
from ti_jepa.envs.registry import get_env_spec, make_scaled_config
from ti_jepa.eval.common import load_baseline, load_tijepa
from ti_jepa.eval.env_utils import build_context_generic, sample_qv1
from ti_jepa.eval.kill_experiment import (
    blind_rollout_baseline,
    blind_rollout_tijepa,
    fit_position_probe,
    rollout_ground_truth,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOMEPAGE = "/root/ti-jepa-homepage"
DISPLAY_SIZE = 200  # matches the existing PNGs on the homepage
MODEL_SIZE = 64  # matches the toy-env checkpoints' trained image_size
HORIZON = 15  # -> 16 points/frames per branch, matching the previous curve length

CKPTS = {
    "pendulum": dict(baseline_k1="checkpoints/pendulum/baseline_k1.pt", tijepa="checkpoints/pendulum/tijepa.pt"),
    "cartpole": dict(baseline_k1="checkpoints/cartpole/baseline_k1.pt", tijepa="checkpoints/cartpole/tijepa.pt"),
    "inertia_ball": dict(baseline_k1="checkpoints/baseline_k1.pt", tijepa="checkpoints/tijepa.pt"),
}

# Known-good (seed, speed_range) picked for a clean, visible divergence --
# same values already used for the homepage's looping GIFs in make_demo_gifs.py.
SEED_CFG = {
    "pendulum": dict(seed=5, speed_range=(1.2, 2.0)),
    "cartpole": dict(seed=7, speed_range=(1.0, 1.8)),
    "inertia_ball": dict(seed=3, speed_range=(0.6, 0.9)),
}


def render_branch_frames(display_env, q, v, horizon, action_dim):
    display_env.set_state(q, v)
    zero = np.zeros(action_dim, dtype=np.float32)
    frames = [display_env.render()]
    for _ in range(horizon):
        frames.append(display_env.step(zero))
    return frames  # list of (H+1) uint8 arrays


def regen_one(env_name, device="cpu"):
    spec = get_env_spec(env_name)
    action_dim = spec.action_dim
    ck = CKPTS[env_name]

    print(f"=== {env_name} ===")
    b1_enc, b1_pred, b1_args = load_baseline(ck["baseline_k1"], device)
    t_enc, t_pred, t_args = load_tijepa(ck["tijepa"], device)
    assert b1_args["k"] == 1, "baseline_k1 checkpoint 的 k 应该是 1"
    kt = t_args["k"]

    image_size = b1_args.get("image_size", MODEL_SIZE)
    assert image_size == MODEL_SIZE and t_args.get("image_size", MODEL_SIZE) == MODEL_SIZE

    d = np.load(cache_path(env_name, image_size))
    full_batch = EpisodeBatch(d["frames"], d["actions"], d["positions"], d["velocities"])
    _, held_out = split_episodes(full_batch, int(d["n_train"]))
    b1_probe = fit_position_probe("baseline", b1_enc, 1, device, held_out)
    t_probe = fit_position_probe("tijepa", t_enc, kt, device, held_out)

    model_cfg = make_scaled_config(env_name, MODEL_SIZE)
    model_env = spec.env_cls(model_cfg)
    display_cfg = make_scaled_config(env_name, DISPLAY_SIZE)
    display_env = spec.env_cls(display_cfg)

    sc = SEED_CFG[env_name]
    rng = np.random.default_rng(sc["seed"])
    q, v1 = sample_qv1(env_name, rng, model_cfg, k=max(kt, 1), horizon=HORIZON, speed_range=sc["speed_range"])
    v2 = -v1

    # ---- ground truth: real physics + real render(), full horizon, both branches ----
    frames_plus = render_branch_frames(display_env, q, v1, HORIZON, action_dim)
    frames_minus = render_branch_frames(display_env, q, v2, HORIZON, action_dim)
    frame_diff0 = float(np.abs(frames_plus[0].astype(np.float32) - frames_minus[0].astype(np.float32)).mean())

    gt_pos1, _ = rollout_ground_truth(model_env, q, v1, HORIZON, action_dim)
    gt_pos2, _ = rollout_ground_truth(model_env, q, v2, HORIZON, action_dim)

    # ---- baseline_k1 (memoryless predictor): single current frame as context ----
    ctx1_1 = build_context_generic(model_env, q, v1, 1, action_dim)
    ctx2_1 = build_context_generic(model_env, q, v2, 1, action_dim)
    z_roll1 = blind_rollout_baseline(b1_enc, b1_pred, ctx1_1, 1, HORIZON, device, action_dim)
    z_roll2 = blind_rollout_baseline(b1_enc, b1_pred, ctx2_1, 1, HORIZON, device, action_dim)
    base_pos1 = b1_probe.predict(z_roll1)
    base_pos2 = b1_probe.predict(z_roll2)

    # ---- TI-JEPA (memoryless predictor, structured z=(q,v)): kt-frame context ----
    ctx1_t = build_context_generic(model_env, q, v1, kt, action_dim)
    ctx2_t = build_context_generic(model_env, q, v2, kt, action_dim)
    q_roll1, _ = blind_rollout_tijepa(t_enc, t_pred, ctx1_t, kt, HORIZON, device, action_dim)
    q_roll2, _ = blind_rollout_tijepa(t_enc, t_pred, ctx2_t, kt, HORIZON, device, action_dim)
    ti_pos1 = t_probe.predict(q_roll1)
    ti_pos2 = t_probe.predict(q_roll2)

    dim0 = 0
    out_dir = os.path.join(HOMEPAGE, "data", env_name)
    os.makedirs(out_dir, exist_ok=True)
    # wipe old (possibly shorter) frame sets so no stale 8-frame leftovers remain
    for f in os.listdir(out_dir):
        if f.startswith("plus_") or f.startswith("minus_"):
            os.remove(os.path.join(out_dir, f))
    for i, fr in enumerate(frames_plus):
        Image.fromarray(fr).save(os.path.join(out_dir, f"plus_{i}.png"))
    for i, fr in enumerate(frames_minus):
        Image.fromarray(fr).save(os.path.join(out_dir, f"minus_{i}.png"))

    n = len(frames_plus)
    assert n == HORIZON + 1 == gt_pos1.shape[0] == base_pos1.shape[0] == ti_pos1.shape[0]
    print(f"  wrote {n} frames/branch, curve length {n} (images and curves now match)")

    entry = {
        "env": env_name,
        "frame_diff0": frame_diff0,
        "plus": [f"data/{env_name}/plus_{i}.png" for i in range(n)],
        "minus": [f"data/{env_name}/minus_{i}.png" for i in range(n)],
        "gt_plus": [float(x) for x in gt_pos1[:, dim0]],
        "gt_minus": [float(x) for x in gt_pos2[:, dim0]],
        "base_plus": [float(x) for x in base_pos1[:, dim0]],
        "base_minus": [float(x) for x in base_pos2[:, dim0]],
        "ti_plus": [float(x) for x in ti_pos1[:, dim0]],
        "ti_minus": [float(x) for x in ti_pos2[:, dim0]],
    }
    return entry


def main():
    os.chdir(ROOT)  # cache_path / checkpoints are relative to ti-jepa/
    ex_path = os.path.join(HOMEPAGE, "data", "examples.json")
    data = json.load(open(ex_path))
    for env_name in ["pendulum", "cartpole", "inertia_ball"]:
        data[env_name] = regen_one(env_name)
    with open(ex_path, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", ex_path)


if __name__ == "__main__":
    main()
