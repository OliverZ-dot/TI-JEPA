"""Protocol B (kill experiment) on the REAL official PushT env + REAL
official pretrained LeWM checkpoint, per the internal project spec (not included in this release) §4.2.

Construction ("same config q, opposite velocity v/-v", exactly as specified):
  1. Pick a random block config q0 = (bx, by, angle) away from the walls.
  2. For velocity v (and independently for -v): build a k=3-frame context by
     TELEPORTING the block backwards along a constant-velocity line and
     rendering each frame (`_set_state` + `render()`, no physics stepping --
     same trick as the InertiaBall toy env). The agent is parked far away
     (not touching the block) throughout, isolating "block carries velocity"
     from "agent is currently pushing it".
  3. Ground truth future: reset the REAL simulator to q0, set the block's
     pymunk body velocity directly to v (bypassing the gym action interface,
     eval-only), hold the agent still (a=0), and step real physics forward.
     The official env's default damping (`space.damping=0`, which in pymunk
     semantics is *maximal* damping, not "no damping") makes the block stop
     dead within one physics substep -- so training data essentially never
     contains real "coasting" motion. For this eval-only diagnostic we
     override `damping=1.0` (pymunk's true frictionless default) so the
     ground truth actually has something to distinguish; this mirrors
     §4.2's own suggested fix ("加大块的惯性/减小阻尼") and is reported
     transparently, not hidden.
  4. Blind rollout: encode the k=3-frame context with the FROZEN official
     encoder, then autoregressively roll the FROZEN official predictor
     forward with all-zero actions (`model.rollout`, history_size=3, exactly
     as it does at inference time for planning).
  5. Decode both the model's predicted latents AND compare against the real
     ground-truth position trajectory via a linear position probe fit on
     real held-out data (single-frame emb -> block/agent position, R^2~0.9+
     per protocol A) -- purely a read-out tool, never touches training.

Metrics per pair, matching kill_experiment.md's toy-env conventions:
  - branch separation ratio: ||decoded_pos(v) - decoded_pos(-v)|| /
    ||gt_pos(v) - gt_pos(-v)|| at each horizon step (1.0 = model tracks the
    true divergence; ~0 = model collapses both velocity branches together)
  - position MSE of decoded rollout vs ground truth, per branch
  - velocity-sign accuracy: does the decoded trajectory's net displacement
    direction match the true v's sign
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

from common import LEWM_REPO, align_pixels, encode_clip_batch, load_clips_dataset, load_model, normalize_pixels

sys.path.insert(0, str(LEWM_REPO))
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402

STEP_DT = 0.1          # seconds per real env.step()
MODEL_STEP_STEPS = 5   # frameskip: 1 "model step" = 5 real env steps = 0.5s
MODEL_STEP_DT = STEP_DT * MODEL_STEP_STEPS
HISTORY = 3
ACTION_DIM = 10  # action_block(5) * 2
AGENT_PARK = (40.0, 40.0)  # far corner, not touching the block
BLOCK_MARGIN = 90.0
WINDOW = 512.0


def fit_position_probe(model, device, num_clips=4000, seed=0):
    """single-frame emb -> [agent_x,agent_y,block_x,block_y,block_angle],
    fit on real held-out pusht_expert_train clips. Eval-only decoding tool."""
    ds = load_clips_dataset("pusht", num_steps=2, frameskip=5, extra_keys=["state"])
    g = torch.Generator().manual_seed(seed)
    n = min(num_clips, len(ds))
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=256, shuffle=False)
    X, Y = [], []
    with torch.no_grad():
        for batch in loader:
            emb = encode_clip_batch(model, batch["pixels"], device).numpy()
            X.append(emb[:, 0])
            Y.append(batch["state"][:, 0, :5].numpy())
    X, Y = np.concatenate(X), np.concatenate(Y)
    reg = Ridge(alpha=1.0).fit(X, Y)
    r2 = reg.score(X, Y)
    print(f"[probe] position decode probe: in-sample R^2={r2:.4f} (n={len(X)})", flush=True)
    return reg


def sample_config(rng):
    bx = rng.uniform(BLOCK_MARGIN, WINDOW - BLOCK_MARGIN)
    by = rng.uniform(BLOCK_MARGIN, WINDOW - BLOCK_MARGIN)
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(40.0, 90.0)
    theta = rng.uniform(0, 2 * np.pi)
    v = np.array([speed * np.cos(theta), speed * np.sin(theta)])
    return bx, by, angle, v


def build_context_frames(env, bx, by, angle, v, k=HISTORY):
    """Teleport the block backwards along a constant-v line for k frames,
    spaced MODEL_STEP_DT apart, agent parked away. Returns (k,H,W,3) uint8."""
    frames = []
    for i in range(k):
        dt_back = (k - 1 - i) * MODEL_STEP_DT
        pos = np.array([bx, by]) - v * dt_back
        state = [AGENT_PARK[0], AGENT_PARK[1], pos[0], pos[1], angle, 0.0, 0.0]
        env.reset(seed=0, options={"state": state, "goal_state": state})
        frames.append(env.render())
    return np.stack(frames, axis=0)


def ground_truth_rollout(env, bx, by, angle, v, horizon_model_steps):
    """Real pymunk physics: block starts at (bx,by,angle) with velocity v,
    agent held at AGENT_PARK via a=0 actions. damping overridden to 1.0 in
    the env passed in. Returns positions (horizon+1, 2) and frames list."""
    state = [AGENT_PARK[0], AGENT_PARK[1], bx, by, angle, 0.0, 0.0]
    env.reset(seed=0, options={"state": state, "goal_state": state})
    env.block.velocity = tuple(v.tolist())
    positions = [np.array([bx, by])]
    frames = [env.render()]
    zero_action = np.zeros(2, dtype=np.float32)
    for _ in range(horizon_model_steps):
        for _ in range(MODEL_STEP_STEPS):
            obs, *_ = env.step(zero_action)
        positions.append(obs["state"][2:4].copy())
        frames.append(env.render())
    return np.stack(positions, axis=0), frames


@torch.no_grad()
def model_blind_rollout(model, context_frames_uint8: np.ndarray, device: str, horizon_model_steps: int):
    """context_frames_uint8: (K,H,W,3) -> predicted emb rollout (K+horizon, D)."""
    pixels = torch.from_numpy(context_frames_uint8).unsqueeze(0)  # (1,K,H,W,3) uint8
    K = pixels.shape[1]
    T = K + horizon_model_steps
    action_seq = torch.zeros(1, 1, T, ACTION_DIM, device=device)
    # JEPA.rollout() calls self.encode() itself on info["pixels"][:,0] (S dim),
    # so pixels must already be normalized/channel-first float, matching what
    # encode() expects when called directly (unlike our encode_clip_batch
    # convenience wrapper which does this externally for the probe scripts).
    pixels_norm = normalize_pixels(align_pixels(pixels)).to(device)  # (1,K,C,H,W)
    pixels_bs = pixels_norm.unsqueeze(1)  # (1,1,K,C,H,W)
    out = model.rollout({"pixels": pixels_bs}, action_seq, history_size=HISTORY)
    pred_emb = out["predicted_emb"][0, 0]  # (K+horizon, D)
    return pred_emb.cpu().numpy()


def decode_positions(probe: Ridge, emb: np.ndarray) -> np.ndarray:
    return probe.predict(emb)[:, 2:4]  # block x,y columns


def run(args):
    device = args.device
    model = load_model("pusht", device)
    probe = fit_position_probe(model, device, num_clips=args.probe_clips, seed=args.seed)

    env_ctx = PushT(render_mode="rgb_array", resolution=224, damping=1.0)
    env_gt = PushT(render_mode="rgb_array", resolution=224, damping=1.0)

    rng = np.random.default_rng(args.seed)
    H = args.horizon
    records = []
    example_pair = None

    for p in range(args.n_pairs):
        bx, by, angle, v = sample_config(rng)
        branch_data = {}
        for sign, v_signed in [("+v", v), ("-v", -v)]:
            ctx = build_context_frames(env_ctx, bx, by, angle, v_signed)
            gt_pos, gt_frames = ground_truth_rollout(env_gt, bx, by, angle, v_signed, H)
            pred_emb = model_blind_rollout(model, ctx, device, H)
            pred_pos_all = decode_positions(probe, pred_emb)  # JEPA.rollout appends one extra dupe step -> len K+H+1
            pred_pos = pred_pos_all[HISTORY - 1 : HISTORY - 1 + H + 1]  # align: index 0 = "now", matches gt_pos (len H+1)
            branch_data[sign] = {
                "gt_pos": gt_pos, "pred_pos": pred_pos,
                "pred_emb": pred_emb[HISTORY - 1 : HISTORY - 1 + H + 1],
            }
            if p == 0 and sign == "+v" and example_pair is None:
                example_pair = {"ctx_frames": ctx, "gt_frames": gt_frames}

        gt_sep = np.linalg.norm(branch_data["+v"]["gt_pos"] - branch_data["-v"]["gt_pos"], axis=-1)
        pred_sep = np.linalg.norm(branch_data["+v"]["pred_pos"] - branch_data["-v"]["pred_pos"], axis=-1)
        z_sep = np.linalg.norm(branch_data["+v"]["pred_emb"] - branch_data["-v"]["pred_emb"], axis=-1)

        pos_mse = {
            sign: float(np.mean((branch_data[sign]["pred_pos"] - branch_data[sign]["gt_pos"]) ** 2))
            for sign in ("+v", "-v")
        }
        # velocity sign accuracy: does decoded net displacement over full horizon match true v direction
        true_disp = v * H * MODEL_STEP_DT
        pred_disp_plus = branch_data["+v"]["pred_pos"][-1] - branch_data["+v"]["pred_pos"][0]
        pred_disp_minus = branch_data["-v"]["pred_pos"][-1] - branch_data["-v"]["pred_pos"][0]
        sign_ok_plus = float(np.dot(pred_disp_plus, true_disp) > 0)
        sign_ok_minus = float(np.dot(pred_disp_minus, -true_disp) > 0)

        records.append({
            "bx": bx, "by": by, "angle": angle, "v": v.tolist(),
            "gt_sep": gt_sep.tolist(), "pred_sep": pred_sep.tolist(), "z_sep": z_sep.tolist(),
            "branch_sep_ratio_final": float(pred_sep[-1] / (gt_sep[-1] + 1e-6)),
            "pos_mse_plus": pos_mse["+v"], "pos_mse_minus": pos_mse["-v"],
            "sign_ok_plus": sign_ok_plus, "sign_ok_minus": sign_ok_minus,
        })
        if (p + 1) % 10 == 0:
            print(f"pair {p+1}/{args.n_pairs} done", flush=True)

    gt_sep_curve = np.mean([r["gt_sep"] for r in records], axis=0)
    pred_sep_curve = np.mean([r["pred_sep"] for r in records], axis=0)
    z_sep_curve = np.mean([r["z_sep"] for r in records], axis=0)
    branch_ratio_mean = float(np.mean([r["branch_sep_ratio_final"] for r in records]))
    pos_mse_mean = float(np.mean([r["pos_mse_plus"] for r in records] + [r["pos_mse_minus"] for r in records]))
    sign_acc = float(np.mean([r["sign_ok_plus"] for r in records] + [r["sign_ok_minus"] for r in records]))

    summary = {
        "env": "pusht_real_official_ckpt", "n_pairs": args.n_pairs, "horizon_model_steps": H,
        "model_step_dt": MODEL_STEP_DT, "damping_override": 1.0,
        "gt_sep_curve": gt_sep_curve.tolist(), "pred_sep_curve": pred_sep_curve.tolist(),
        "z_sep_curve": z_sep_curve.tolist(),
        "branch_separation_ratio_at_final_horizon_mean": branch_ratio_mean,
        "position_mse_mean": pos_mse_mean, "velocity_sign_accuracy": sign_acc,
    }
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2), flush=True)

    Path("results").mkdir(exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"saved -> {args.out_json}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    t = np.arange(H + 1) * MODEL_STEP_DT
    axes[0].plot(t, gt_sep_curve, "k-", lw=2, label="ground truth |pos(+v)-pos(-v)|")
    axes[0].plot(t, pred_sep_curve, "r--", lw=2, label="official LeWM (decoded) |pos(+v)-pos(-v)|")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("branch separation (px)")
    axes[0].set_title(f"Real PushT kill experiment (n={args.n_pairs} pairs)\nofficial pretrained checkpoint, blind rollout")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, z_sep_curve, "b-", lw=2)
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("||z(+v) - z(-v)|| (predicted latent space)")
    axes[1].set_title("Latent-space branch separation over blind rollout")
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130)
    print(f"saved -> {args.out_fig}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--probe_clips", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="results/kill_experiment_pusht_real.json")
    ap.add_argument("--out_fig", default="results/kill_experiment_pusht_real.png")
    args = ap.parse_args()
    run(args)
