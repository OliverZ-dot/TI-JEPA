"""Protocol B (kill experiment) for the official-scale (ViT-Tiny/14 + AdaLN)
baseline_k1/baseline_k3/TI-JEPA checkpoints trained on REAL dm_control
Reacher pixels, using the REAL live `swm/ReacherDMControl-v0` env for both
context construction and ground-truth rollout (mirrors real_pusht's PushT
kill_experiment.py, adapted to Reacher's `set_state(qpos,qvel)` interface).

Construction ("same qpos q0, opposite velocity v/-v"):
  1. Sample q0=(shoulder,elbow) away from the elbow's joint limit, and a
     velocity v=(v_shoulder,v_elbow) at dataset-typical scale.
  2. Context (k=3 frames): for each of the two branches (+v, -v),
     back-extrapolate qpos linearly at constant velocity over the last
     k model-steps (`qpos_i = q0 - v_signed*(k-1-i)*MODEL_STEP_DT`), and for
     each i call `env.reset(options={"state":[qpos_i, v_signed]})` +
     `env.render()`. Real Reacher dynamics are nonlinear (coupled 2-link
     torque-controlled arm), so this is an approximation of "what the last
     k frames would have looked like", exactly the same construction trick
     `real_pusht/kill_experiment.py` uses for PushT (there justified for a
     free rigid body; here it's a genuine approximation for a short window,
     reported as such, not hidden).
  3. Ground truth future: `env.reset(options={"state":[q0, v_signed]})`, then
     step REAL physics forward with a=0 (`MODEL_STEP_STEPS` raw `.step()`
     calls per model step, matching the cache's frameskip decimation),
     recording `finger_pos` (2D Cartesian, bounded, non-periodic -- a clean
     decode target unlike raw qpos) at each model step.
  4. Blind rollout: encode the k-frame context with the FROZEN checkpoint's
     encoder+predictor, autoregressively roll forward with a=0 for
     `horizon` model steps (predictor and encoder never see real GT after
     the context).
  5. Decode predicted embeddings -> finger_pos via a linear Ridge probe
     fit on held-out real cache windows (eval-only tool, never touches
     training).

Metrics per pair: branch-separation ratio (||decoded_pos(+v)-decoded_pos(-v)||
/ ||gt_pos(+v)-gt_pos(-v)|| at final horizon), rollout position MSE per
branch, velocity-sign accuracy (does net displacement direction match v's
sign). Same conventions as the custom-env / real-PushT kill experiments.
"""

from __future__ import annotations

import argparse
import json
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, _REPO_ROOT)
from ti_jepa.models import LeWMStyleEncoder, TIJEPAEncoder  # noqa: E402
from ti_jepa.official_predictor import OfficialBaselinePredictor, OfficialTIJEPAPredictor  # noqa: E402

if os.environ.get("SWM_SRC"):
    sys.path.insert(0, os.environ["SWM_SRC"])
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper  # noqa: E402

from reacher_data import RealReacherWindowsFromCache, collate  # noqa: E402

RAW_STEP_DT = 0.04           # seconds per env.step() (action_repeat=2 * physics_timestep 0.02)
MODEL_STEP_STEPS = 3          # frameskip used by reacher_build_cache.py
MODEL_STEP_DT = RAW_STEP_DT * MODEL_STEP_STEPS
ELBOW_LIMIT = 2.5             # stay within elbow's real joint range when sampling q0


def load_ckpt(path, device):
    state = torch.load(path, map_location=device)
    args = state["args"]
    if args.get("model") == "tijepa":
        encoder = TIJEPAEncoder(image_size=args["image_size"], q_dim=args["q_dim"], v_dim=args["v_dim"],
                                 k=args["k"], feat_dim=args["feat_dim"], backbone="vit").to(device)
        predictor = OfficialTIJEPAPredictor(q_dim=args["q_dim"], v_dim=args["v_dim"], action_dim=2).to(device)
        model_type = "tijepa"
    else:
        encoder = LeWMStyleEncoder(image_size=args["image_size"], z_dim=args["z_dim"],
                                    feat_dim=args["feat_dim"], backbone="vit").to(device)
        predictor = OfficialBaselinePredictor(per_step_dim=args["z_dim"], k=args["k"], action_dim=2,
                                               out_dim=args["z_dim"]).to(device)
        model_type = "baseline"
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval(); predictor.eval()
    return encoder, predictor, args, model_type


@torch.no_grad()
def fit_position_probe(encoder, model_type, args, device, cache_path, num_windows=3000, seed=0):
    """single-frame emb (at last context step) -> finger_pos. Eval-only decode tool."""
    ds = RealReacherWindowsFromCache(cache_path, k=args["k"], split="val", seed=0)
    n = min(num_windows, len(ds))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=128, shuffle=False,
                                          collate_fn=collate)
    k = args["k"]
    X = []
    for batch in loader:
        pixels = batch["o_ctx"].to(device)
        b = pixels.shape[0]
        if model_type == "baseline":
            z = encoder(pixels.reshape(b * k, 3, args["image_size"], args["image_size"])).reshape(b, k, -1)[:, -1]
        else:
            q_window, _, _ = encoder.forward_window(pixels)
            z = q_window[:, -1]
        X.append(z.cpu().numpy())
    X = np.concatenate(X, 0)
    # finger_pos ground truth at the last context frame (window schema index k-1);
    # not carried by __getitem__, so index the cache arrays directly, same idx order
    # the (non-shuffled) DataLoader iterated the Subset in.
    Y = np.stack([ds.finger_pos[ds.index[i] + k - 1] for i in idx], 0)
    reg = Ridge(alpha=1.0).fit(X, Y)
    print(f"[probe] finger_pos decode probe: in-sample R^2={reg.score(X, Y):.4f} (n={len(X)})", flush=True)
    return reg


def sample_config(rng):
    q0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-ELBOW_LIMIT, ELBOW_LIMIT)])
    speed = rng.uniform(1.0, 2.5, size=2)
    sign = rng.choice([-1.0, 1.0], size=2)
    v = speed * sign
    return q0, v


def build_context_frames(env, q0, v, k):
    frames = []
    for i in range(k):
        dt_back = (k - 1 - i) * MODEL_STEP_DT
        qpos_i = q0 - v * dt_back
        state = np.concatenate([qpos_i, v])
        env.reset(seed=0, options={"state": state})
        frames.append(env.render())
    return np.stack(frames, axis=0)  # (k,224,224,3) uint8


def ground_truth_rollout(env, q0, v, horizon_model_steps):
    state = np.concatenate([q0, v])
    env.reset(seed=0, options={"state": state})
    positions = [env.info["finger_pos"].copy()]
    zero_action = np.zeros(2, dtype=np.float32)
    for _ in range(horizon_model_steps):
        info = None
        for _ in range(MODEL_STEP_STEPS):
            _, _, _, _, info = env.step(zero_action)
        positions.append(info["finger_pos"].copy())
    return np.stack(positions, axis=0)  # (H+1, 2)


@torch.no_grad()
def blind_rollout(encoder, predictor, model_type, args, ctx_frames_uint8, device, horizon):
    k = args["k"]
    image_size = args["image_size"]
    pix = torch.from_numpy(ctx_frames_uint8).float().permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255.0  # (1,k,3,S,S)
    zero_action = torch.zeros(1, 2, device=device)
    if model_type == "baseline":
        z_dim = args["z_dim"]
        z_hist = encoder(pix.reshape(k, 3, image_size, image_size)).reshape(1, k, z_dim)
        traj = [z_hist[:, i] for i in range(k)]
        for _ in range(horizon):
            z_pred = predictor(torch.stack(traj[-k:], dim=1), zero_action)
            traj.append(z_pred)
        return torch.stack(traj, dim=1)[0].cpu().numpy()  # (k+horizon, z_dim)
    else:
        q_window, v_t, _ = encoder.forward_window(pix)
        q_t = q_window[0, -1:].clone()
        v_cur = v_t.clone()
        q_traj = [q_window[0, i:i + 1] for i in range(k)]
        for _ in range(horizon):
            q_hat, v_hat = predictor(q_t, v_cur, zero_action)
            q_traj.append(q_hat)
            q_t, v_cur = q_hat, v_hat
        return torch.cat(q_traj, dim=0).cpu().numpy()  # (k+horizon, q_dim)


def run_one_arm(name, ckpt_path, device, cache_path, n_pairs, horizon, seed):
    encoder, predictor, args, model_type = load_ckpt(ckpt_path, device)
    print(f"=== {name} ({model_type}) ===", flush=True)
    probe = fit_position_probe(encoder, model_type, args, device, cache_path, seed=seed)

    env_ctx = ReacherDMControlWrapper(task="hard", seed=seed)
    env_gt = ReacherDMControlWrapper(task="hard", seed=seed)
    rng = np.random.default_rng(seed)
    k = args["k"]
    records = []
    for p in range(n_pairs):
        q0, v = sample_config(rng)
        branch = {}
        for sign, v_signed in [("+v", v), ("-v", -v)]:
            ctx = build_context_frames(env_ctx, q0, v_signed, k)
            gt_pos = ground_truth_rollout(env_gt, q0, v_signed, horizon)
            emb_traj = blind_rollout(encoder, predictor, model_type, args, ctx, device, horizon)
            pos_traj = probe.predict(emb_traj)  # (k+horizon, 2)
            pred_pos = pos_traj[k - 1:]          # align index0="now", len horizon+1
            branch[sign] = {"gt": gt_pos, "pred": pred_pos}

        gt_sep = np.linalg.norm(branch["+v"]["gt"] - branch["-v"]["gt"], axis=-1)
        pred_sep = np.linalg.norm(branch["+v"]["pred"] - branch["-v"]["pred"], axis=-1)
        true_disp = branch["+v"]["gt"][-1] - branch["+v"]["gt"][0]
        pred_disp_plus = branch["+v"]["pred"][-1] - branch["+v"]["pred"][0]
        pred_disp_minus = branch["-v"]["pred"][-1] - branch["-v"]["pred"][0]
        sign_ok_plus = float(np.dot(pred_disp_plus, true_disp) > 0)
        sign_ok_minus = float(np.dot(pred_disp_minus, -true_disp) > 0)
        pos_mse_plus = float(np.mean((branch["+v"]["pred"] - branch["+v"]["gt"]) ** 2))
        pos_mse_minus = float(np.mean((branch["-v"]["pred"] - branch["-v"]["gt"]) ** 2))
        records.append({
            "gt_sep": gt_sep.tolist(), "pred_sep": pred_sep.tolist(),
            "ratio_final": float(pred_sep[-1] / (gt_sep[-1] + 1e-6)),
            "sign_ok_plus": sign_ok_plus, "sign_ok_minus": sign_ok_minus,
            "pos_mse_plus": pos_mse_plus, "pos_mse_minus": pos_mse_minus,
        })
        if (p + 1) % 10 == 0:
            print(f"  pair {p+1}/{n_pairs}", flush=True)

    gt_curve = np.mean([r["gt_sep"] for r in records], axis=0)
    pred_curve = np.mean([r["pred_sep"] for r in records], axis=0)
    ratio_mean = float(np.mean([r["ratio_final"] for r in records]))
    sign_acc = float(np.mean([r["sign_ok_plus"] for r in records] + [r["sign_ok_minus"] for r in records]))
    pos_mse_mean = float(np.mean([r["pos_mse_plus"] for r in records] + [r["pos_mse_minus"] for r in records]))
    print(f"{name}: branch_sep_ratio={ratio_mean:.4f} sign_acc={sign_acc:.4f} pos_mse={pos_mse_mean:.6f}", flush=True)
    return {"model_type": model_type, "gt_curve": gt_curve.tolist(), "pred_curve": pred_curve.tolist(),
            "ratio_mean": ratio_mean, "velocity_sign_accuracy": sign_acc, "position_mse_mean": pos_mse_mean,
            "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_k3_ckpt", required=True)
    ap.add_argument("--baseline_k1_ckpt", required=True)
    ap.add_argument("--tijepa_ckpt", required=True)
    ap.add_argument("--cache", default="cache/reacher_cache.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_pairs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args_cli = ap.parse_args()

    out = {}
    out["baseline_k3"] = run_one_arm("baseline_k3", args_cli.baseline_k3_ckpt, args_cli.device, args_cli.cache,
                                       args_cli.n_pairs, args_cli.horizon, args_cli.seed)
    out["baseline_k1"] = run_one_arm("baseline_k1", args_cli.baseline_k1_ckpt, args_cli.device, args_cli.cache,
                                       args_cli.n_pairs, args_cli.horizon, args_cli.seed)
    out["tijepa"] = run_one_arm("tijepa", args_cli.tijepa_ckpt, args_cli.device, args_cli.cache,
                                  args_cli.n_pairs, args_cli.horizon, args_cli.seed)

    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved ->", args_cli.out, flush=True)


if __name__ == "__main__":
    main()
