"""Protocol A + B for OUR OWN matched-budget checkpoints (baseline / TI-JEPA)
trained on real PushT pixels by real_train.py. Mirrors probe_ladder.py /
kill_experiment.py but swaps the official JEPA model for our nn.Module pair
and the ImageNet-normalized ViT preprocessing for the plain [0,1] CNN
preprocessing real_train.py uses.
"""

from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

sys.path.insert(0, _REPO_ROOT)
from ti_jepa.models import LeWMStyleEncoder, HistoryPredictor, TIJEPAEncoder, TIJEPAPredictor

from common import LEWM_REPO, pearsonr_per_dim
from real_data import RealPushTWindowsFromCache

sys.path.insert(0, str(LEWM_REPO))
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402

ALPHAS = np.logspace(-3, 4, 15)
STEP_DT = 0.1
MODEL_STEP_STEPS = 5
MODEL_STEP_DT = STEP_DT * MODEL_STEP_STEPS
AGENT_PARK = (40.0, 40.0)
BLOCK_MARGIN = 90.0
WINDOW = 512.0


def load_ckpt(path, device):
    state = torch.load(path, map_location=device)
    args = state["args"]
    if args.get("model") == "tijepa":
        encoder = TIJEPAEncoder(image_size=args["image_size"], q_dim=args["q_dim"], v_dim=args["v_dim"], k=args["k"]).to(device)
        predictor = TIJEPAPredictor(q_dim=args["q_dim"], v_dim=args["v_dim"], action_dim=2).to(device)
        model_type = "tijepa"
    else:
        encoder = LeWMStyleEncoder(image_size=args["image_size"], z_dim=args["z_dim"]).to(device)
        predictor = HistoryPredictor(per_step_dim=args["z_dim"], k=args["k"], action_dim=2, out_dim=args["z_dim"]).to(device)
        model_type = "baseline"
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval(); predictor.eval()
    return encoder, predictor, args, model_type


def fit_probe_cv(X, Y, train_mask, val_mask, seed):
    Xtr, Ytr, Xval, Yval = X[train_mask], Y[train_mask], X[val_mask], Y[val_mask]
    preds = []
    for d in range(Y.shape[-1]):
        r = RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr[:, d])
        preds.append(r.predict(Xval))
    pred = np.stack(preds, axis=-1)
    r2 = r2_score(Yval, pred, multioutput="raw_values")
    r = pearsonr_per_dim(pred, Yval)
    g = np.random.default_rng(seed + 999)
    shuf = g.permutation(len(Yval))
    r2_shuf = r2_score(Yval[shuf], pred, multioutput="raw_values")
    return {"r2_mean": float(np.mean(r2)), "pearson_r_mean": float(np.mean(np.abs(r))),
            "r2_mean_SHUFFLED_CONTROL": float(np.mean(r2_shuf)), "n": int(len(Yval))}


@torch.no_grad()
def protocol_a(encoder, model_type, args, device, cache_path, seed=0):
    ds = RealPushTWindowsFromCache(cache_path, k=args["k"], split="val", seed=0)
    print(f"protocol A: {len(ds)} held-out windows", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                          collate_fn=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]})

    z_single, z_pair, z_window, v_explicit, states, ep_of = [], [], [], [], [], []
    k = args["k"]
    for wi, batch in enumerate(loader):
        pixels = batch["pixels"].to(device)  # (B,k+1,3,S,S)
        o_ctx = pixels[:, :k]
        b = o_ctx.shape[0]
        if model_type == "baseline":
            z_ctx = encoder(o_ctx.reshape(b * k, 3, args["image_size"], args["image_size"])).reshape(b, k, -1)
            z_single.append(z_ctx[:, -1].cpu().numpy())
            z_pair.append(z_ctx[:, -2:].reshape(b, -1).cpu().numpy())
            z_window.append(z_ctx.reshape(b, -1).cpu().numpy())
        else:
            q_window, v_t, _ = encoder.forward_window(o_ctx)
            z_single.append(q_window[:, -1].cpu().numpy())
            z_pair.append(q_window[:, -2:].reshape(b, -1).cpu().numpy())
            z_window.append(q_window.reshape(b, -1).cpu().numpy())
            v_explicit.append(v_t.cpu().numpy())
        states.append(batch["state"].numpy())

    z_single, z_pair, z_window = map(lambda x: np.concatenate(x, 0), (z_single, z_pair, z_window))
    states = np.concatenate(states, 0)  # (N,k+1,7)
    block_vel = states[:, k, 2:4] - states[:, k - 1, 2:4]
    agent_vel = states[:, k - 1, 5:7]

    n = len(z_single)
    ep_ids = np.array(ds.window_episode_ids)
    uniq_eps = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    perm_eps = rng.permutation(uniq_eps)
    n_val_eps = max(1, int(0.3 * len(uniq_eps)))
    val_eps = set(perm_eps[:n_val_eps].tolist())
    val_mask = np.isin(ep_ids, list(val_eps))
    train_mask = ~val_mask

    rows = {"single": z_single, "pair": z_pair, "window_k3": z_window}
    if model_type == "tijepa":
        rows["v_explicit"] = np.concatenate(v_explicit, 0)

    results = {}
    for name, X in rows.items():
        r_block = fit_probe_cv(X, block_vel, train_mask, val_mask, seed)
        r_agent = fit_probe_cv(X, agent_vel, train_mask, val_mask, seed)
        results[name] = {"block_vel": r_block, "agent_vel": r_agent}
        print(f"  feat={name:12s} block_vel r2={r_block['r2_mean']:+.4f}(shuf={r_block['r2_mean_SHUFFLED_CONTROL']:+.4f}) "
              f"agent_vel r2={r_agent['r2_mean']:+.4f}(shuf={r_agent['r2_mean_SHUFFLED_CONTROL']:+.4f})", flush=True)
    return results


def fit_position_probe_ours(encoder, model_type, args, device, cache_path, seed=0):
    ds = RealPushTWindowsFromCache(cache_path, k=args["k"], split="val", seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                          collate_fn=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]})
    k = args["k"]
    X, Y = [], []
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"].to(device)
            o_ctx = pixels[:, :k]
            b = o_ctx.shape[0]
            if model_type == "baseline":
                z = encoder(o_ctx.reshape(b * k, 3, args["image_size"], args["image_size"])).reshape(b, k, -1)[:, -1]
            else:
                q_window, _, _ = encoder.forward_window(o_ctx)
                z = q_window[:, -1]
            X.append(z.cpu().numpy())
            Y.append(batch["state"][:, k - 1, :5].numpy())
    X, Y = np.concatenate(X), np.concatenate(Y)
    reg = Ridge(alpha=1.0).fit(X, Y)
    print(f"position decode probe (ours, {model_type}): in-sample R^2={reg.score(X, Y):.4f} (n={len(X)})", flush=True)
    return reg


def sample_config(rng):
    bx = rng.uniform(BLOCK_MARGIN, WINDOW - BLOCK_MARGIN)
    by = rng.uniform(BLOCK_MARGIN, WINDOW - BLOCK_MARGIN)
    angle = rng.uniform(0, 2 * np.pi)
    speed = rng.uniform(40.0, 90.0)
    theta = rng.uniform(0, 2 * np.pi)
    v = np.array([speed * np.cos(theta), speed * np.sin(theta)])
    return bx, by, angle, v


def render_resized(env, image_size):
    img = env.render()
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = torch.nn.functional.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).numpy()  # (S,S,3) float in [0,1] -- keep float for direct model feed


def build_context_frames(env, bx, by, angle, v, k, image_size):
    frames = []
    for i in range(k):
        dt_back = (k - 1 - i) * MODEL_STEP_DT
        pos = np.array([bx, by]) - v * dt_back
        state = [AGENT_PARK[0], AGENT_PARK[1], pos[0], pos[1], angle, 0.0, 0.0]
        env.reset(seed=0, options={"state": state, "goal_state": state})
        frames.append(render_resized(env, image_size))
    return np.stack(frames, 0)  # (k,S,S,3) float


def ground_truth_rollout(env, bx, by, angle, v, horizon_model_steps):
    state = [AGENT_PARK[0], AGENT_PARK[1], bx, by, angle, 0.0, 0.0]
    env.reset(seed=0, options={"state": state, "goal_state": state})
    env.block.velocity = tuple(v.tolist())
    positions = [np.array([bx, by])]
    zero_action = np.zeros(2, dtype=np.float32)
    for _ in range(horizon_model_steps):
        for _ in range(MODEL_STEP_STEPS):
            obs, *_ = env.step(zero_action)
        positions.append(obs["state"][2:4].copy())
    return np.stack(positions, 0)


@torch.no_grad()
def blind_rollout_ours(encoder, predictor, model_type, args, ctx_frames, device, horizon):
    k = args["k"]
    image_size = args["image_size"]
    pix = torch.from_numpy(ctx_frames).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)  # (1,k,3,S,S)
    zero_action = torch.zeros(1, 2, device=device)
    if model_type == "baseline":
        z_dim = args["z_dim"]
        z_hist = encoder(pix.reshape(k, 3, image_size, image_size)).reshape(1, k, z_dim)
        traj = [z_hist[:, i] for i in range(k)]
        for _ in range(horizon):
            z_pred = predictor(torch.stack(traj[-k:], dim=1), zero_action)
            traj.append(z_pred)
        return torch.stack(traj, dim=1)[0].cpu().numpy()  # (k+horizon, z_dim) -- decode all via same position probe
    else:
        q_window, v_t, _ = encoder.forward_window(pix)
        q_t = q_window[0, -1:].clone()
        v_cur = v_t.clone()
        q_traj = [q_window[0, i:i+1] for i in range(k)]
        for _ in range(horizon):
            q_hat, v_hat = predictor(q_t, v_cur, zero_action)
            q_traj.append(q_hat)
            q_t, v_cur = q_hat, v_hat
        return torch.cat(q_traj, dim=0).cpu().numpy()  # (k+horizon, q_dim)


def protocol_b(encoder, predictor, model_type, args, device, cache_path, n_pairs=40, horizon=10, seed=0):
    probe = fit_position_probe_ours(encoder, model_type, args, device, cache_path)
    env_ctx = PushT(render_mode="rgb_array", resolution=224, damping=1.0)
    env_gt = PushT(render_mode="rgb_array", resolution=224, damping=1.0)
    rng = np.random.default_rng(seed)
    k = args["k"]
    records = []
    for p in range(n_pairs):
        bx, by, angle, v = sample_config(rng)
        branch = {}
        for sign, v_signed in [("+v", v), ("-v", -v)]:
            ctx = build_context_frames(env_ctx, bx, by, angle, v_signed, k, args["image_size"])
            gt_pos = ground_truth_rollout(env_gt, bx, by, angle, v_signed, horizon)
            z_traj = blind_rollout_ours(encoder, predictor, model_type, args, ctx, device, horizon)
            pos_traj = probe.predict(z_traj)[:, 2:4]
            pred_pos = pos_traj[k - 1:]
            branch[sign] = {"gt": gt_pos, "pred": pred_pos}
        gt_sep = np.linalg.norm(branch["+v"]["gt"] - branch["-v"]["gt"], axis=-1)
        pred_sep = np.linalg.norm(branch["+v"]["pred"] - branch["-v"]["pred"], axis=-1)
        true_disp = v * horizon * MODEL_STEP_DT
        pred_disp_plus = branch["+v"]["pred"][-1] - branch["+v"]["pred"][0]
        pred_disp_minus = branch["-v"]["pred"][-1] - branch["-v"]["pred"][0]
        sign_ok_plus = float(np.dot(pred_disp_plus, true_disp) > 0)
        sign_ok_minus = float(np.dot(pred_disp_minus, -true_disp) > 0)
        pos_mse_plus = float(np.mean((branch["+v"]["pred"] - branch["+v"]["gt"]) ** 2))
        pos_mse_minus = float(np.mean((branch["-v"]["pred"] - branch["-v"]["gt"]) ** 2))
        records.append({"gt_sep": gt_sep.tolist(), "pred_sep": pred_sep.tolist(),
                        "ratio_final": float(pred_sep[-1] / (gt_sep[-1] + 1e-6)),
                        "sign_ok_plus": sign_ok_plus, "sign_ok_minus": sign_ok_minus,
                        "pos_mse_plus": pos_mse_plus, "pos_mse_minus": pos_mse_minus})
        if (p + 1) % 10 == 0:
            print(f"  pair {p+1}/{n_pairs}", flush=True)
    gt_curve = np.mean([r["gt_sep"] for r in records], axis=0)
    pred_curve = np.mean([r["pred_sep"] for r in records], axis=0)
    ratio_mean = float(np.mean([r["ratio_final"] for r in records]))
    sign_acc = float(np.mean([r["sign_ok_plus"] for r in records] + [r["sign_ok_minus"] for r in records]))
    pos_mse_mean = float(np.mean([r["pos_mse_plus"] for r in records] + [r["pos_mse_minus"] for r in records]))
    print(f"protocol B ({model_type}): branch_separation_ratio={ratio_mean:.4f} "
          f"sign_acc={sign_acc:.4f} pos_mse={pos_mse_mean:.1f}", flush=True)
    return {"gt_curve": gt_curve.tolist(), "pred_curve": pred_curve.tolist(), "ratio_mean": ratio_mean,
            "velocity_sign_accuracy": sign_acc, "position_mse_mean": pos_mse_mean, "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache/pusht_cache.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_pairs", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", required=True)
    args_cli = ap.parse_args()

    encoder, predictor, args, model_type = load_ckpt(args_cli.ckpt, args_cli.device)
    print(f"loaded {model_type} from {args_cli.ckpt}, args={args}", flush=True)

    res_a = protocol_a(encoder, model_type, args, args_cli.device, args_cli.cache)
    res_b = protocol_b(encoder, predictor, model_type, args, args_cli.device, args_cli.cache,
                        n_pairs=args_cli.n_pairs, horizon=args_cli.horizon)

    Path(args_cli.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args_cli.out, "w") as f:
        json.dump({"model_type": model_type, "protocol_a": res_a, "protocol_b": res_b}, f, indent=2)
    print("saved ->", args_cli.out, flush=True)


if __name__ == "__main__":
    main()
