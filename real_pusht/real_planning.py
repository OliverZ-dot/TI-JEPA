"""Protocol C 的"legacy check"（真实 benchmark 上的下游规划，r11）：
在真实、可交互的 PushT gym 环境（pymunk 物理，非阻尼覆盖版本，即训练数据本身
采集时用的默认设置）上，从真实专家轨迹里取一个真实初始状态 + 该轨迹后续若干
步之后真实达到的方块目标位姿作为 goal，用我们自己训练的 (baseline_k1 /
baseline_k3 / tijepa_k1 / tijepa_k3) encoder+predictor pair 的纯 latent 滚动
（不碰真实物理）当 CEM 的 cost function，做闭环 MPC 把方块推向目标，看：

  (a) 这些从零训练的小模型能不能在真实、接触丰富的 PushT 上支持起码有意义的
      闭环控制（"没把正常能力搞坏"——这是本文档 PAPER_OUTLINE 里排第一优先级
      的缺口："真实 benchmark 上下游规划证据仍是零"）；
  (b) 在同样的 (q,v) vs baseline 对照下，TI-JEPA 是否仍有优势，或至少不比
      matched-memory baseline 差太多。

和 Protocol B（kill_experiment/real_eval_ours.py）的关键不同：这里**不**覆盖
`damping`，环境保持训练数据采集时的默认物理（有阻尼、有接触），是一个诚实的
"正常部署条件"下的 check，而不是刻意放大速度信号的诊断实验。

方法：
  - 目标 = 同一条真实专家轨迹里，从起点往后数 mpc_total_steps 个"model step"
    (每个 model step = 5 个真实 env.step，对齐训练时的 frameskip=5) 之后，
    专家真实达到的方块位姿 (block_x, block_y, angle)。这保证目标在给定的步数
    预算内是"人类专家证明可达的"，不是随手放一个可能物理上不可达的目标。
  - CEM 代价 = 用学到的 encoder+predictor 对候选动作序列做纯 latent 滚动，
    读出头 (linear ridge probe，在同一批真实像素上只读不训) 解码出
    (block_x,block_y,angle)，与目标的加权平方距离。
  - 5 条臂：baseline_k1 / baseline_k3(real) / tijepa_k1 / tijepa_k3(real) /
    zero_action / random。因为真实接触物理没有便宜的解析 oracle（不像
    InertiaBall/Pendulum/CartPole 可以直接写出解析动力学），这里没有
    oracle_cem 臂——这是一个诚实的局限，在 the results log 里会明确写出来。
"""

from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, str(Path(__file__).parent))

from common import LEWM_REPO  # noqa: E402
from real_eval_ours import load_ckpt, fit_position_probe_ours, MODEL_STEP_STEPS  # noqa: E402

sys.path.insert(0, str(LEWM_REPO))
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402

from ti_jepa.eval.planning import CEMPlanner  # noqa: E402

ANGLE_WEIGHT = 3000.0  # px^2 per rad^2，让 20px 位置误差和 ~20deg 角度误差同量级
SUCCESS_POS_PX = 20.0
SUCCESS_ANGLE_RAD = np.pi / 9


def wrap_angle(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


@torch.no_grad()
def cost_baseline(encoder, predictor, pos_probe, ctx_frames, k, device, action_seqs, goal, act_pen_coef):
    """ctx_frames: (k,S,S,3) float[0,1]. action_seqs: (N,H,2)."""
    N, H, _ = action_seqs.shape
    S = ctx_frames.shape[1]
    o_t = torch.from_numpy(ctx_frames).permute(0, 3, 1, 2).float().to(device)  # (k,3,S,S)
    z_hist0 = encoder(o_t)  # (k, z_dim)
    z_dim = z_hist0.shape[-1]
    window = z_hist0.unsqueeze(0).expand(N, k, z_dim).clone()
    actions = torch.from_numpy(action_seqs).float().to(device)
    for t in range(H):
        z_next = predictor(window, actions[:, t])
        window = torch.cat([window[:, 1:, :], z_next.unsqueeze(1)], dim=1)
    z_final = window[:, -1, :].cpu().numpy()
    dec = pos_probe.predict(z_final)  # (N,5) agent_x,agent_y,block_x,block_y,angle
    return _decode_cost(dec, action_seqs, goal, act_pen_coef)


@torch.no_grad()
def cost_tijepa(encoder, predictor, pos_probe, ctx_frames, k, device, action_seqs, goal, act_pen_coef):
    N, H, _ = action_seqs.shape
    o_t = torch.from_numpy(ctx_frames).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)  # (1,k,3,S,S)
    q_window0, v_t0, _ = encoder.forward_window(o_t)
    q_t = q_window0[:, -1, :].expand(N, -1).clone()
    v_t = v_t0.expand(N, -1).clone()
    actions = torch.from_numpy(action_seqs).float().to(device)
    for t in range(H):
        q_t, v_t = predictor(q_t, v_t, actions[:, t])
    q_final = q_t.cpu().numpy()
    dec = pos_probe.predict(q_final)
    return _decode_cost(dec, action_seqs, goal, act_pen_coef)


def _decode_cost(dec, action_seqs, goal, act_pen_coef):
    block_xy = dec[:, 2:4]
    angle = dec[:, 4]
    pos_d2 = np.sum((block_xy - goal[None, :2]) ** 2, axis=-1)
    ang_d2 = wrap_angle(angle - goal[2]) ** 2
    act_pen = act_pen_coef * np.sum(action_seqs ** 2, axis=(1, 2))
    return pos_d2 + ANGLE_WEIGHT * ang_d2 + act_pen


def block_goal_dist(state, goal):
    """state: 7-dim real env state. goal: (block_x,block_y,angle)."""
    pos_d = float(np.linalg.norm(state[2:4] - goal[:2]))
    ang_d = float(abs(wrap_angle(state[4] - goal[2])))
    return pos_d, ang_d


def run_mpc_episode(env, encoder, predictor, model_type, pos_probe, k, device,
                     init_state, init_ctx_frames, goal, mpc_total_steps, mpc_horizon,
                     n_samples, n_iters, a_max, act_pen_coef, seed):
    env.reset(seed=0, options={"state": init_state.tolist(), "goal_state": init_state.tolist()})
    ctx = deque(init_ctx_frames[-k:].copy(), maxlen=k)
    cur_state = init_state.copy()
    for step_i in range(mpc_total_steps):
        ctx_arr = np.stack(list(ctx), axis=0)
        planner = CEMPlanner(mpc_horizon, action_dim=2, n_samples=n_samples, n_iters=n_iters,
                              a_max=a_max, seed=seed + step_i)
        if model_type == "baseline":
            cost_fn = lambda a: cost_baseline(encoder, predictor, pos_probe, ctx_arr, k, device, a, goal, act_pen_coef)
        else:
            cost_fn = lambda a: cost_tijepa(encoder, predictor, pos_probe, ctx_arr, k, device, a, goal, act_pen_coef)
        seq, _ = planner.plan(cost_fn)
        a0 = seq[0]
        obs = None
        for _ in range(MODEL_STEP_STEPS):
            obs, *_ = env.step(a0)
        frame = env.render().astype(np.float32) / 255.0
        ctx.append(frame)
        cur_state = obs["state"]
    return cur_state


def run_mpc_episode_scalar_policy(env, policy_fn, init_state, goal, mpc_total_steps, rng):
    env.reset(seed=0, options={"state": init_state.tolist(), "goal_state": init_state.tolist()})
    cur_state = init_state.copy()
    for step_i in range(mpc_total_steps):
        a0 = policy_fn(rng)
        obs = None
        for _ in range(MODEL_STEP_STEPS):
            obs, *_ = env.step(a0)
        cur_state = obs["state"]
    return cur_state


def sample_tasks(cache, n_tasks, max_k, cap_steps, seed, val_frac=0.1, min_disp_px=30.0):
    n_ep = len(cache["ep_starts"])
    g = np.random.default_rng(0)  # MUST match real_eval_ours / RealPushTWindowsFromCache split seed=0
    perm = g.permutation(n_ep)
    n_val = max(1, int(val_frac * n_ep))
    val_eps = perm[:n_val]
    ep_lens = cache["ep_lens"]
    ep_starts = cache["ep_starts"]
    state = cache["state"]
    usable = []
    for e in val_eps:
        L = int(ep_lens[e])
        if L - max_k < 3:
            continue
        s = int(ep_starts[e])
        # only keep episodes where the human expert actually moved the block a
        # meaningful amount by the end -- otherwise the "task" is vacuous (no
        # policy, however good, could do anything different from doing nothing).
        disp = np.linalg.norm(state[s + L - 1, 2:4] - state[s + max_k - 1, 2:4])
        if disp >= min_disp_px:
            usable.append(e)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(usable, size=min(n_tasks, len(usable)), replace=False)
    tasks = []
    for ep in chosen:
        start, L = int(cache["ep_starts"][ep]), int(cache["ep_lens"][ep])
        n_steps = min(cap_steps, L - max_k)
        goal_idx = max_k - 1 + n_steps
        init_state = cache["state"][start: start + max_k].astype(np.float64)  # (max_k,7)
        init_pixels = cache["pixels"][start: start + max_k].astype(np.float32) / 255.0
        goal_state = cache["state"][start + goal_idx].astype(np.float64)
        tasks.append({
            "ep": int(ep), "n_steps": n_steps,
            "init_state": init_state[-1],  # state at last context frame == "now"
            "init_ctx": init_pixels,
            "goal": goal_state[2:5].copy(),  # block_x, block_y, angle
        })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/pusht_cache.npz")
    ap.add_argument("--baseline_k1", default="checkpoints_k1/baseline_k1.pt")
    ap.add_argument("--baseline_k3", default="checkpoints/baseline_real.pt")
    ap.add_argument("--tijepa_k1", default="checkpoints_k1/tijepa_k1.pt")
    ap.add_argument("--tijepa_k3", default="checkpoints/tijepa_real.pt")
    ap.add_argument("--n_tasks", type=int, default=40)
    ap.add_argument("--cap_steps", type=int, default=10)
    ap.add_argument("--mpc_horizon", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=150)
    ap.add_argument("--n_iters", type=int, default=5)
    ap.add_argument("--a_max", type=float, default=1.0)
    ap.add_argument("--act_pen_coef", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/real_planning_legacy_check.json")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    cache = np.load(args.cache)
    resolution = int(cache["image_size"])

    ckpts = {
        "baseline_k1": args.baseline_k1,
        "baseline_k3": args.baseline_k3,
        "tijepa_k1": args.tijepa_k1,
        "tijepa_k3": args.tijepa_k3,
    }
    models = {}
    max_k = 1
    for name, path in ckpts.items():
        enc, pred, cargs, model_type = load_ckpt(path, device)
        print(f"loaded {name} ({model_type}, k={cargs['k']}) from {path}", flush=True)
        pos_probe = fit_position_probe_ours(enc, model_type, cargs, device, args.cache)
        models[name] = {"encoder": enc, "predictor": pred, "args": cargs, "model_type": model_type, "probe": pos_probe}
        max_k = max(max_k, cargs["k"])

    tasks = sample_tasks(cache, args.n_tasks, max_k, args.cap_steps, args.seed)
    print(f"sampled {len(tasks)} tasks (val-episode, cap_steps={args.cap_steps})", flush=True)

    env = PushT(render_mode="rgb_array", resolution=resolution)  # NO damping override: legacy/default physics

    arm_names = list(ckpts.keys()) + ["zero_action", "random"]
    results = {a: {"final_pos_d": [], "final_ang_d": [], "init_pos_d": [], "init_ang_d": [],
                    "success": [], "n_steps": []} for a in arm_names}

    rng_global = np.random.default_rng(args.seed + 7777)

    for ti, task in enumerate(tasks):
        init_state, init_ctx, goal, n_steps = task["init_state"], task["init_ctx"], task["goal"], task["n_steps"]
        init_pos_d, init_ang_d = block_goal_dist(init_state, goal)

        for name in ckpts.keys():
            m = models[name]
            k = m["args"]["k"]
            final_state = run_mpc_episode(
                env, m["encoder"], m["predictor"], m["model_type"], m["probe"], k, device,
                init_state, init_ctx, goal, n_steps, args.mpc_horizon,
                args.n_samples, args.n_iters, args.a_max, args.act_pen_coef,
                seed=args.seed + ti * 97 + hash(name) % 1000)
            pd, ad = block_goal_dist(final_state, goal)
            results[name]["final_pos_d"].append(pd)
            results[name]["final_ang_d"].append(ad)
            results[name]["init_pos_d"].append(init_pos_d)
            results[name]["init_ang_d"].append(init_ang_d)
            results[name]["success"].append(bool(pd < SUCCESS_POS_PX and ad < SUCCESS_ANGLE_RAD))
            results[name]["n_steps"].append(n_steps)

        rng_zero = np.random.default_rng(args.seed + 1 + ti)
        final_state = run_mpc_episode_scalar_policy(env, lambda r: np.zeros(2, dtype=np.float32),
                                                      init_state, goal, n_steps, rng_zero)
        pd, ad = block_goal_dist(final_state, goal)
        results["zero_action"]["final_pos_d"].append(pd)
        results["zero_action"]["final_ang_d"].append(ad)
        results["zero_action"]["init_pos_d"].append(init_pos_d)
        results["zero_action"]["init_ang_d"].append(init_ang_d)
        results["zero_action"]["success"].append(bool(pd < SUCCESS_POS_PX and ad < SUCCESS_ANGLE_RAD))
        results["zero_action"]["n_steps"].append(n_steps)

        rng_rand = np.random.default_rng(args.seed + 2 + ti)
        rand_policy = lambda r: np.clip(r.normal(0, args.a_max / 2, size=2), -args.a_max, args.a_max).astype(np.float32)
        final_state = run_mpc_episode_scalar_policy(env, rand_policy, init_state, goal, n_steps, rng_rand)
        pd, ad = block_goal_dist(final_state, goal)
        results["random"]["final_pos_d"].append(pd)
        results["random"]["final_ang_d"].append(ad)
        results["random"]["init_pos_d"].append(init_pos_d)
        results["random"]["init_ang_d"].append(init_ang_d)
        results["random"]["success"].append(bool(pd < SUCCESS_POS_PX and ad < SUCCESS_ANGLE_RAD))
        results["random"]["n_steps"].append(n_steps)

        print(f"  task {ti+1}/{len(tasks)} (ep={task['ep']}, n_steps={n_steps}) init_pos_d={init_pos_d:.1f} "
              + " ".join(f"{a}={results[a]['final_pos_d'][-1]:.1f}" for a in arm_names), flush=True)

    summary = {}
    for name, r in results.items():
        init_d = np.array(r["init_pos_d"])
        fin_d = np.array(r["final_pos_d"])
        progress = (init_d - fin_d) / np.maximum(init_d, 1e-6)
        summary[name] = {
            "mean_final_pos_d": float(np.mean(fin_d)),
            "mean_final_ang_d": float(np.mean(r["final_ang_d"])),
            "mean_init_pos_d": float(np.mean(init_d)),
            "success_rate": float(np.mean(r["success"])),
            "mean_improvement_px": float(np.mean(init_d - fin_d)),
            "mean_progress_frac": float(np.mean(progress)),
            "n": len(r["final_pos_d"]),
        }

    # paired significance: matched-memory comparisons
    def paired_test(a, b):
        da = np.array(results[a]["final_pos_d"])
        db = np.array(results[b]["final_pos_d"])
        try:
            stat, p = sstats.wilcoxon(da, db)
        except ValueError:
            p = float("nan")
        return {"a_closer_frac": float(np.mean(da < db)), "wilcoxon_p": float(p),
                "mean_diff_b_minus_a": float(np.mean(db - da))}

    sig = {
        "tijepa_k1_vs_baseline_k1": paired_test("tijepa_k1", "baseline_k1"),
        "tijepa_k3_vs_baseline_k3": paired_test("tijepa_k3", "baseline_k3"),
        "tijepa_k1_vs_zero": paired_test("tijepa_k1", "zero_action"),
        "baseline_k1_vs_zero": paired_test("baseline_k1", "zero_action"),
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== SIGNIFICANCE ===")
    print(json.dumps(sig, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "significance": sig, "results": results,
                   "config": vars(args)}, f, indent=2)
    print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
