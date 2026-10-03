"""协议 C（说明书 §4.3, §5.2 末尾 "成功: 停在目标"）：必须带速度才成功的规划。

任务："stop-at-goal" —— 系统当前有构型 q_t 和（模型不可直接读取的）速度 v_t，
CEM 必须在 H 步内选一串动作把系统送到目标 g 并尽量刹停。这在很大程度上
依赖对**当前速度**的准确估计：不知道 v_t，就不知道要刹多少车。这是仅靠
单帧位置（pose）原则上无法解出、必须用到运动信息的下游任务 —— 用来验证
identifiability gap 是否真的影响下游（而不仅仅是探针/盲滚这两个诊断本身）。

四个对照臂：
  - oracle_cem   : CEM 直接用真实物理 (env_utils.oracle_dynamics_batch) 当
                   "世界模型"评估候选动作序列——不经过任何学到的表征，是本
                   任务在给定动作空间/horizon 下的可达上界。
  - baseline_cem : CEM 用官方式 baseline (LeWMStyleEncoder+HistoryPredictor,
                   k=3 历史 predictor) 的纯 latent 滚动当世界模型评估候选，
                   decode 用同一个位置探针（不训练、只读出）。
  - tijepa_cem   : 同上，换成 TI-JEPA (q,v) 无记忆 predictor。
  - zero_action  : 什么都不做，衡量任务本身的难度/基线。
  - random       : 均匀随机动作序列，下界对照。

规划是 open-loop（单次 CEM 之后盲执行整段序列，不重新观测），也支持
closed-loop MPC（--closed_loop）。本文件对三个环境（InertiaBall/Pendulum/
CartPole）通用，用 --env 切换。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ti_jepa.data import EpisodeBatch, cache_path, split_episodes
from ti_jepa.envs.registry import get_env_spec, make_scaled_config
from ti_jepa.eval.common import load_baseline, load_tijepa, to_tensor_frames
from ti_jepa.eval.env_utils import build_context_generic, oracle_dynamics_batch, sample_task


class CEMPlanner:
    """标准 CEM：候选动作序列 (H,action_dim)，按 cost 排序取 elite，重估 (mean,std)。"""

    def __init__(self, horizon, action_dim, n_samples=256, n_iters=6, elite_frac=0.1,
                 a_max=0.1, init_std=0.05, seed=0):
        self.H = horizon
        self.action_dim = action_dim
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.n_elite = max(2, int(n_samples * elite_frac))
        self.a_max = a_max
        self.init_std = init_std
        self.rng = np.random.default_rng(seed)

    def plan(self, cost_fn):
        """cost_fn: (N,H,action_dim) ndarray -> (N,) ndarray of costs (lower is better)."""
        mean = np.zeros((self.H, self.action_dim), dtype=np.float32)
        std = np.full((self.H, self.action_dim), self.init_std, dtype=np.float32)
        best_seq, best_cost = mean.copy(), np.inf
        for _ in range(self.n_iters):
            samples = mean[None] + std[None] * self.rng.standard_normal((self.n_samples, self.H, self.action_dim))
            samples = np.clip(samples, -self.a_max, self.a_max).astype(np.float32)
            costs = cost_fn(samples)
            order = np.argsort(costs)
            elite = samples[order[: self.n_elite]]
            if costs[order[0]] < best_cost:
                best_cost = float(costs[order[0]])
                best_seq = samples[order[0]].copy()
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-3
        return best_seq, best_cost


@torch.no_grad()
def model_rollout_cost_baseline(encoder, predictor, pos_probe, o_ctx, k, device, action_seqs, goal,
                                 act_pen_coef=1e-3):
    """action_seqs: (N,H,action_dim) -> (N,) 最终位置到 goal 的平方距离 + 小动作惩罚。"""
    N, H, _ = action_seqs.shape
    img_s = o_ctx.shape[1]
    o_t = to_tensor_frames(o_ctx, device).unsqueeze(0)  # (1,k,3,H,W)
    z_hist0 = encoder(o_t.reshape(k, 3, img_s, img_s)).unsqueeze(0)  # (1,k,z_dim)
    z_dim = z_hist0.shape[-1]
    window = z_hist0.expand(N, k, z_dim).clone()
    actions = torch.from_numpy(action_seqs).to(device)  # (N,H,action_dim)
    for t in range(H):
        z_next = predictor(window, actions[:, t])
        window = torch.cat([window[:, 1:, :], z_next.unsqueeze(1)], dim=1)
    z_final = window[:, -1, :].cpu().numpy()
    pos_final = pos_probe.predict(z_final)
    dist2 = np.sum((pos_final - goal[None, :]) ** 2, axis=-1)
    act_pen = act_pen_coef * np.sum(action_seqs ** 2, axis=(1, 2))
    return dist2 + act_pen


@torch.no_grad()
def model_rollout_cost_tijepa(encoder, predictor, pos_probe, o_ctx, k, device, action_seqs, goal,
                               act_pen_coef=1e-3):
    N, H, _ = action_seqs.shape
    o_t = to_tensor_frames(o_ctx, device).unsqueeze(0)  # (1,k,3,H,W)
    q_window0, v_t0, _ = encoder.forward_window(o_t)
    q_t = q_window0[:, -1, :].expand(N, -1).clone()
    v_t = v_t0.expand(N, -1).clone()
    actions = torch.from_numpy(action_seqs).to(device)
    for t in range(H):
        q_t, v_t = predictor(q_t, v_t, actions[:, t])
    q_final = q_t.cpu().numpy()
    pos_final = pos_probe.predict(q_final)
    dist2 = np.sum((pos_final - goal[None, :]) ** 2, axis=-1)
    act_pen = act_pen_coef * np.sum(action_seqs ** 2, axis=(1, 2))
    return dist2 + act_pen


def oracle_cost(env_name, env_cfg, q0, v0, action_seqs, goal, act_pen_coef=1e-3):
    pos_final, _ = oracle_dynamics_batch(env_name, env_cfg, q0, v0, action_seqs)
    dists = np.sum((pos_final - goal[None, :]) ** 2, axis=-1)
    act_pen = act_pen_coef * np.sum(action_seqs ** 2, axis=(1, 2))
    return dists + act_pen


def run_closed_loop_mpc(env_name, env, env_cfg, q0, v0, goal, cost_fn_builder, k, horizon,
                         total_steps, n_samples, n_iters, seed, action_dim, a_max=0.1):
    """MPC: at every real step, rebuild the k-frame context from the TRUE current
    state, replan an H-step CEM sequence, execute only the first action on the
    real simulator, then repeat."""
    env.set_state(q0, v0)
    for step_i in range(total_steps):
        q_now, v_now = env.get_state()
        ctx = build_context_generic(env, q_now, v_now, k, action_dim)
        env.set_state(q_now, v_now)  # build_context's teleports mutate env state; restore
        planner = CEMPlanner(horizon, action_dim, n_samples=n_samples, n_iters=n_iters,
                              a_max=a_max, seed=seed + step_i)
        cost_fn = cost_fn_builder(q_now, v_now, ctx, goal)
        seq, _ = planner.plan(cost_fn)
        env.step(seq[0])
    pos_final, _ = env.get_state()
    return float(np.linalg.norm(pos_final - goal))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["inertia_ball", "pendulum", "cartpole"], default="inertia_ball")
    ap.add_argument("--baseline_ckpt", type=str, default="checkpoints/baseline.pt")
    ap.add_argument("--tijepa_ckpt", type=str, default="checkpoints/tijepa.pt")
    ap.add_argument("--baseline_rnn_ckpt", type=str, default=None,
                     help="可选：RSSM/Dreamer风格递归聚合器baseline，加一个 rnn_cem/rnn_mpc 对照臂")
    ap.add_argument("--n_tasks", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--n_iters", type=int, default=6)
    ap.add_argument("--success_thresh", type=float, default=0.05)
    ap.add_argument("--mpc_horizon", type=int, default=4)
    ap.add_argument("--mpc_total_steps", type=int, default=10)
    ap.add_argument("--a_max", type=float, default=0.1,
                     help="每步动作幅度上限；调小可让任务更依赖利用已有速度而非暴力纠偏")
    ap.add_argument("--act_pen_coef", type=float, default=1e-3,
                     help="动作幅度惩罚系数；调大可抑制暴力大动作策略")
    ap.add_argument("--speed_lo", type=float, default=0.06)
    ap.add_argument("--speed_hi", type=float, default=0.14)
    ap.add_argument("--goal_radius_lo", type=float, default=0.15)
    ap.add_argument("--goal_radius_hi", type=float, default=0.35)
    ap.add_argument("--out_json", type=str, default="results/planning_cem.json")
    ap.add_argument("--closed_loop", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = get_env_spec(args.env)

    print("loading models + fitting readout-only position probes ...", flush=True)
    from ti_jepa.eval.kill_experiment import fit_position_probe, fit_velocity_probe
    b_enc, b_pred, b_args = load_baseline(args.baseline_ckpt, device)
    t_enc, t_pred, t_args = load_tijepa(args.tijepa_ckpt, device)

    image_size = b_args.get("image_size", 64)
    assert t_args.get("image_size", 64) == image_size, "baseline / tijepa 的 image_size 不一致"

    has_rnn = args.baseline_rnn_ckpt is not None
    if has_rnn:
        r_enc, r_pred, r_args = load_baseline(args.baseline_rnn_ckpt, device)
        assert r_args.get("image_size", 64) == image_size, "baseline_rnn 的 image_size 不一致"

    d = np.load(cache_path(args.env, image_size))
    full_batch = EpisodeBatch(d["frames"], d["actions"], d["positions"], d["velocities"])
    _, held_out = split_episodes(full_batch, int(d["n_train"]))

    b_pos_probe = fit_position_probe("baseline", b_enc, b_args["k"], device, held_out)
    t_pos_probe = fit_position_probe("tijepa", t_enc, t_args["k"], device, held_out)
    _ = fit_velocity_probe(t_enc, t_args["k"], device, held_out)
    if has_rnn:
        r_pos_probe = fit_position_probe("baseline", r_enc, r_args["k"], device, held_out)

    kb, kt = b_args["k"], t_args["k"]
    kr = r_args["k"] if has_rnn else 0
    action_dim = spec.action_dim
    env_cfg = make_scaled_config(args.env, image_size)
    env = spec.env_cls(env_cfg)
    max_k = max(kb, kt, kr)

    rng = np.random.default_rng(args.seed)
    arm_names = ["oracle_cem", "baseline_cem", "tijepa_cem", "zero_action", "random"]
    if has_rnn:
        arm_names.append("rnn_cem")
    results = {name: {"final_dist": [], "success": []} for name in arm_names}
    tasks = []

    for task_i in range(args.n_tasks):
        q0, v0, goal = sample_task(args.env, rng, env_cfg, max_k, args.horizon,
                                    speed_range=(args.speed_lo, args.speed_hi),
                                    goal_radius=(args.goal_radius_lo, args.goal_radius_hi))
        tasks.append((q0, v0, goal))
        ctx_full = build_context_generic(env, q0, v0, max_k, action_dim)
        ctx_b, ctx_t = ctx_full[-kb:], ctx_full[-kt:]
        if has_rnn:
            ctx_r = ctx_full[-kr:]

        planner_seed = int(rng.integers(0, 2**31 - 1))

        planner = CEMPlanner(args.horizon, action_dim, n_samples=args.n_samples, n_iters=args.n_iters,
                              a_max=args.a_max, seed=planner_seed)
        seq_oracle, _ = planner.plan(
            lambda a: oracle_cost(args.env, env_cfg, q0, v0, a, goal, args.act_pen_coef))

        planner = CEMPlanner(args.horizon, action_dim, n_samples=args.n_samples, n_iters=args.n_iters,
                              a_max=args.a_max, seed=planner_seed)
        seq_baseline, _ = planner.plan(
            lambda a: model_rollout_cost_baseline(b_enc, b_pred, b_pos_probe, ctx_b, kb, device, a, goal,
                                                   args.act_pen_coef))

        planner = CEMPlanner(args.horizon, action_dim, n_samples=args.n_samples, n_iters=args.n_iters,
                              a_max=args.a_max, seed=planner_seed)
        seq_tijepa, _ = planner.plan(
            lambda a: model_rollout_cost_tijepa(t_enc, t_pred, t_pos_probe, ctx_t, kt, device, a, goal,
                                                 args.act_pen_coef))

        arm_seqs = [("oracle_cem", seq_oracle), ("baseline_cem", seq_baseline), ("tijepa_cem", seq_tijepa)]

        if has_rnn:
            planner = CEMPlanner(args.horizon, action_dim, n_samples=args.n_samples, n_iters=args.n_iters,
                                  a_max=args.a_max, seed=planner_seed)
            seq_rnn, _ = planner.plan(
                lambda a: model_rollout_cost_baseline(r_enc, r_pred, r_pos_probe, ctx_r, kr, device, a, goal,
                                                       args.act_pen_coef))
            arm_seqs.append(("rnn_cem", seq_rnn))

        seq_zero = np.zeros((args.horizon, action_dim), dtype=np.float32)
        seq_random = np.clip(rng.normal(0, args.a_max / 2, size=(args.horizon, action_dim)),
                              -args.a_max, args.a_max).astype(np.float32)
        arm_seqs += [("zero_action", seq_zero), ("random", seq_random)]

        for name, seq in arm_seqs:
            env.set_state(q0, v0)
            env.rollout_open_loop(seq)
            pos_final, _ = env.get_state()
            dist = float(np.linalg.norm(pos_final - goal))
            results[name]["final_dist"].append(dist)
            results[name]["success"].append(bool(dist < args.success_thresh))

        if (task_i + 1) % 10 == 0:
            print(f"  task {task_i+1}/{args.n_tasks}", flush=True)

    summary = {}
    for name, r in results.items():
        summary[name] = {
            "mean_final_dist": float(np.mean(r["final_dist"])),
            "success_rate": float(np.mean(r["success"])),
            "n": len(r["final_dist"]),
        }
    summary["config"] = {"env": args.env, "n_tasks": args.n_tasks, "horizon": args.horizon,
                          "n_samples": args.n_samples, "n_iters": args.n_iters,
                          "success_thresh": args.success_thresh, "a_max": args.a_max}
    print("open-loop:", json.dumps(summary, indent=2), flush=True)

    if args.closed_loop:
        mpc_arm_names = ["oracle_mpc", "baseline_mpc", "tijepa_mpc"]
        if has_rnn:
            mpc_arm_names.append("rnn_mpc")
        mpc_results = {name: {"final_dist": [], "success": []} for name in mpc_arm_names}
        for task_i, (q0, v0, goal) in enumerate(tasks):
            seed_i = args.seed + 1000 + task_i
            d_oracle = run_closed_loop_mpc(
                args.env, env, env_cfg, q0, v0, goal,
                lambda qn, vn, ctx, g: (lambda a: oracle_cost(args.env, env_cfg, qn, vn, a, g, args.act_pen_coef)),
                max_k, args.mpc_horizon, args.mpc_total_steps, args.n_samples, args.n_iters, seed_i,
                action_dim, a_max=args.a_max)
            d_baseline = run_closed_loop_mpc(
                args.env, env, env_cfg, q0, v0, goal,
                lambda qn, vn, ctx, g: (lambda a: model_rollout_cost_baseline(
                    b_enc, b_pred, b_pos_probe, ctx[-kb:], kb, device, a, g, args.act_pen_coef)),
                max_k, args.mpc_horizon, args.mpc_total_steps, args.n_samples, args.n_iters, seed_i,
                action_dim, a_max=args.a_max)
            d_tijepa = run_closed_loop_mpc(
                args.env, env, env_cfg, q0, v0, goal,
                lambda qn, vn, ctx, g: (lambda a: model_rollout_cost_tijepa(
                    t_enc, t_pred, t_pos_probe, ctx[-kt:], kt, device, a, g, args.act_pen_coef)),
                max_k, args.mpc_horizon, args.mpc_total_steps, args.n_samples, args.n_iters, seed_i,
                action_dim, a_max=args.a_max)
            mpc_ds = [("oracle_mpc", d_oracle), ("baseline_mpc", d_baseline), ("tijepa_mpc", d_tijepa)]
            if has_rnn:
                d_rnn = run_closed_loop_mpc(
                    args.env, env, env_cfg, q0, v0, goal,
                    lambda qn, vn, ctx, g: (lambda a: model_rollout_cost_baseline(
                        r_enc, r_pred, r_pos_probe, ctx[-kr:], kr, device, a, g, args.act_pen_coef)),
                    max_k, args.mpc_horizon, args.mpc_total_steps, args.n_samples, args.n_iters, seed_i,
                    action_dim, a_max=args.a_max)
                mpc_ds.append(("rnn_mpc", d_rnn))
            for name, d in mpc_ds:
                mpc_results[name]["final_dist"].append(d)
                mpc_results[name]["success"].append(bool(d < args.success_thresh))
            if (task_i + 1) % 10 == 0:
                print(f"  mpc task {task_i+1}/{len(tasks)}", flush=True)

        mpc_summary = {}
        for name, r in mpc_results.items():
            mpc_summary[name] = {"mean_final_dist": float(np.mean(r["final_dist"])),
                                  "success_rate": float(np.mean(r["success"])), "n": len(r["final_dist"])}
        mpc_summary["config"] = {"mpc_horizon": args.mpc_horizon, "mpc_total_steps": args.mpc_total_steps}
        print("closed-loop MPC:", json.dumps(mpc_summary, indent=2), flush=True)
        summary["closed_loop_mpc"] = mpc_summary
        results["_mpc_raw"] = mpc_results

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("saved ->", args.out_json, flush=True)


if __name__ == "__main__":
    main()
