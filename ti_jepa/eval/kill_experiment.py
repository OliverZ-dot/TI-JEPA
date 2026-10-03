"""协议 B / Kill Experiment（说明书 §4.2, §0 step 3）：同构型、不同速度。

流程：
  1. 固定 q_t（构型），设两档速度 v1、v2=-v1（同一速度大小，方向相反）。
  2. 反推最近 k 帧的 context（用 env_utils.build_context_generic 的"取反速度
     正推"技巧，等价于"假设过去 k-1 步都在以该速度匀速滑行"，但对三个环境
     的真实动力学都成立，不只是 InertiaBall 的自由质点特例）。
  3. 渲染两份 context，当前帧 o_t 在无 blur 时应完全一致（因为 q 相同）。
  4. 用真实物理，a=0 各走 H 步，得到 ground truth 未来。
  5. 三个模型分别读 context，a=0 盲滚 H 步（不再喂真实帧）：

       A. baseline (k=3)：官方 LeWM 的真实设定——单帧 target，但 predictor
          能看 3 帧历史（§1："预测器可以看历史，N=3"）。
       B. baseline_k1（消融）：把 predictor 的历史也砍掉（k=1），逼所有信息
          都必须走单帧 z_t。这是"单帧 target 不可识别速度"最纯粹的操作化，
          不会被"predictor 偷偷用历史算有限差分"这个已知混淆项污染
          （§1 原文就提到这个混淆："预测器若看窗口，可以算出有限差分速度，
          但速度活在计算图里，不进入被监督的 z"）。
       C. TI-JEPA：z=(q,v) 结构化，predictor 严格无记忆（只吃当前 z_t+action，
          不额外喂历史窗口，见 models.TIJEPAPredictor）。

  比较：
    - 当前帧表征的距离（z_t / q_t，理论上都应 ≈0，定义性质）
    - TI-JEPA 的 v_t 距离 + 符号准确率（核心指标）
    - 三者的盲 rollout 位置 MSE、以及两条分支（+v/-v）在 rollout 里的
      branch separation（相对 ground truth 的比例）——这才是最终判据。

Go / No-go 判据见说明书 §4.2：
  Go：单帧/无历史时分不开、零动作预测塌向平均；TI-JEPA 能分、滑行 MSE 明显低。

本文件对 InertiaBall / Pendulum / CartPole 三个环境通用，用 --env 切换
（见 ti_jepa/envs/registry.py 和 ti_jepa/eval/env_utils.py）。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ti_jepa.data import EpisodeBatch, cache_path, split_episodes, WindowDataset
from ti_jepa.envs.registry import get_env_spec, make_scaled_config
from ti_jepa.eval.env_utils import build_context_generic, sample_qv1
from ti_jepa.eval.common import (
    episode_train_test_split,
    extract_probe_data,
    load_baseline,
    load_tijepa,
    to_tensor_frames,
)


def fit_position_probe(model_type, encoder, k, device, held_out_batch, n_probe_train_eps=280):
    """线性探针：把当前帧表征 (baseline z_t / TI-JEPA q_t) 解码回真实位置坐标。
    只用来把 latent rollout 转成"位置轨迹"方便和 ground truth 比较、画图，
    不参与任何模型的训练 loss（符合 §12 的硬约束：训练不读特权态）。
    """
    from sklearn.linear_model import Ridge

    data = extract_probe_data(model_type, encoder, held_out_batch, k, device)
    train_mask, test_mask = episode_train_test_split(data["episode_id"], n_probe_train_eps)
    key = "z_t" if model_type == "baseline" else "q_t"
    X = data[key]

    ds = WindowDataset(held_out_batch, k=k)
    q_true = np.stack([ds[i]["q_ctx"][-1].numpy() for i in range(len(ds))])

    reg = Ridge(alpha=1.0)
    reg.fit(X[train_mask], q_true[train_mask])
    pred = reg.predict(X[test_mask])
    r2 = 1 - np.mean((pred - q_true[test_mask]) ** 2) / np.mean(
        (q_true[test_mask] - q_true[test_mask].mean(0)) ** 2
    )
    print(f"  position probe held-out R^2 = {r2:.3f}")
    return reg


def fit_velocity_probe(encoder, k, device, held_out_batch, n_probe_train_eps=280):
    """把 TI-JEPA 的 v_t（模型内部维度）线性映射回真实速度编码，只用来做符号判定/画图。"""
    from sklearn.linear_model import Ridge

    data = extract_probe_data("tijepa", encoder, held_out_batch, k, device)
    train_mask, test_mask = episode_train_test_split(data["episode_id"], n_probe_train_eps)
    X, y = data["v_t_model"], data["v_t_true"]
    reg = Ridge(alpha=1.0)
    reg.fit(X[train_mask], y[train_mask])
    pred = reg.predict(X[test_mask])
    r2 = 1 - np.mean((pred - y[test_mask]) ** 2) / np.mean((y[test_mask] - y[test_mask].mean(0)) ** 2)
    print(f"  velocity probe (v_t -> true v) held-out R^2 = {r2:.3f}")
    return reg


def sample_pairs(env_name, rng, cfg, n_pairs, k, dt, horizon, speed_range):
    """返回 n_pairs 个 (q, v1, v2) 三元组，v2=-v1（同一速度大小方向/符号全部相反，
    见 env_utils 顶部注释：这对三个环境的编码表示都成立）。"""
    pairs = []
    for _ in range(n_pairs):
        q, v1 = sample_qv1(env_name, rng, cfg, k, horizon, speed_range)
        pairs.append((q, v1, -v1))
    return pairs


def build_context(env, q, v, k, dt, action_dim=None):
    action_dim = action_dim if action_dim is not None else np.asarray(v).shape[-1]
    return build_context_generic(env, q, v, k, action_dim)


def rollout_ground_truth(env, q, v, horizon, action_dim):
    env.set_state(q, v)
    actions = np.zeros((horizon, action_dim), dtype=np.float32)
    positions, velocities = env.rollout_open_loop(actions)
    return positions, velocities  # (H+1, pos_dim) each, includes t=0


@torch.no_grad()
def blind_rollout_baseline(encoder, predictor, o_ctx, k, horizon, device, action_dim):
    """o_ctx: (k,H,W,3) numpy -> 盲滚 horizon 步，返回 (horizon+1, z_dim) 的 z 轨迹（含起点）。"""
    img_s = o_ctx.shape[1]
    o_t = to_tensor_frames(o_ctx, device).unsqueeze(0)  # (1,k,3,H,W)
    z_hist = encoder(o_t.reshape(k, 3, img_s, img_s)).unsqueeze(0)  # (1,k,z_dim)
    zs = [z_hist[0, -1].cpu().numpy()]
    action = torch.zeros(1, action_dim, device=device)
    window = z_hist
    for _ in range(horizon):
        z_next = predictor(window, action)
        zs.append(z_next[0].cpu().numpy())
        window = torch.cat([window[:, 1:, :], z_next.unsqueeze(1)], dim=1)
    return np.stack(zs)


@torch.no_grad()
def blind_rollout_tijepa(encoder, predictor, o_ctx, k, horizon, device, action_dim):
    """返回 (q_traj, v_traj)：q_traj (horizon+1, q_dim)，v_traj (horizon+1, v_dim)，含起点。"""
    o_t = to_tensor_frames(o_ctx, device).unsqueeze(0)  # (1,k,3,H,W)
    q_window, v_t, _ = encoder.forward_window(o_t)
    q_t = q_window[:, -1, :]
    qs = [q_t[0].cpu().numpy()]
    vs = [v_t[0].cpu().numpy()]
    action = torch.zeros(1, action_dim, device=device)
    for _ in range(horizon):
        q_hat, v_hat = predictor(q_t, v_t, action)
        qs.append(q_hat[0].cpu().numpy())
        vs.append(v_hat[0].cpu().numpy())
        q_t, v_t = q_hat, v_hat
    return np.stack(qs), np.stack(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["inertia_ball", "pendulum", "cartpole"], default="inertia_ball")
    ap.add_argument("--baseline_ckpt", type=str, default="checkpoints/baseline.pt")
    ap.add_argument("--baseline_k1_ckpt", type=str, default="checkpoints/baseline_k1.pt")
    ap.add_argument("--tijepa_ckpt", type=str, default="checkpoints/tijepa.pt")
    ap.add_argument("--baseline_rnn_ckpt", type=str, default=None,
                     help="可选：RSSM/Dreamer风格递归聚合器baseline的checkpoint "
                          "(--predictor rnn训练得到)。提供时会作为第4个臂加入对比，"
                          "回应'为什么不直接用递归隐状态'这个问题。不提供则完全不受影响"
                          "（向后兼容，行为和原三臂版本一致）。")
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--speed_lo", type=float, default=0.03)
    ap.add_argument("--speed_hi", type=float, default=0.08)
    ap.add_argument("--out_json", type=str, default="results/kill_experiment.json")
    ap.add_argument("--out_fig", type=str, default="results/kill_experiment.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = get_env_spec(args.env)

    print("加载模型 + 拟合探针（探针只用于把 latent 解码成坐标，不进任何训练 loss）")
    print("[baseline k=3, 官方式历史 predictor]")
    b3_enc, b3_pred, b3_args = load_baseline(args.baseline_ckpt, device)

    print("[baseline k=1, 无记忆 predictor 消融]")
    b1_enc, b1_pred, b1_args = load_baseline(args.baseline_k1_ckpt, device)

    print("[TI-JEPA, 无记忆 predictor]")
    t_enc, t_pred, t_args = load_tijepa(args.tijepa_ckpt, device)

    has_rnn = args.baseline_rnn_ckpt is not None
    if has_rnn:
        print("[baseline RNN, RSSM/Dreamer风格递归聚合器对照]")
        br_enc, br_pred, br_args = load_baseline(args.baseline_rnn_ckpt, device)

    # 所有 checkpoint 的 image_size 必须一致，才能共用同一份 held-out 数据/env_cfg
    image_size = b3_args.get("image_size", 64)
    assert b1_args.get("image_size", 64) == image_size and t_args.get("image_size", 64) == image_size, (
        "baseline / baseline_k1 / tijepa 三个 checkpoint 的 image_size 不一致，无法公平对比"
    )
    if has_rnn:
        assert br_args.get("image_size", 64) == image_size, "baseline_rnn 的 image_size 与其它臂不一致"

    d = np.load(cache_path(args.env, image_size))
    full_batch = EpisodeBatch(d["frames"], d["actions"], d["positions"], d["velocities"])
    _, held_out = split_episodes(full_batch, int(d["n_train"]))

    b3_pos_probe = fit_position_probe("baseline", b3_enc, b3_args["k"], device, held_out)
    b1_pos_probe = fit_position_probe("baseline", b1_enc, b1_args["k"], device, held_out)
    t_pos_probe = fit_position_probe("tijepa", t_enc, t_args["k"], device, held_out)
    t_vel_probe = fit_velocity_probe(t_enc, t_args["k"], device, held_out)
    if has_rnn:
        br_pos_probe = fit_position_probe("baseline", br_enc, br_args["k"], device, held_out)

    k3, k1, kt = b3_args["k"], b1_args["k"], t_args["k"]
    assert k1 == 1, "baseline_k1 checkpoint 的 k 应该是 1"
    kr = br_args["k"] if has_rnn else 0

    env_cfg = make_scaled_config(args.env, image_size)
    env = spec.env_cls(env_cfg)
    action_dim = spec.action_dim

    rng = np.random.default_rng(args.seed)
    max_k = max(k3, kt, kr)
    pairs = sample_pairs(args.env, rng, env_cfg, args.n_pairs, max_k, env_cfg.dt, args.horizon,
                          (args.speed_lo, args.speed_hi))

    records = []
    example_for_plot = None

    for i, (q, v1, v2) in enumerate(pairs):
        ctx1_3 = build_context(env, q, v1, k3, env_cfg.dt, action_dim)
        ctx2_3 = build_context(env, q, v2, k3, env_cfg.dt, action_dim)
        ctx1_1 = ctx1_3[-1:]
        ctx2_1 = ctx2_3[-1:]
        o1_now, o2_now = ctx1_3[-1], ctx2_3[-1]
        img_diff = float(np.abs(o1_now.astype(np.float32) - o2_now.astype(np.float32)).mean())

        gt_pos1, _ = rollout_ground_truth(env, q, v1, args.horizon, action_dim)
        gt_pos2, _ = rollout_ground_truth(env, q, v2, args.horizon, action_dim)

        # ---- baseline k=3 (official-style history predictor) ----
        with torch.no_grad():
            z1_now = b3_enc(to_tensor_frames(ctx1_3[-1:], device)).cpu().numpy()[0]
            z2_now = b3_enc(to_tensor_frames(ctx2_3[-1:], device)).cpu().numpy()[0]
        z_dist_now = float(np.linalg.norm(z1_now - z2_now))

        z_roll1 = blind_rollout_baseline(b3_enc, b3_pred, ctx1_3, k3, args.horizon, device, action_dim)
        z_roll2 = blind_rollout_baseline(b3_enc, b3_pred, ctx2_3, k3, args.horizon, device, action_dim)
        pos_roll1_b3 = b3_pos_probe.predict(z_roll1)
        pos_roll2_b3 = b3_pos_probe.predict(z_roll2)
        b3_mse = (float(np.mean((pos_roll1_b3 - gt_pos1) ** 2)) + float(np.mean((pos_roll2_b3 - gt_pos2) ** 2))) / 2
        b3_branch_sep = float(np.linalg.norm(pos_roll1_b3[-1] - pos_roll2_b3[-1]))

        # ---- baseline k=1 (memoryless predictor ablation) ----
        z_roll1_1 = blind_rollout_baseline(b1_enc, b1_pred, ctx1_1, 1, args.horizon, device, action_dim)
        z_roll2_1 = blind_rollout_baseline(b1_enc, b1_pred, ctx2_1, 1, args.horizon, device, action_dim)
        pos_roll1_b1 = b1_pos_probe.predict(z_roll1_1)
        pos_roll2_b1 = b1_pos_probe.predict(z_roll2_1)
        b1_mse = (float(np.mean((pos_roll1_b1 - gt_pos1) ** 2)) + float(np.mean((pos_roll2_b1 - gt_pos2) ** 2))) / 2
        b1_branch_sep = float(np.linalg.norm(pos_roll1_b1[-1] - pos_roll2_b1[-1]))

        # ---- baseline RNN (RSSM/Dreamer-style recurrent aggregator, optional 4th arm) ----
        if has_rnn:
            ctx1_r, ctx2_r = ctx1_3[-kr:], ctx2_3[-kr:]
            z_roll1_r = blind_rollout_baseline(br_enc, br_pred, ctx1_r, kr, args.horizon, device, action_dim)
            z_roll2_r = blind_rollout_baseline(br_enc, br_pred, ctx2_r, kr, args.horizon, device, action_dim)
            pos_roll1_br = br_pos_probe.predict(z_roll1_r)
            pos_roll2_br = br_pos_probe.predict(z_roll2_r)
            br_mse = (float(np.mean((pos_roll1_br - gt_pos1) ** 2)) + float(np.mean((pos_roll2_br - gt_pos2) ** 2))) / 2
            br_branch_sep = float(np.linalg.norm(pos_roll1_br[-1] - pos_roll2_br[-1]))

        # ---- tijepa (memoryless predictor) ----
        ctx1_t, ctx2_t = ctx1_3[-kt:], ctx2_3[-kt:]
        with torch.no_grad():
            qw1, v1_model, _ = t_enc.forward_window(to_tensor_frames(ctx1_t, device).unsqueeze(0))
            qw2, v2_model, _ = t_enc.forward_window(to_tensor_frames(ctx2_t, device).unsqueeze(0))
        q_dist_now = float(np.linalg.norm(qw1[0, -1].cpu().numpy() - qw2[0, -1].cpu().numpy()))
        v1_model_np, v2_model_np = v1_model[0].cpu().numpy(), v2_model[0].cpu().numpy()
        v_dist_now = float(np.linalg.norm(v1_model_np - v2_model_np))
        v1_decoded = t_vel_probe.predict(v1_model_np[None, :])[0]
        v2_decoded = t_vel_probe.predict(v2_model_np[None, :])[0]
        sign_ok = bool(np.dot(v1_decoded, v1) > 0 and np.dot(v2_decoded, v2) > 0)

        q_roll1, _ = blind_rollout_tijepa(t_enc, t_pred, ctx1_t, kt, args.horizon, device, action_dim)
        q_roll2, _ = blind_rollout_tijepa(t_enc, t_pred, ctx2_t, kt, args.horizon, device, action_dim)
        pos_roll1_t = t_pos_probe.predict(q_roll1)
        pos_roll2_t = t_pos_probe.predict(q_roll2)
        t_mse = (float(np.mean((pos_roll1_t - gt_pos1) ** 2)) + float(np.mean((pos_roll2_t - gt_pos2) ** 2))) / 2
        t_branch_sep = float(np.linalg.norm(pos_roll1_t[-1] - pos_roll2_t[-1]))

        gt_branch_sep = float(np.linalg.norm(gt_pos1[-1] - gt_pos2[-1]))

        records.append({
            "img_diff": img_diff,
            "z_dist_now_baseline": z_dist_now,
            "q_dist_now_tijepa": q_dist_now,
            "v_dist_now_tijepa": v_dist_now,
            "sign_ok_tijepa": sign_ok,
            "baseline_k3_mse": b3_mse,
            "baseline_k1_mse": b1_mse,
            "tijepa_mse": t_mse,
            "baseline_k3_branch_sep": b3_branch_sep,
            "baseline_k1_branch_sep": b1_branch_sep,
            "tijepa_branch_sep": t_branch_sep,
            "gt_branch_sep": gt_branch_sep,
        })
        if has_rnn:
            records[-1]["baseline_rnn_mse"] = br_mse
            records[-1]["baseline_rnn_branch_sep"] = br_branch_sep

        if i == 0:
            example_for_plot = dict(
                o1_now=o1_now, o2_now=o2_now, gt_pos1=gt_pos1, gt_pos2=gt_pos2,
                pos_roll1_b3=pos_roll1_b3, pos_roll2_b3=pos_roll2_b3,
                pos_roll1_b1=pos_roll1_b1, pos_roll2_b1=pos_roll2_b1,
                pos_roll1_t=pos_roll1_t, pos_roll2_t=pos_roll2_t,
            )
            if has_rnn:
                example_for_plot["pos_roll1_br"] = pos_roll1_br
                example_for_plot["pos_roll2_br"] = pos_roll2_br

    keys = [k_ for k_ in records[0].keys() if k_ != "sign_ok_tijepa"]
    arr = {k_: np.array([r[k_] for r in records]) for k_ in keys}
    sign_acc = float(np.mean([r["sign_ok_tijepa"] for r in records]))

    def ratio(x):
        return float(arr[x].mean() / (arr["gt_branch_sep"].mean() + 1e-8))

    summary = {
        "env": args.env,
        "n_pairs": args.n_pairs,
        "horizon": args.horizon,
        "img_diff_mean": float(arr["img_diff"].mean()),
        "z_dist_now_baseline_mean": float(arr["z_dist_now_baseline"].mean()),
        "q_dist_now_tijepa_mean": float(arr["q_dist_now_tijepa"].mean()),
        "v_dist_now_tijepa_mean": float(arr["v_dist_now_tijepa"].mean()),
        "v_sign_accuracy_tijepa": sign_acc,
        "baseline_k3_rollout_pos_mse_mean": float(arr["baseline_k3_mse"].mean()),
        "baseline_k1_rollout_pos_mse_mean": float(arr["baseline_k1_mse"].mean()),
        "tijepa_rollout_pos_mse_mean": float(arr["tijepa_mse"].mean()),
        "gt_branch_sep_mean": float(arr["gt_branch_sep"].mean()),
        "baseline_k3_branch_sep_ratio_to_gt": ratio("baseline_k3_branch_sep"),
        "baseline_k1_branch_sep_ratio_to_gt": ratio("baseline_k1_branch_sep"),
        "tijepa_branch_sep_ratio_to_gt": ratio("tijepa_branch_sep"),
    }
    if has_rnn:
        summary["baseline_rnn_rollout_pos_mse_mean"] = float(arr["baseline_rnn_mse"].mean())
        summary["baseline_rnn_branch_sep_ratio_to_gt"] = ratio("baseline_rnn_branch_sep")
        summary["baseline_rnn_k"] = kr

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "records": [
            {kk: (bool(vv) if kk == "sign_ok_tijepa" else float(vv)) for kk, vv in r.items()} for r in records
        ]}, f, indent=2)
    print("saved", args.out_json)

    make_figure(example_for_plot, summary, args.out_fig)
    print("saved", args.out_fig)


def make_figure(ex, summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].imshow(ex["o1_now"])
    axes[0].set_title("o_t  (v = +v)")
    axes[0].axis("off")
    axes[1].imshow(ex["o2_now"])
    axes[1].set_title("o_t  (v = -v)\n(should look identical to left)")
    axes[1].axis("off")

    ax = axes[2]
    H = ex["gt_pos1"].shape[0]
    steps = np.arange(H)
    dim0 = 0
    ax.plot(steps, ex["gt_pos1"][:, dim0], "k-", lw=2, label="ground truth (+v)")
    ax.plot(steps, ex["gt_pos2"][:, dim0], "k--", lw=2, label="ground truth (-v)")
    ax.plot(steps, ex["pos_roll1_b3"][:, dim0], color="orange", ls="-", label="baseline k=3 (+v)")
    ax.plot(steps, ex["pos_roll2_b3"][:, dim0], color="orange", ls="--", label="baseline k=3 (-v)")
    ax.plot(steps, ex["pos_roll1_b1"][:, dim0], "r-", label="baseline k=1 (+v)")
    ax.plot(steps, ex["pos_roll2_b1"][:, dim0], "r--", label="baseline k=1 (-v)")
    ax.plot(steps, ex["pos_roll1_t"][:, dim0], "b-", label="TI-JEPA (+v)")
    ax.plot(steps, ex["pos_roll2_t"][:, dim0], "b--", label="TI-JEPA (-v)")
    if "pos_roll1_br" in ex:
        ax.plot(steps, ex["pos_roll1_br"][:, dim0], color="purple", ls="-", label="baseline RNN (+v)")
        ax.plot(steps, ex["pos_roll2_br"][:, dim0], color="purple", ls="--", label="baseline RNN (-v)")
    ax.set_xlabel("model step (coast, a=0)")
    ax.set_ylabel("pos[0] (decoded via linear probe)")
    ax.set_title("Same config, opposite velocity: blind rollout trajectory")
    ax.legend(fontsize=7, ncol=2)

    title = (
        f"kill experiment [{summary.get('env','inertia_ball')}]  (n_pairs={summary['n_pairs']}, H={summary['horizon']})   "
        f"img_diff={summary['img_diff_mean']:.4f}\n"
        f"branch_sep / gt ratio:  baseline(k=3)={summary['baseline_k3_branch_sep_ratio_to_gt']:.2f}   "
        f"baseline(k=1)={summary['baseline_k1_branch_sep_ratio_to_gt']:.2f}   "
        f"TI-JEPA={summary['tijepa_branch_sep_ratio_to_gt']:.2f}"
    )
    if "baseline_rnn_branch_sep_ratio_to_gt" in summary:
        title += f"   baseline(RNN)={summary['baseline_rnn_branch_sep_ratio_to_gt']:.2f}"
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)


if __name__ == "__main__":
    main()
