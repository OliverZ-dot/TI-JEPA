"""协议 A：探针梯子（Identifiability Ladder），对应说明书 §4.1。

冻住 encoder，在held-out 数据上，从下面几种输入线性回归真实速度 v=(vx,vy)：

  1. 单帧 z_t / q_t
  2. (z_t, z_{t+1}) 拼接
  3. (z_{t-k+1:t}) 整个历史窗口
  4. （仅 TI-JEPA）显式的 v_t

按 episode 切 probe 的 train/test，不按窗口随机切（防止同一条轨迹的相邻步
泄漏到两侧）。

用法：
  python -m ti_jepa.eval.probe_ladder --baseline_ckpt checkpoints/baseline.pt \
      --tijepa_ckpt checkpoints/tijepa.pt --out results/probe_ladder.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ti_jepa.data import EpisodeBatch, cache_path, split_episodes
from ti_jepa.eval.common import (
    episode_train_test_split,
    extract_probe_data,
    fit_and_eval_probe,
    load_baseline,
    load_tijepa,
)


def run_ladder_baseline(encoder, k, device, held_out_batch, n_probe_train_eps=280):
    data = extract_probe_data("baseline", encoder, held_out_batch, k, device)
    train_mask, test_mask = episode_train_test_split(data["episode_id"], n_probe_train_eps)

    v_true = data["v_t_true"]  # 当前时刻真实速度（用来测"单帧能不能读出当前速度"）
    rows = {}

    X = data["z_t"]
    rows["z_t (single frame)"] = fit_and_eval_probe(X[train_mask], v_true[train_mask], X[test_mask], v_true[test_mask])

    X = np.concatenate([data["z_t"], data["z_next"]], axis=1)
    rows["(z_t, z_t+1) pair"] = fit_and_eval_probe(X[train_mask], v_true[train_mask], X[test_mask], v_true[test_mask])

    X = data["z_hist"]
    rows["z_{t-k+1:t} window"] = fit_and_eval_probe(X[train_mask], v_true[train_mask], X[test_mask], v_true[test_mask])

    # 位置对照：z_t 能不能读出 *位置*（应该能，用真实速度积分近似当前位置没有 ground truth 位置，跳过，
    # 位置对照在 kill_experiment 里用 q 直接看）
    return rows


def run_ladder_tijepa(encoder, k, device, held_out_batch, n_probe_train_eps=280):
    data = extract_probe_data("tijepa", encoder, held_out_batch, k, device)
    train_mask, test_mask = episode_train_test_split(data["episode_id"], n_probe_train_eps)

    v_true = data["v_t_true"]
    rows = {}

    X = data["q_t"]
    rows["q_t (single frame, pose only)"] = fit_and_eval_probe(
        X[train_mask], v_true[train_mask], X[test_mask], v_true[test_mask]
    )

    X = np.concatenate([data["q_t"], data["q_next"]], axis=1)
    rows["(q_t, q_t+1) pair"] = fit_and_eval_probe(X[train_mask], v_true[train_mask], X[test_mask], v_true[test_mask])

    X = data["q_hist"]
    rows["q_{t-k+1:t} window"] = fit_and_eval_probe(X[train_mask], v_true[train_mask], X[test_mask], v_true[test_mask])

    X = data["v_t_model"]
    rows["v_t (explicit motion head) [主数字]"] = fit_and_eval_probe(
        X[train_mask], v_true[train_mask], X[test_mask], v_true[test_mask]
    )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["inertia_ball", "pendulum", "cartpole"], default="inertia_ball")
    ap.add_argument("--baseline_ckpt", type=str, default="checkpoints/baseline.pt")
    ap.add_argument("--tijepa_ckpt", type=str, default="checkpoints/tijepa.pt")
    ap.add_argument("--baseline_rnn_ckpt", type=str, default=None,
                     help="可选：RSSM/Dreamer风格递归聚合器baseline，加一列对照（见 kill_experiment.py 的同名参数）")
    ap.add_argument("--out", type=str, default="results/probe_ladder.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    b_enc, _, b_args = load_baseline(args.baseline_ckpt, device)
    t_enc, _, t_args = load_tijepa(args.tijepa_ckpt, device)
    image_size = b_args.get("image_size", 64)
    assert t_args.get("image_size", 64) == image_size, "baseline / tijepa 的 image_size 不一致"
    has_rnn = args.baseline_rnn_ckpt is not None
    if has_rnn:
        br_enc, _, br_args = load_baseline(args.baseline_rnn_ckpt, device)
        assert br_args.get("image_size", 64) == image_size, "baseline_rnn 的 image_size 不一致"

    d = np.load(cache_path(args.env, image_size))
    batch = EpisodeBatch(d["frames"], d["actions"], d["positions"], d["velocities"])
    _, held_out = split_episodes(batch, int(d["n_train"]))
    print("held-out episodes for probing:", held_out.frames.shape[0])

    print("\n== LeWM-style baseline ==")
    baseline_rows = run_ladder_baseline(b_enc, b_args["k"], device, held_out)
    for name, r in baseline_rows.items():
        print(f"  {name:35s}  r={r['r']:+.3f}  mse={r['mse']:.5f}")

    print("\n== TI-JEPA ==")
    tijepa_rows = run_ladder_tijepa(t_enc, t_args["k"], device, held_out)
    for name, r in tijepa_rows.items():
        print(f"  {name:35s}  r={r['r']:+.3f}  mse={r['mse']:.5f}")

    out_dict = {"baseline": baseline_rows, "tijepa": tijepa_rows}
    if has_rnn:
        print("\n== baseline RNN (RSSM/Dreamer-style recurrent aggregator) ==")
        rnn_rows = run_ladder_baseline(br_enc, br_args["k"], device, held_out)
        for name, r in rnn_rows.items():
            print(f"  {name:35s}  r={r['r']:+.3f}  mse={r['mse']:.5f}")
        out_dict["baseline_rnn"] = rnn_rows

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print("\nsaved to", args.out)


if __name__ == "__main__":
    main()
