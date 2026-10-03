"""评测脚本共用的加载 / 推理工具。"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ti_jepa.models import (
    HistoryPredictor,
    LeWMStyleEncoder,
    RecurrentPredictor,
    TIJEPAEncoder,
    TIJEPAPredictor,
)


def load_baseline(path: str, device: str = "cuda"):
    state = torch.load(path, map_location=device)
    args = state["args"]
    image_size = args.get("image_size", 64)
    action_dim = args.get("action_dim", 2)
    feat_dim = args.get("feat_dim", 128)
    channels = args.get("channels", None)
    backbone = args.get("backbone", "conv")
    predictor_type = args.get("predictor", "mlp")
    encoder = LeWMStyleEncoder(image_size=image_size, z_dim=args["z_dim"], feat_dim=feat_dim, channels=channels,
                                backbone=backbone).to(device)
    if predictor_type == "mlp":
        predictor = HistoryPredictor(
            per_step_dim=args["z_dim"], k=args["k"], action_dim=action_dim, out_dim=args["z_dim"]
        ).to(device)
    elif predictor_type == "rnn":
        predictor = RecurrentPredictor(
            per_step_dim=args["z_dim"], k=args["k"], action_dim=action_dim, out_dim=args["z_dim"]
        ).to(device)
    else:
        from ti_jepa.official_predictor import OfficialBaselinePredictor
        predictor = OfficialBaselinePredictor(
            per_step_dim=args["z_dim"], k=args["k"], action_dim=action_dim, out_dim=args["z_dim"]
        ).to(device)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor, args


def load_tijepa(path: str, device: str = "cuda"):
    state = torch.load(path, map_location=device)
    args = state["args"]
    image_size = args.get("image_size", 64)
    action_dim = args.get("action_dim", 2)
    feat_dim = args.get("feat_dim", 128)
    channels = args.get("channels", None)
    backbone = args.get("backbone", "conv")
    predictor_type = args.get("predictor", "mlp")
    encoder = TIJEPAEncoder(image_size=image_size, q_dim=args["q_dim"], v_dim=args["v_dim"], k=args["k"],
                             feat_dim=feat_dim, channels=channels, backbone=backbone).to(device)
    if predictor_type == "mlp":
        predictor = TIJEPAPredictor(q_dim=args["q_dim"], v_dim=args["v_dim"], action_dim=action_dim).to(device)
    else:
        from ti_jepa.official_predictor import OfficialTIJEPAPredictor
        predictor = OfficialTIJEPAPredictor(q_dim=args["q_dim"], v_dim=args["v_dim"], action_dim=action_dim).to(device)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor, args


def to_tensor_frames(frames: np.ndarray, device: str) -> torch.Tensor:
    """frames: (..., H, W, 3) uint8 -> (..., 3, H, W) float in [0,1]."""
    t = torch.from_numpy(frames).float() / 255.0
    # move channel dim (last) to position -3
    perm = list(range(t.dim()))
    perm = perm[:-3] + [perm[-1], perm[-3], perm[-2]]
    return t.permute(*perm).to(device)


@torch.no_grad()
def extract_probe_data(model_type: str, encoder, batch, k: int, device: str, batch_size: int = 512):
    """按 episode 抽取窗口，跑 encoder，收集探针梯子(§4.1)需要的各种表征。

    返回 dict，所有数组按窗口对齐（第 0 维 = 窗口数），并带 `episode_id`，
    方便外面按 episode 切 probe 的 train/test（不打乱时间步，防泄漏）。
    """
    from ti_jepa.data import WindowDataset

    ds = WindowDataset(batch, k=k)
    n = len(ds)
    episode_id = np.array([ds.index[i][0] for i in range(n)])

    outs = {"episode_id": episode_id, "v_t_true": [], "v_next_true": []}
    if model_type == "baseline":
        outs.update({"z_t": [], "z_next": [], "z_hist": []})
    else:
        outs.update({"q_t": [], "q_next": [], "q_hist": [], "v_t_model": [], "v_next_model": []})

    for start in range(0, n, batch_size):
        idxs = range(start, min(start + batch_size, n))
        items = [ds[i] for i in idxs]
        o_ctx = torch.stack([it["o_ctx"] for it in items]).to(device)     # (B,k,3,H,W)
        o_next = torch.stack([it["o_next"] for it in items]).to(device)  # (B,3,H,W)
        v_ctx = torch.stack([it["v_ctx"] for it in items]).numpy()
        v_next = torch.stack([it["v_next"] for it in items]).numpy()
        b = o_ctx.shape[0]

        outs["v_t_true"].append(v_ctx[:, -1, :])
        outs["v_next_true"].append(v_next)

        if model_type == "baseline":
            img_s = o_ctx.shape[-1]
            z_ctx = encoder(o_ctx.reshape(b * k, 3, img_s, img_s)).reshape(b, k, -1)
            z_next = encoder(o_next)
            outs["z_t"].append(z_ctx[:, -1, :].cpu().numpy())
            outs["z_next"].append(z_next.cpu().numpy())
            outs["z_hist"].append(z_ctx.reshape(b, -1).cpu().numpy())
        else:
            q_window, v_t, _ = encoder.forward_window(o_ctx)
            o_next_window = torch.cat([o_ctx[:, 1:], o_next.unsqueeze(1)], dim=1)
            q_window_next, v_next_model, _ = encoder.forward_window(o_next_window)
            outs["q_t"].append(q_window[:, -1, :].cpu().numpy())
            outs["q_next"].append(q_window_next[:, -1, :].cpu().numpy())
            outs["q_hist"].append(q_window.reshape(b, -1).cpu().numpy())
            outs["v_t_model"].append(v_t.cpu().numpy())
            outs["v_next_model"].append(v_next_model.cpu().numpy())

    for key in list(outs.keys()):
        if key != "episode_id":
            outs[key] = np.concatenate(outs[key], axis=0)
    return outs


def fit_and_eval_probe(X_train, y_train, X_test, y_test, alpha: float = 1.0):
    """线性探针（Ridge），返回按维度平均的 Pearson r 和 MSE。"""
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr

    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    if y_test.ndim == 1:
        y_test = y_test[:, None]
        pred = pred[:, None]
    rs = []
    for d in range(y_test.shape[1]):
        if np.std(pred[:, d]) < 1e-9 or np.std(y_test[:, d]) < 1e-9:
            rs.append(0.0)
        else:
            rs.append(float(pearsonr(pred[:, d], y_test[:, d])[0]))
    mse = float(np.mean((pred - y_test) ** 2))
    return {"r": float(np.mean(rs)), "r_per_dim": rs, "mse": mse}


def episode_train_test_split(episode_id: np.ndarray, n_train_eps: int, seed: int = 0):
    """按 episode（不按窗口）切 train/test index，防止同一条轨迹的窗口同时出现在两侧。"""
    uniq = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    train_eps = set(perm[:n_train_eps].tolist())
    train_mask = np.isin(episode_id, list(train_eps))
    return train_mask, ~train_mask
