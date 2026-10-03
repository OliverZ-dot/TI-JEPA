"""InertiaBall 数据生成：随机冲量 rollout，产出 (frames, actions, positions, velocities)。

按 episode 生成（而不是随机打乱时间步），方便后续探针/训练严格按 episode 切
train/test，避免时间泄漏（§4.1 要求）。
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ti_jepa.envs.registry import get_env_spec

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def cache_path(env_name: str, image_size: int = 64, data_dir: Optional[str] = None) -> str:
    """统一的数据缓存文件名。image_size=64（历史默认值）时文件名不加后缀，
    保持和已有缓存/结果完全向后兼容；其它分辨率（如官方规模 224）加 `_{size}`
    后缀，避免覆盖/污染已经生成过的 64x64 数据集。"""
    data_dir = data_dir if data_dir is not None else _DATA_DIR
    suffix = "" if image_size == 64 else f"_{image_size}"
    return os.path.abspath(os.path.join(data_dir, f"{env_name}{suffix}.npz"))


@dataclasses.dataclass
class EpisodeBatch:
    frames: np.ndarray       # (E, T+1, H, W, 3) uint8
    actions: np.ndarray      # (E, T, action_dim) float32
    positions: np.ndarray    # (E, T+1, pos_dim) float32
    velocities: np.ndarray   # (E, T+1, vel_dim) float32


def generate_episodes(
    n_episodes: int,
    n_steps: int,
    env_cfg=None,
    seed: int = 0,
    env_name: str = "inertia_ball",
    impulse_scale: Optional[float] = None,
    impulse_prob: Optional[float] = None,
) -> EpisodeBatch:
    """env_name 选环境（见 ti_jepa/envs/registry.py），env_cfg 是该环境自己的
    Config dataclass（不传则用默认值）。三个环境共用同一份 episode 生成逻辑，
    只是 action_dim / pos_dim / vel_dim 从 registry 里读，不再硬编码 2。"""
    spec = get_env_spec(env_name)
    env_cfg = env_cfg or spec.config_cls()
    impulse_scale = impulse_scale if impulse_scale is not None else spec.default_impulse_scale
    impulse_prob = impulse_prob if impulse_prob is not None else spec.default_impulse_prob

    S = env_cfg.image_size
    frames = np.zeros((n_episodes, n_steps + 1, S, S, 3), dtype=np.uint8)
    actions = np.zeros((n_episodes, n_steps, spec.action_dim), dtype=np.float32)
    positions = np.zeros((n_episodes, n_steps + 1, spec.pos_dim), dtype=np.float32)
    velocities = np.zeros((n_episodes, n_steps + 1, spec.vel_dim), dtype=np.float32)

    master_rng = np.random.default_rng(seed)
    env = spec.env_cls(env_cfg)
    for e in range(n_episodes):
        ep_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        frames[e, 0] = env.reset(ep_rng)
        positions[e, 0], velocities[e, 0] = env.get_state()
        acts = spec.action_seq_fn(ep_rng, n_steps, impulse_scale, impulse_prob)
        actions[e] = acts
        for t in range(n_steps):
            frames[e, t + 1] = env.step(acts[t])
            positions[e, t + 1], velocities[e, t + 1] = env.get_state()

    return EpisodeBatch(frames, actions, positions, velocities)


class WindowDataset(Dataset):
    """从 EpisodeBatch 里切出长度 k+1 的滑窗：context o_{t-k+1:t}，
    action a_t，label o_{t+1}/q_{t+1}/v_{t+1}。按 episode 存好，__getitem__
    随机取 (episode, start) 对，天然不跨 episode。
    """

    def __init__(self, batch: EpisodeBatch, k: int = 3):
        self.batch = batch
        self.k = k
        n_ep, n_frames = batch.frames.shape[0], batch.frames.shape[1]
        self.max_start = n_frames - 1 - k  # 需要 k 帧 context + 1 帧 label
        assert self.max_start >= 0, "n_steps 太短，装不下 k 帧 context + 1 帧 label"
        self.index = [(e, s) for e in range(n_ep) for s in range(self.max_start + 1)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        e, s = self.index[idx]
        k = self.k
        o_ctx = self.batch.frames[e, s : s + k]           # (k, H, W, 3)
        o_next = self.batch.frames[e, s + k]                # (H, W, 3)
        a_t = self.batch.actions[e, s + k - 1]               # 作用在 context 最后一帧上的动作
        q_ctx = self.batch.positions[e, s : s + k]           # (k, 2)
        q_next = self.batch.positions[e, s + k]
        v_ctx = self.batch.velocities[e, s : s + k]
        v_next = self.batch.velocities[e, s + k]

        o_ctx_t = torch.from_numpy(o_ctx).permute(0, 3, 1, 2).float() / 255.0
        o_next_t = torch.from_numpy(o_next).permute(2, 0, 1).float() / 255.0
        return {
            "o_ctx": o_ctx_t,          # (k, 3, H, W)
            "o_next": o_next_t,        # (3, H, W)
            "action": torch.from_numpy(a_t).float(),
            "q_ctx": torch.from_numpy(q_ctx).float(),
            "q_next": torch.from_numpy(q_next).float(),
            "v_ctx": torch.from_numpy(v_ctx).float(),
            "v_next": torch.from_numpy(v_next).float(),
        }


def split_episodes(batch: EpisodeBatch, n_train: int):
    train = EpisodeBatch(
        batch.frames[:n_train], batch.actions[:n_train], batch.positions[:n_train], batch.velocities[:n_train]
    )
    test = EpisodeBatch(
        batch.frames[n_train:], batch.actions[n_train:], batch.positions[n_train:], batch.velocities[n_train:]
    )
    return train, test
