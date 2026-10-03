"""两套模型，严格对应说明书 §3：

1. `LeWMStyleEncoder` / `LeWMStylePredictor`：官方 LeWM 的精神复现（简化版）——
   单帧 encoder，z_t = E(o_t)，target 是下一帧的单帧嵌入 z_{t+1} = E(o_{t+1})。
   predictor 可以看 k 帧历史 + 动作，但 encoder 本身逐帧独立、memoryless。
   这是我们要"打靶"的基线，不是最终要交付的方法。

2. `TIJEPAEncoder` / `TIJEPAPredictor`：§3.2 的 pose/motion 拆分——
   q_t 只看当前单帧；v_t 由最近 k 帧 pose 的差分算出（motion_enc 无 bias，
   保证静止 ⇒ 零差分 ⇒ 零速度，见 §3.2 的约束）。

两者共享同一个 `ConvBackbone`（小 CNN，替代官方 ViT-Tiny，为了能在单张卡上
快速跑通 InertiaBall 这一步；迁回 PushT / 官方 ViT 时只需替换 backbone）。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ConvBackbone(nn.Module):
    """CNN 图像编码器：(B,3,H,W) uint8/float -> (B, feat_dim)。

    `channels=None` 时用原始的小规模配置 (32,64,64,64)，4 层 stride-2 卷积。
    传入更宽/更深的 `channels` 列表（例如 (64,128,128,128,128)，5 层）可以做
    "scale validation"：backbone 容量是否是某些环境（如 CartPole 的细杆）
    v_t 探针上不去的瓶颈（r10 honest caveat 的后续追问，见 the results log）。
    """

    def __init__(self, image_size: int = 64, feat_dim: int = 128, channels=None):
        super().__init__()
        # IMPORTANT: default channels/kernel pattern below must stay byte-for-byte
        # identical to the original architecture (first layer 5x5, rest 3x3, all
        # stride-2, channels 32/64/64/64), or every existing checkpoint's
        # state_dict (InertiaBall/Pendulum/CartPole/real-PushT, baseline+TI-JEPA)
        # will fail to load. Only pass a non-default `channels` for NEW,
        # from-scratch "bigger backbone" runs.
        channels = list(channels) if channels is not None else [32, 64, 64, 64]
        layers = []
        in_c = 3
        for i, out_c in enumerate(channels):
            k, p = (5, 2) if i == 0 else (3, 1)
            layers += [nn.Conv2d(in_c, out_c, kernel_size=k, stride=2, padding=p), nn.ReLU(inplace=True)]
            in_c = out_c
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            n = self.net(dummy).flatten(1).shape[1]
        self.proj = nn.Linear(n, feat_dim)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # o: (B, 3, H, W) float in [0,1]
        h = self.net(o)
        h = h.flatten(1)
        return self.proj(h)


def bias_free_mlp(in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int = 1) -> nn.Sequential:
    """所有 Linear 都 bias=False，激活用 ReLU（ReLU(0)=0）。
    保证 f(0) = 0：输入全零时输出必为零向量。用于 motion_enc，实现
    "重复帧 -> 零差分 -> 零运动" 的约束（§3.2）。
    """
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(d, hidden, bias=False), nn.ReLU(inplace=True)]
        d = hidden
    layers += [nn.Linear(d, out_dim, bias=False)]
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------
# 1) LeWM-style baseline：单帧 target，encoder memoryless
# ----------------------------------------------------------------------------
def make_backbone(backbone: str, image_size: int, feat_dim: int, channels=None):
    """backbone='conv'（默认，向后兼容旧checkpoint）-> 小 CNN（ConvBackbone）；
    backbone='vit' -> 官方规模 ViT-Tiny/14（见 vit_backbone.py），用于
    "扩大规模，尝试原本 LeWM 规模" 的验证实验（用户请求）。"""
    if backbone == "conv":
        return ConvBackbone(image_size, feat_dim, channels=channels)
    elif backbone == "vit":
        from ti_jepa.vit_backbone import ViTTinyBackbone
        return ViTTinyBackbone(image_size=image_size, feat_dim=feat_dim)
    raise ValueError(f"unknown backbone {backbone!r}")


def make_head(backbone: str, feat_dim: int, out_dim: int):
    """conv backbone：保持原来的 ReLU+Linear（不动旧 checkpoint 的行为）。
    vit backbone：先线性投到 out_dim，再接官方的 projector 风格
    （Linear->BatchNorm->GELU->Linear, hidden=2048，见
    official_predictor.py::MLPHead）——这是官方规模抗坍缩的关键结构，之前
    scale-up 实验漏掉了它，导致全体坍缩。TI-JEPA 的 q_dim(=96) != feat_dim
    (=192)，所以 projector 统一放在 out_dim 空间，而不要求两者相等。"""
    if backbone == "vit":
        from ti_jepa.official_predictor import MLPHead
        return nn.Sequential(nn.Linear(feat_dim, out_dim), MLPHead(out_dim))
    return nn.Sequential(nn.ReLU(inplace=True), nn.Linear(feat_dim, out_dim))


class LeWMStyleEncoder(nn.Module):
    def __init__(self, image_size: int = 64, z_dim: int = 8, feat_dim: int = 128, channels=None,
                 backbone: str = "conv"):
        super().__init__()
        self.backbone = make_backbone(backbone, image_size, feat_dim, channels=channels)
        self.head = make_head(backbone, feat_dim, z_dim)
        self.z_dim = z_dim

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """o: (B, 3, H, W) 单帧 -> z: (B, z_dim)。逐帧独立，不看历史。"""
        return self.head(self.backbone(o))


class HistoryPredictor(nn.Module):
    """通用的历史条件 predictor：flatten(k帧 z) + action -> 下一步预测。"""

    def __init__(self, per_step_dim: int, k: int, action_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = per_step_dim * k + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_hist: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z_hist: (B, k, per_step_dim)，action: (B, action_dim)。"""
        b = z_hist.shape[0]
        flat = z_hist.reshape(b, -1)
        x = torch.cat([flat, action], dim=-1)
        return self.net(x)


class RecurrentPredictor(nn.Module):
    """RSSM / Dreamer 风格的递归聚合器 —— 与 `HistoryPredictor` **同一个接口**
    （`forward(z_hist, action) -> z_pred`，z_hist: (B,k,per_step_dim)），唯一
    区别是把 k 帧窗口 "拼接成一个大向量再过 MLP" 换成 "用 GRUCell 逐步吸收成
    一个隐状态 h"。这是回应"为什么不直接用一个递归隐状态（RSSM/Dreamer/
    PlaNet 那套）来解决单帧目标丢速度的问题，而要费劲搞一个显式 (q,v) 拆分"
    这个最直接、最容易被审稿人问到的问题的对照实验（见
    the paper draft 关于 RSSM 对照的讨论）。

    刻意保持极简（单层 GRUCell + 两层 MLP head），因为这里要验证的是
    "recurrence 这个机制本身是否天然解决了速度不可辨识"，不是"更强的 RNN
    架构工程"——如果连这个简化版都能在 kill experiment / planning 上和
    TI-JEPA 打平甚至更好，那才是对 TI-JEPA 存在必要性的真实威胁；如果不能，
    则说明"多帧编码进 predictor 的输入"（无论以窗口拼接还是递归方式）本身
    并不足够，问题出在 target 只监督单帧这件事上（Prop. 1 的核心论点），
    支持我们"必须显式重构 target"这一结论。
    """

    def __init__(self, per_step_dim: int, k: int, action_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.per_step_dim = per_step_dim
        self.k = k
        self.hidden = hidden
        self.gru_cell = nn.GRUCell(per_step_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_hist: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z_hist: (B, k, per_step_dim) —— 沿 k 维逐步喂给 GRUCell（不额外喂
        中间动作，和 HistoryPredictor 的窗口约定一致：数据集只提供 context
        最后一步的动作，见 data.py::WindowDataset）；用最终隐状态 h 和当前
        动作预测下一步。k 不必等于训练时的 `self.k`（GRUCell 天然支持变长
        序列），只是记录下来方便日志/checkpoint 里核对。"""
        b, k, _ = z_hist.shape
        h = torch.zeros(b, self.hidden, device=z_hist.device, dtype=z_hist.dtype)
        for t in range(k):
            h = self.gru_cell(z_hist[:, t, :], h)
        x = torch.cat([h, action], dim=-1)
        return self.head(x)


# ----------------------------------------------------------------------------
# 2) TI-JEPA：pose (单帧) + motion (帧差) 结构化 z
# ----------------------------------------------------------------------------
class TIJEPAEncoder(nn.Module):
    def __init__(self, image_size: int = 64, q_dim: int = 4, v_dim: int = 4,
                 k: int = 3, feat_dim: int = 128, motion_hidden: int = 64, channels=None,
                 backbone: str = "conv"):
        super().__init__()
        self.backbone = make_backbone(backbone, image_size, feat_dim, channels=channels)
        self.pose_head = make_head(backbone, feat_dim, q_dim)
        self.k = k
        self.q_dim = q_dim
        self.v_dim = v_dim
        # motion_enc 输入是 (k-1) 个相邻 pose 差分，flatten 后过 bias-free MLP
        self.motion_enc = bias_free_mlp((k - 1) * q_dim, motion_hidden, v_dim, n_hidden_layers=1)

    def encode_pose(self, o: torch.Tensor) -> torch.Tensor:
        """单帧 -> q。绝不允许看到别的帧，防止把速度偷运进 q。"""
        return self.pose_head(self.backbone(o))

    def encode_motion_from_poses(self, q_window: torch.Tensor) -> torch.Tensor:
        """q_window: (B, k, q_dim) 最近 k 帧的 pose -> v_t。"""
        diffs = q_window[:, 1:, :] - q_window[:, :-1, :]  # (B, k-1, q_dim)
        b = diffs.shape[0]
        return self.motion_enc(diffs.reshape(b, -1))

    def forward_window(self, o_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """o_window: (B, k, 3, H, W) 最近 k 帧 -> (q_window, v_t, z_t)。
        q_window: (B,k,q_dim) 每帧的 pose；v_t: (B,v_dim) 当前时刻的运动；
        z_t = concat(q_window[:, -1], v_t)。
        """
        b, k, c, h, w = o_window.shape
        flat = o_window.reshape(b * k, c, h, w)
        q_flat = self.encode_pose(flat)
        q_window = q_flat.reshape(b, k, self.q_dim)
        v_t = self.encode_motion_from_poses(q_window)
        z_t = torch.cat([q_window[:, -1, :], v_t], dim=-1)
        return q_window, v_t, z_t


class TIJEPAPredictor(nn.Module):
    """predictor 严格按 §3.2 原文："predictor 输入是 z_t=(q_t,v_t) 和 a_t，
    输出 ẑ_{t+1}=(q̂_{t+1},v̂_{t+1})"——只喂**当前一步**的 z_t，不额外喂 pose
    历史窗口。这是 TI-JEPA 和 baseline 的关键差异：baseline 的 predictor 必须
    靠看 k 帧历史窗口这个"外挂"才能间接算出速度；TI-JEPA 把 v 做成 z 的一部分
    之后，predictor 应该可以是**无记忆的单步转移**，因为 (q,v) 本身就是
    Markov 状态。如果 predictor 还要再吃一遍历史窗口，就没有验证到"z 本身
    是否够用"，会把 baseline 的"历史外挂"优势也悄悄搬进 TI-JEPA，让对比不公平。
    """

    def __init__(self, q_dim: int, v_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q_dim, self.v_dim = q_dim, v_dim
        in_dim = q_dim + v_dim + action_dim
        out_dim = q_dim + v_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, q_t: torch.Tensor, v_t: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([q_t, v_t, action], dim=-1)
        out = self.net(x)
        q_hat, v_hat = out[:, : self.q_dim], out[:, self.q_dim :]
        return q_hat, v_hat
