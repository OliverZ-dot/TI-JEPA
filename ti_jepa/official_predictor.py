"""官方规模 predictor：AdaLN-conditioned Transformer（ARPredictor），逐字移植自
官方仓库 `module.py`（`Attention` / `FeedForward` / `ConditionalBlock` /
`Transformer` / `ARPredictor` / `Embedder`），用于"扩大规模，尝试原本 LeWM
规模"这一验证实验（用户请求）。

官方配置（`config/train/model/lewm.yaml`）：
    predictor: depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1
    action_encoder: module.Embedder（把 frameskip 帧的原始 action 拼一起，
                     过 Conv1d + MLP 得到和 embed_dim 一样维度的条件向量）

两个 wrapper，和 baseline / TI-JEPA 两套 encoder 对齐：
  - `OfficialBaselinePredictor`：num_frames=k，AR 地看 k 帧历史 z + action
    历史（官方原始设定——baseline 的"预测器可以看历史"，见说明书 §1）。
  - `OfficialTIJEPAPredictor`：num_frames=1，只喂当前一步 z_t=(q_t,v_t)+
    action_t（和小规模 `models.TIJEPAPredictor` 的设计原则一致：predictor
    严格无记忆，不能通过偷看历史窗口把速度"外挂"进来，否则就没有验证到
    z=(q,v) 本身是否是充分的 Markov 状态，见 models.py 里的详细注释）。
    用同一套 ARPredictor 类，只是 num_frames=1，保证两条 arm 的 predictor
    参数量/深度/宽度完全对齐，是一个公平的容量对比。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ----------------------------------------------------------------------------
# 逐字移植自官方 module.py（仅删掉未用到的 Block/dropout 细节保持一致）
# ----------------------------------------------------------------------------
def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.0, block_class=ConditionalBlock):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    """action encoder：(B,T,action_dim) -> (B,T,emb_dim)。逐字同官方。"""

    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLPHead(nn.Module):
    """逐字同官方 `module.MLP`：Linear -> norm_fn(hidden) -> GELU -> Linear。
    官方用它做 `projector`（接在 encoder 输出后）和 `pred_proj`（接在
    predictor 输出后），hidden_dim=2048，norm_fn=BatchNorm1d
    （见 `config/train/model/lewm.yaml`）。BatchNorm 是这类自监督/JEPA
    方法里已知的抗坍缩机制之一（强制跨 batch 的统计量，见 VICReg/SimSiam），
    我们自己的小 CNN 配方一直没有它——这也是"官方规模在默认小配方超参下会
    坍缩"的候选原因之一，补上它是这次 scale-up 修正的一部分。"""

    def __init__(self, dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction. 逐字同官方。"""

    def __init__(self, *, num_frames, depth, heads, mlp_dim, input_dim, hidden_dim,
                 output_dim=None, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim, hidden_dim, output_dim or input_dim, depth, heads, dim_head, mlp_dim,
            dropout, block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


# ----------------------------------------------------------------------------
# ti_jepa 侧的两个 wrapper，接口对齐 models.py 的 HistoryPredictor / TIJEPAPredictor，
# 方便 train.py / eval/*.py 用同一套调用方式，只是内部换成官方规模的 AdaLN transformer。
# ----------------------------------------------------------------------------
OFFICIAL_PREDICTOR_KWARGS = dict(depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1)


class OfficialBaselinePredictor(nn.Module):
    """baseline：AR 地看 k 帧历史 z + action 历史，输出下一步 z（对齐
    `models.HistoryPredictor` 的调用签名：forward(z_hist, action) -> z_pred，
    但这里 action 只是"当前一步"的 action，会在内部广播成 k 帧的 dummy 历史，
    因为 kill_experiment.py 的盲 rollout 是逐步喂 a=0 单步 action。"""

    def __init__(self, per_step_dim: int, k: int, action_dim: int, out_dim: int):
        super().__init__()
        assert out_dim == per_step_dim
        self.k = k
        self.action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim, emb_dim=per_step_dim)
        self.ar = ARPredictor(
            num_frames=k, input_dim=per_step_dim, hidden_dim=per_step_dim, output_dim=per_step_dim,
            **OFFICIAL_PREDICTOR_KWARGS,
        )
        self.pred_proj = MLPHead(per_step_dim)

    def forward(self, z_hist: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z_hist: (B,k,D)；action: (B,action_dim)（当前一步）或 (B,k,action_dim)（历史）。"""
        b, k, d = z_hist.shape
        if action.dim() == 2:
            action = action.unsqueeze(1).expand(b, k, action.shape[-1])
        act_emb = self.action_encoder(action)  # (B,k,D)
        out = self.ar(z_hist, act_emb)
        return self.pred_proj(out[:, -1, :])


class OfficialTIJEPAPredictor(nn.Module):
    """TI-JEPA：严格无记忆，只吃当前一步 z_t=(q_t,v_t) + action_t（T=1），
    用和 baseline 完全相同深度/宽度的 ARPredictor，保证容量对齐、公平对比
    （见 models.TIJEPAPredictor 的设计原则注释）。"""

    def __init__(self, q_dim: int, v_dim: int, action_dim: int):
        super().__init__()
        self.q_dim, self.v_dim = q_dim, v_dim
        d = q_dim + v_dim
        self.action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim, emb_dim=d)
        self.ar = ARPredictor(
            num_frames=1, input_dim=d, hidden_dim=d, output_dim=d,
            **OFFICIAL_PREDICTOR_KWARGS,
        )
        # pred_proj 只打在 q 分支上，不碰 v：v 必须保持"有限差分派生、可被
        # bias-free 约束"的干净语义（§5 method 的设计约束2），把它也塞进一个
        # 带 BatchNorm 的非线性投影会破坏这个结构性保证、把 q/v 重新搓混。
        # q 分支和 baseline 的 z 承担同样的"要被 SIGReg 塑形成各向同性高斯"
        # 的角色，所以官方的抗坍缩 projector 补在这里才是对的位置。
        self.pred_proj_q = MLPHead(q_dim)

    def forward(self, q_t: torch.Tensor, v_t: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_t = torch.cat([q_t, v_t], dim=-1).unsqueeze(1)  # (B,1,D)
        act_emb = self.action_encoder(action.unsqueeze(1))  # (B,1,D)
        out = self.ar(z_t, act_emb)[:, 0, :]
        q_hat, v_hat = out[:, : self.q_dim], out[:, self.q_dim :]
        return self.pred_proj_q(q_hat), v_hat


if __name__ == "__main__":
    bp = OfficialBaselinePredictor(per_step_dim=192, k=3, action_dim=2, out_dim=192)
    z = torch.randn(4, 3, 192)
    a = torch.randn(4, 2)
    print("baseline out", bp(z, a).shape, "params(M)=", sum(p.numel() for p in bp.parameters()) / 1e6)

    tp = OfficialTIJEPAPredictor(q_dim=96, v_dim=96, action_dim=2)
    q, v = torch.randn(4, 96), torch.randn(4, 96)
    qh, vh = tp(q, v, a)
    print("tijepa out", qh.shape, vh.shape, "params(M)=", sum(p.numel() for p in tp.parameters()) / 1e6)
