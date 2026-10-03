"""SIGReg：Sketched Isotropic Gaussian Regularization。

按说明书 §1 / §3.3：把一批向量的随机投影推向标准正态分布（不是只匹配均值/方差，
是匹配整条分布），从而防止 JEPA 坍缩，且不需要负样本或 stop-gradient target。

实现方式（Epps–Pulley 型特征函数检验，随机投影扩到多维 = "sketched"）：

1. 采样 M 个随机单位方向 w_1..w_M（各向同性检验的"探针方向"）。
2. 把输入 z（形状 [N, d]，N 通常是 batch*time）投影到每个方向： p_m = z @ w_m。
3. 对若干频率 t_k，比较经验特征函数 phi_hat(t_k) = E[cos(t_k p)] + i E[sin(t_k p)]
   和标准正态的特征函数 phi(t_k) = exp(-t_k^2/2)。
4. 损失 = 对所有方向、所有频率的 |phi_hat - phi|^2 均值。

这个量在 p ~ N(0,1) 时精确为 0，且同时惩罚均值、方差、以及更高阶矩偏离高斯，
不需要显式算协方差矩阵，适合把正则打在低维 pose / 残差上。
"""

from __future__ import annotations

from typing import Optional

import torch


class SIGReg(torch.nn.Module):
    def __init__(self, num_directions: int = 64, freqs: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_directions = num_directions
        if freqs is None:
            # 覆盖低频到中频，足够刻画均值/方差/尾部形状的偏差
            freqs = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [N, d]。返回标量损失。"""
        if z.dim() > 2:
            z = z.reshape(-1, z.shape[-1])
        n, d = z.shape
        if n < 2:
            return z.sum() * 0.0

        w = torch.randn(d, self.num_directions, device=z.device, dtype=z.dtype)
        w = w / (w.norm(dim=0, keepdim=True) + 1e-8)  # [d, M]

        p = z @ w  # [N, M]
        t = self.freqs.to(z.device, z.dtype)  # [K]

        # tp: [N, M, K]
        tp = p.unsqueeze(-1) * t.view(1, 1, -1)
        cos_emp = tp.cos().mean(dim=0)  # [M, K]
        sin_emp = tp.sin().mean(dim=0)  # [M, K]

        cos_target = torch.exp(-0.5 * t * t).view(1, -1).expand_as(cos_emp)
        sin_target = torch.zeros_like(sin_emp)

        loss = (cos_emp - cos_target).pow(2) + (sin_emp - sin_target).pow(2)
        return loss.mean()
