"""Pendulum：带重力恢复力的二阶转动惯性环境（新 benchmark #2）。

动机：InertiaBall 是无重力自由平动质点，物理上"最干净"但也最简单——
没有恢复力、没有非线性。Pendulum 换成转动 DOF + 重力恢复力矩，是控制论里
最经典的"必须知道速度才能刹得住"的例子（单摆摆到目标角度并悬停，
如果不知道当前角速度，不可能算出正确的反向力矩）。同时它在视觉上和
InertiaBall 完全不同（一根杆 + 一个球，而不是单点），用来证明我们的结论
不是 InertiaBall 这一个玩具环境的偶然产物。

状态 = (theta, omega)，theta 是杆的角度（0 = 竖直向下悬停，逆时针为正，
theta 允许无限累积不做 wrap，保证物理积分连续；omega 是角速度）。

按说明书 §12 的约束：ground truth (theta, omega) 只用于探针/评测，绝不
参与任何模型的训练 loss。

为了让"位置"探针任务不撞见角度绕圈导致的不可线性解码问题（这是我们在
Reacher 上踩过的坑，§ the results log），对外暴露的 get_state()/set_state()
统一走 sin/cos 参数化：

    q = (sin(theta), cos(theta))                       非绕圈、可线性解码
    v = d/dt q = (omega*cos(theta), -omega*sin(theta))  q 的时间导数

这样 q_dim=v_dim=2，和 InertiaBall 的 (x,y)/(vx,vy) 维度完全一致，
上层 data.py / models.py / eval/*.py 可以直接复用，无需改维度相关代码。

物理（每步，dt=1 的"回合制"物理，和 InertiaBall 一样是"瞬时冲量"范式）：
    omega <- omega + action[0]                  (动作 = 施加的角冲量)
    omega <- omega - gravity * sin(theta) * dt   (重力恢复力矩)
    omega <- omega * (1 - damping)
    theta <- theta + omega * dt
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclasses.dataclass
class PendulumConfig:
    image_size: int = 64
    dt: float = 1.0
    gravity: float = 0.055        # 重力恢复力矩系数（每步角速度变化量的尺度）
    damping: float = 0.0          # 0 = 无摩擦
    max_speed_reset: float = 0.35 # reset 时角速度采样上限（rad/step）
    rod_length_px_frac: float = 0.36  # 杆长（画布归一化，pivot 到 bob 的距离）
    bob_radius_px: int = 5
    rod_width_px: int = 2
    pivot_radius_px: int = 2
    background: Tuple[int, int, int] = (15, 15, 20)
    bob_color: Tuple[int, int, int] = (240, 200, 60)
    rod_color: Tuple[int, int, int] = (150, 150, 160)
    pivot_color: Tuple[int, int, int] = (90, 90, 100)
    seed: Optional[int] = None
    action_dim: int = 1
    pos_dim: int = 2
    vel_dim: int = 2


class Pendulum:
    """单摆：theta=0 竖直向下悬停（稳定平衡点），重力把它往 theta=0 拉回。"""

    def __init__(self, config: Optional[PendulumConfig] = None):
        self.cfg = config or PendulumConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.theta = 0.0
        self.omega = 0.0

    # ------------------------------------------------------------------
    def reset(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else self.rng
        self.theta = float(rng.uniform(-np.pi, np.pi))
        self.omega = float(rng.uniform(-self.cfg.max_speed_reset, self.cfg.max_speed_reset))
        return self.render()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """q=(sin,cos), v=(omega*cos,-omega*sin) -> 恢复 (theta, omega)。"""
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        s, c = q[0], q[1]
        norm = np.hypot(s, c) + 1e-8
        s, c = s / norm, c / norm
        self.theta = float(np.arctan2(s, c))
        self.omega = float(v[0] * c - v[1] * s)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        s, c = np.sin(self.theta), np.cos(self.theta)
        q = np.array([s, c], dtype=np.float32)
        v = np.array([self.omega * c, -self.omega * s], dtype=np.float32)
        return q, v

    def get_raw_state(self) -> Tuple[float, float]:
        """(theta, omega) 原始标量，仅供环境内部/校准脚本用，不进探针。"""
        return self.theta, self.omega

    def set_raw_state(self, theta: float, omega: float) -> None:
        self.theta = float(theta)
        self.omega = float(omega)

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        torque = float(action[0]) if action.size > 0 else 0.0
        self.omega += torque
        self.omega += -self.cfg.gravity * np.sin(self.theta) * self.cfg.dt
        self.omega *= (1.0 - self.cfg.damping)
        self.theta += self.omega * self.cfg.dt
        return self.render()

    def rollout_open_loop(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """actions: (H, 1) -> (q_traj (H+1,2), v_traj (H+1,2))，含起点。"""
        qs, vs = [self.get_state()[0]], [self.get_state()[1]]
        for a in actions:
            self.step(a)
            q, v = self.get_state()
            qs.append(q)
            vs.append(v)
        return np.stack(qs), np.stack(vs)

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        S = self.cfg.image_size
        img = Image.new("RGB", (S, S), self.cfg.background)
        draw = ImageDraw.Draw(img)

        pivot = np.array([0.5, 0.28]) * S
        L = self.cfg.rod_length_px_frac * S
        # theta=0 -> bob 竖直向下（图像坐标 y 向下为正）
        bob = pivot + L * np.array([np.sin(self.theta), np.cos(self.theta)])

        draw.line([tuple(pivot), tuple(bob)], fill=self.cfg.rod_color, width=self.cfg.rod_width_px)
        r_piv = self.cfg.pivot_radius_px
        draw.ellipse([pivot[0] - r_piv, pivot[1] - r_piv, pivot[0] + r_piv, pivot[1] + r_piv],
                     fill=self.cfg.pivot_color)
        r = self.cfg.bob_radius_px
        draw.ellipse([bob[0] - r, bob[1] - r, bob[0] + r, bob[1] + r], fill=self.cfg.bob_color)

        return np.array(img, dtype=np.uint8)


def make_pendulum_action_sequence(
    rng: np.random.Generator, n_steps: int, impulse_scale: float = 0.12, impulse_prob: float = 0.15
) -> np.ndarray:
    """随机稀疏角冲量序列，(n_steps, 1)。"""
    actions = np.zeros((n_steps, 1), dtype=np.float32)
    for t in range(n_steps):
        if rng.random() < impulse_prob:
            sign = rng.choice([-1.0, 1.0])
            mag = rng.uniform(0.3, 1.0) * impulse_scale
            actions[t, 0] = sign * mag
    return actions
