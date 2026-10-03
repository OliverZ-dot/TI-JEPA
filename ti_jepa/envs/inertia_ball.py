"""InertiaBall: 最小化的二阶惯性环境。

按说明书 §5.2 实现：一个球在（近似）无摩擦平面上滑动，动作是瞬时冲量。
状态 = (x, y, vx, vy)。观测 = 俯视 RGB 渲染，无 motion blur（可选开启，用于
§4.2 对照实验）。

设计上特意让 set_state 暴露 (q, v) 拆分，方便 kill experiment 直接摆放
"同构型不同速度"的两份状态并各自渲染 / 前推。

物理上只有 20 行：
    p <- p + v * dt
    v <- v + a          (a 是这一步的冲量，直接加进速度)
    v <- v * (1 - damping)
    边界反弹（可关）
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class InertiaBallConfig:
    image_size: int = 64
    dt: float = 1.0
    damping: float = 0.0          # 0 = 无摩擦；>0 = 弱阻尼
    restitution: float = 0.9      # 撞墙反弹系数
    bounce_walls: bool = True
    ball_radius_px: int = 4
    max_speed: float = 0.18       # 每步位移上限（归一化坐标，[0,1]画布）
    world_margin: float = 0.06    # 球心允许到达的边界（留半径）
    motion_blur_substeps: int = 1  # 1 = 无blur；>1 = 曝光期间多次子采样叠加
    background: Tuple[int, int, int] = (15, 15, 20)
    ball_color: Tuple[int, int, int] = (240, 200, 60)
    seed: Optional[int] = None


class InertiaBall:
    """单球二阶惯性环境。

    观测坐标系：位置 (x, y) in [0, 1]^2，(0,0) 为画布左上角。
    """

    def __init__(self, config: Optional[InertiaBallConfig] = None):
        self.cfg = config or InertiaBallConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.x = np.zeros(2, dtype=np.float32)
        self.v = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def reset(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else self.rng
        m = self.cfg.world_margin
        self.x = rng.uniform(m, 1.0 - m, size=2).astype(np.float32)
        speed = rng.uniform(0.0, self.cfg.max_speed)
        angle = rng.uniform(0, 2 * np.pi)
        self.v = (speed * np.array([np.cos(angle), np.sin(angle)])).astype(np.float32)
        return self.render()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """直接设置 (position, velocity)，用于 kill experiment 构造同 q 不同 v 的对。"""
        self.x = np.asarray(q, dtype=np.float32).copy()
        self.v = np.asarray(v, dtype=np.float32).copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.v.copy()

    # ------------------------------------------------------------------
    # 动力学
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> np.ndarray:
        """action: 2D 瞬时冲量，直接加到速度上（世界坐标，归一化画布单位/步）。"""
        action = np.asarray(action, dtype=np.float32)
        self.v = self.v + action
        self.v = self.v * (1.0 - self.cfg.damping)
        self.x = self.x + self.v * self.cfg.dt

        if self.cfg.bounce_walls:
            m = self.cfg.world_margin
            for i in range(2):
                if self.x[i] < m:
                    self.x[i] = m + (m - self.x[i])
                    self.v[i] = -self.v[i] * self.cfg.restitution
                elif self.x[i] > 1.0 - m:
                    self.x[i] = (1.0 - m) - (self.x[i] - (1.0 - m))
                    self.v[i] = -self.v[i] * self.cfg.restitution
        return self.render()

    def rollout_open_loop(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """给定动作序列，返回 (positions, velocities)，包含起点，长度 = len(actions)+1。"""
        xs = [self.x.copy()]
        vs = [self.v.copy()]
        for a in actions:
            self.step(a)
            xs.append(self.x.copy())
            vs.append(self.v.copy())
        return np.stack(xs), np.stack(vs)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        """返回 (H, W, 3) uint8 RGB。若 motion_blur_substeps > 1，用瞬时速度在
        本帧曝光窗口内做子采样叠加，模拟运动模糊（默认关闭）。
        """
        S = self.cfg.image_size
        img = np.zeros((S, S, 3), dtype=np.float32)
        img[:, :, 0] = self.cfg.background[0]
        img[:, :, 1] = self.cfg.background[1]
        img[:, :, 2] = self.cfg.background[2]

        n_sub = max(1, self.cfg.motion_blur_substeps)
        if n_sub == 1:
            positions = [self.x]
            weights = [1.0]
        else:
            # 曝光窗口内假设匀速，均匀采样 [-0.5dt, 0.5dt]
            offsets = np.linspace(-0.5, 0.5, n_sub)
            positions = [self.x + off * self.v * self.cfg.dt for off in offsets]
            weights = [1.0 / n_sub] * n_sub

        for pos, w in zip(positions, weights):
            self._draw_disk(img, pos, w)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _draw_disk(self, img: np.ndarray, pos_norm: np.ndarray, weight: float) -> None:
        S = self.cfg.image_size
        r = self.cfg.ball_radius_px
        cx = pos_norm[0] * S
        cy = pos_norm[1] * S
        x0, x1 = max(0, int(cx - r - 1)), min(S, int(cx + r + 2))
        y0, y1 = max(0, int(cy - r - 1)), min(S, int(cy + r + 2))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist2 = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2
        mask = dist2 <= r * r
        color = np.array(self.cfg.ball_color, dtype=np.float32)
        region = img[y0:y1, x0:x1]
        alpha = weight
        region[mask] = region[mask] * (1 - alpha) + color * alpha
        img[y0:y1, x0:x1] = region


def make_impulse_action_sequence(
    rng: np.random.Generator, n_steps: int, impulse_scale: float = 0.01, impulse_prob: float = 0.15
) -> np.ndarray:
    """随机稀疏冲量序列：大部分步 a=0（滑行），偶尔来一个冲量。"""
    actions = np.zeros((n_steps, 2), dtype=np.float32)
    for t in range(n_steps):
        if rng.random() < impulse_prob:
            angle = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(0.3, 1.0) * impulse_scale
            actions[t] = mag * np.array([np.cos(angle), np.sin(angle)])
    return actions
