"""CartPole：小车-倒摆耦合二阶环境（新 benchmark #3，也是说明书应急预案里
明确提到的备选主环境 "PushT 过阻尼 → 主环境改 CartPole+Ball"）。

两个耦合的转动/平动 DOF（cart 位置 x、pole 角度 theta），力施加在小车上，
但因为 pole 通过力矩耦合项影响小车的有效加速度，正确预测"这个力会让小车
走到哪"本质上需要知道**当前两个 DOF 的速度**（尤其 pole 的角速度——同一个
(x, theta) 瞬时构型，如果 pole 正在向左倒 vs 向右倒，同样的力会让小车轨迹
完全不同）。这是比 InertiaBall/Pendulum 更"硬"的耦合动力学场景。

物理用经典 Barto-Sutton-Anderson (1983) / OpenAI Gym CartPole 方程（无摩擦
简化版），theta=0 为竖直向上（不稳定平衡点——推它一下，同角度不同角速度的
两个分支会迅速分道扬镳，是天然的 kill-experiment 素材）。每个 env.step 内部
用若干细分子步做 Euler 积分以保证数值稳定，对外仍表现为"一步 = 一次动作"。

sin/cos 参数化（避免角度绕圈导致线性探针失效，教训来自 Reacher，见
the results log）：

    q = (x, sin(theta), cos(theta))                                   3维
    v = d/dt q = (x_dot, theta_dot*cos(theta), -theta_dot*sin(theta)) 3维

q_dim = v_dim = 3（InertiaBall/Pendulum 是 2），data.py/models.py 均支持
任意维度，无需改动。
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclasses.dataclass
class CartPoleConfig:
    image_size: int = 64
    dt: float = 1.0                 # 外部一步的物理时长（秒）
    n_substeps: int = 10             # 内部细分子步数（数值稳定性）
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    half_length: float = 0.5         # 杆半长（米）
    force_scale: float = 3.0         # action=1.0 对应的力（牛顿）
    x_range: float = 1.6             # 渲染 + 边界反弹用的小车位置范围（米）
    x_restitution: float = 0.7       # 小车撞到 x_range 边界时的反弹系数
    reset_x_range: float = 0.5       # reset 时 x 采样范围
    reset_theta_range: float = 0.4   # reset 时 theta 采样范围（rad）
    reset_max_speed: float = 0.5     # reset 时 x_dot / theta_dot 采样上限
    cart_w_frac: float = 0.22
    cart_h_frac: float = 0.10
    pole_len_frac: float = 0.30
    pole_width_px: int = 2
    pole_tip_radius_px: int = 4
    background: Tuple[int, int, int] = (15, 15, 20)
    cart_color: Tuple[int, int, int] = (90, 140, 220)
    pole_color: Tuple[int, int, int] = (150, 150, 160)
    bob_color: Tuple[int, int, int] = (240, 200, 60)
    track_color: Tuple[int, int, int] = (60, 60, 70)
    seed: Optional[int] = None
    action_dim: int = 1
    pos_dim: int = 3
    vel_dim: int = 3


class CartPole:
    """theta=0 = 竖直向下悬停（稳定平衡点，杆挂在小车下方，类似小车拖着一个摆）；
    state = (x, x_dot, theta, theta_dot)。

    选稳定而非经典 CartPole-v1 的"倒立摆"（不稳定平衡点）配置，是因为随机/
    近零动作的被动 rollout 在不稳定平衡点附近几步内就会发散到极端角度和
    数百米外的小车位置（见 scripts/sanity_check_envs.py 早期版本的校准记录），
    根本没法用来做表征学习的被动数据采集。稳定摆挂配置保留了完全相同的
    "力施加在小车上、通过耦合项影响杆的角加速度"的二阶耦合动力学，只是
    平衡点从"倒立"换成"悬挂"，使随机探索的数据在合理范围内有界、可采集，
    同时依然是货真价实的双自由度耦合系统（kill experiment 的"同构型不同
    速度分道扬镳"论证不依赖平衡点是否稳定）。
    """

    def __init__(self, config: Optional[CartPoleConfig] = None):
        self.cfg = config or CartPoleConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

    # ------------------------------------------------------------------
    def reset(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else self.rng
        c = self.cfg
        self.x = float(rng.uniform(-c.reset_x_range, c.reset_x_range))
        self.theta = float(rng.uniform(-c.reset_theta_range, c.reset_theta_range))
        self.x_dot = float(rng.uniform(-c.reset_max_speed, c.reset_max_speed))
        self.theta_dot = float(rng.uniform(-c.reset_max_speed, c.reset_max_speed))
        return self.render()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        x, s, c_ = q[0], q[1], q[2]
        norm = np.hypot(s, c_) + 1e-8
        s, c_ = s / norm, c_ / norm
        self.x = float(x)
        self.theta = float(np.arctan2(s, c_))
        self.x_dot = float(v[0])
        self.theta_dot = float(v[1] * c_ - v[2] * s)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        s, c_ = np.sin(self.theta), np.cos(self.theta)
        q = np.array([self.x, s, c_], dtype=np.float32)
        v = np.array([self.x_dot, self.theta_dot * c_, -self.theta_dot * s], dtype=np.float32)
        return q, v

    def get_raw_state(self) -> Tuple[float, float, float, float]:
        return self.x, self.x_dot, self.theta, self.theta_dot

    def set_raw_state(self, x: float, x_dot: float, theta: float, theta_dot: float) -> None:
        self.x, self.x_dot, self.theta, self.theta_dot = float(x), float(x_dot), float(theta), float(theta_dot)

    # ------------------------------------------------------------------
    def _accel(self, force: float) -> Tuple[float, float]:
        """经典 CartPole 耦合方程，返回 (x_acc, theta_acc)。"""
        c = self.cfg
        total_mass = c.cart_mass + c.pole_mass
        pole_ml = c.pole_mass * c.half_length
        st, ct = np.sin(self.theta), np.cos(self.theta)
        temp = (force + pole_ml * self.theta_dot ** 2 * st) / total_mass
        # 悬挂（稳定）配置：重力项符号相对经典倒立摆方程翻转，使 theta=0
        # 成为吸引子而不是排斥子。
        theta_acc = (-c.gravity * st - ct * temp) / (
            c.half_length * (4.0 / 3.0 - c.pole_mass * ct ** 2 / total_mass)
        )
        x_acc = temp - pole_ml * theta_acc * ct / total_mass
        return x_acc, theta_acc

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        force = float(action[0]) * self.cfg.force_scale if action.size > 0 else 0.0
        dt_sub = self.cfg.dt / self.cfg.n_substeps
        c = self.cfg
        for _ in range(self.cfg.n_substeps):
            x_acc, theta_acc = self._accel(force)
            self.x_dot += x_acc * dt_sub
            self.x += self.x_dot * dt_sub
            self.theta_dot += theta_acc * dt_sub
            self.theta += self.theta_dot * dt_sub
            if self.x < -c.x_range:
                self.x = -c.x_range + (-c.x_range - self.x)
                self.x_dot = -self.x_dot * c.x_restitution
            elif self.x > c.x_range:
                self.x = c.x_range - (self.x - c.x_range)
                self.x_dot = -self.x_dot * c.x_restitution
        return self.render()

    def rollout_open_loop(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        qs, vs = [self.get_state()[0]], [self.get_state()[1]]
        for a in actions:
            self.step(a)
            q, v = self.get_state()
            qs.append(q)
            vs.append(v)
        return np.stack(qs), np.stack(vs)

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        c = self.cfg
        S = c.image_size
        img = Image.new("RGB", (S, S), c.background)
        draw = ImageDraw.Draw(img)

        track_y = S * 0.62
        draw.line([(0, track_y), (S, track_y)], fill=c.track_color, width=1)

        cart_px_x = S * 0.5 + (self.x / c.x_range) * (S * 0.5)
        cart_px_x = float(np.clip(cart_px_x, 2, S - 2))
        cw, ch = c.cart_w_frac * S, c.cart_h_frac * S
        draw.rectangle(
            [cart_px_x - cw / 2, track_y - ch / 2, cart_px_x + cw / 2, track_y + ch / 2],
            fill=c.cart_color,
        )

        pole_len = c.pole_len_frac * S
        pivot = np.array([cart_px_x, track_y + ch / 2])
        # theta=0 -> 竖直向下悬停（图像 y 向下为正，所以是 +cos）
        tip = pivot + pole_len * np.array([np.sin(self.theta), np.cos(self.theta)])
        draw.line([tuple(pivot), tuple(tip)], fill=c.pole_color, width=c.pole_width_px)
        r = c.pole_tip_radius_px
        draw.ellipse([tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r], fill=c.bob_color)

        return np.array(img, dtype=np.uint8)


def make_cartpole_action_sequence(
    rng: np.random.Generator, n_steps: int, impulse_scale: float = 0.35, impulse_prob: float = 0.2
) -> np.ndarray:
    """随机稀疏水平力序列，(n_steps, 1)，值域大致在 [-1,1]（乘 force_scale 生效）。"""
    actions = np.zeros((n_steps, 1), dtype=np.float32)
    for t in range(n_steps):
        if rng.random() < impulse_prob:
            sign = rng.choice([-1.0, 1.0])
            mag = rng.uniform(0.3, 1.0) * impulse_scale
            actions[t, 0] = sign * mag
    return actions
