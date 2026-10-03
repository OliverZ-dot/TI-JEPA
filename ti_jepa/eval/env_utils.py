"""跨环境共用的评测工具：把 kill_experiment.py / planning.py 里"构造同构型
不同速度的一对状态"、"反推 context"、"采样规划任务"这几件事从 InertiaBall
专用代码里抽出来，做成可插拔的每环境采样函数 + 一个通用的"倒放法"反推
context（利用二阶保守动力学的时间反演对称性：把速度取反、用真实动力学
正推 i 步，等价于用原速度反推 i 步——因为 render() 在三个环境里都只画
位置不画速度，取反不影响成像，取反+正推得到的位置序列就是我们要的
"过去 i 步"位置，参见模块内 build_context_generic 的注释）。

关键不变量（三个环境都满足，靠 (sin,cos) 派生速度编码的线性结构保证）：
    v2_encoded = -v1_encoded  <=>  两个物理自由度（如有）全部反号
这让"同构型、完全反向速度"的 pair 构造可以对三个环境用同一行代码
`v2 = -v1` 完成，只有 v1 怎么采样是每环境特有的。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sample_qv1(env_name: str, rng: np.random.Generator, cfg, k: int, horizon: int,
                speed_range: Tuple[float, float]):
    """返回 (q, v1)，q/v1 都是环境的编码表示（和 env.get_state() 同一套坐标）。
    v2 = -v1 由调用方直接算，不需要再采样一次。"""
    if env_name == "inertia_ball":
        speed = rng.uniform(*speed_range)
        angle = rng.uniform(0, 2 * np.pi)
        v1 = (speed * np.array([np.cos(angle), np.sin(angle)])).astype(np.float32)
        safe_margin = cfg.world_margin + (max(k, horizon) + 3) * speed * cfg.dt + 0.02
        safe_margin = min(safe_margin, 0.45)
        q = rng.uniform(safe_margin, 1.0 - safe_margin, size=2).astype(np.float32)
        return q, v1

    if env_name == "pendulum":
        theta = rng.uniform(-np.pi, np.pi)
        speed = rng.uniform(*speed_range)
        sign = rng.choice([-1.0, 1.0])
        omega = sign * speed
        s, c = np.sin(theta), np.cos(theta)
        q = np.array([s, c], dtype=np.float32)
        v1 = np.array([omega * c, -omega * s], dtype=np.float32)
        return q, v1

    if env_name == "cartpole":
        safe_margin_x = min(cfg.x_range - (max(k, horizon) + 3) * speed_range[1] * cfg.dt - 0.05, cfg.x_range * 0.6)
        safe_margin_x = max(safe_margin_x, 0.1)
        x = rng.uniform(-safe_margin_x, safe_margin_x)
        theta = rng.uniform(-cfg.reset_theta_range, cfg.reset_theta_range)
        x_dot = rng.uniform(*speed_range) * rng.choice([-1.0, 1.0])
        theta_dot = rng.uniform(*speed_range) * rng.choice([-1.0, 1.0])
        s, c = np.sin(theta), np.cos(theta)
        q = np.array([x, s, c], dtype=np.float32)
        v1 = np.array([x_dot, theta_dot * c, -theta_dot * s], dtype=np.float32)
        return q, v1

    raise ValueError(env_name)


def sample_task(env_name: str, rng: np.random.Generator, cfg, k: int, horizon: int,
                 speed_range: Tuple[float, float], goal_radius: Tuple[float, float] = (0.15, 0.35)):
    """返回 (q0, v0, goal)，goal 和 q 同维度（pos_dim），用于协议 C 的"到达并刹停"任务。"""
    if env_name == "inertia_ball":
        speed = rng.uniform(*speed_range)
        angle = rng.uniform(0, 2 * np.pi)
        v0 = (speed * np.array([np.cos(angle), np.sin(angle)])).astype(np.float32)
        safe_margin = min(cfg.world_margin + (k + horizon + 2) * speed * cfg.dt + 0.02, 0.4)
        q0 = rng.uniform(safe_margin, 1.0 - safe_margin, size=2).astype(np.float32)
        g_angle = rng.uniform(0, 2 * np.pi)
        g_radius = rng.uniform(*goal_radius)
        goal = q0 + g_radius * np.array([np.cos(g_angle), np.sin(g_angle)])
        goal = np.clip(goal, safe_margin, 1.0 - safe_margin).astype(np.float32)
        return q0, v0, goal

    if env_name == "pendulum":
        theta0 = rng.uniform(-np.pi, np.pi)
        speed = rng.uniform(*speed_range)
        omega0 = speed * rng.choice([-1.0, 1.0])
        s, c = np.sin(theta0), np.cos(theta0)
        q0 = np.array([s, c], dtype=np.float32)
        v0 = np.array([omega0 * c, -omega0 * s], dtype=np.float32)
        goal_theta = theta0 + rng.uniform(-1, 1) * rng.uniform(*goal_radius) * np.pi
        goal = np.array([np.sin(goal_theta), np.cos(goal_theta)], dtype=np.float32)
        return q0, v0, goal

    if env_name == "cartpole":
        safe_margin_x = min(cfg.x_range - (k + horizon + 2) * speed_range[1] * cfg.dt - 0.05, cfg.x_range * 0.5)
        safe_margin_x = max(safe_margin_x, 0.1)
        x0 = rng.uniform(-safe_margin_x, safe_margin_x)
        theta0 = rng.uniform(-cfg.reset_theta_range, cfg.reset_theta_range)
        x_dot0 = rng.uniform(*speed_range) * rng.choice([-1.0, 1.0])
        theta_dot0 = rng.uniform(*speed_range) * rng.choice([-1.0, 1.0])
        s, c = np.sin(theta0), np.cos(theta0)
        q0 = np.array([x0, s, c], dtype=np.float32)
        v0 = np.array([x_dot0, theta_dot0 * c, -theta_dot0 * s], dtype=np.float32)
        g_frac = rng.uniform(*goal_radius) * rng.choice([-1.0, 1.0])
        goal_x = float(np.clip(x0 + g_frac * cfg.x_range, -safe_margin_x, safe_margin_x))
        goal = np.array([goal_x, 0.0, 1.0], dtype=np.float32)  # 目标：小车到 goal_x，杆竖直悬停 (theta=0)
        return q0, v0, goal

    raise ValueError(env_name)


def inertia_ball_dynamics_batch(cfg, q0, v0, action_seqs):
    """(N,H,2) action_seqs -> (N,2) final (pos,vel)，纯物理无渲染（InertiaBall）。"""
    N, H, _ = action_seqs.shape
    x = np.tile(np.asarray(q0, dtype=np.float64)[None, :], (N, 1))
    v = np.tile(np.asarray(v0, dtype=np.float64)[None, :], (N, 1))
    m = cfg.world_margin
    for t in range(H):
        v = v + action_seqs[:, t]
        v = v * (1.0 - cfg.damping)
        x = x + v * cfg.dt
        if cfg.bounce_walls:
            for i in range(2):
                lo = x[:, i] < m
                x[lo, i] = m + (m - x[lo, i])
                v[lo, i] = -v[lo, i] * cfg.restitution
                hi = x[:, i] > 1.0 - m
                x[hi, i] = (1.0 - m) - (x[hi, i] - (1.0 - m))
                v[hi, i] = -v[hi, i] * cfg.restitution
    return x.astype(np.float32), v.astype(np.float32)


def pendulum_dynamics_batch(cfg, q0, v0, action_seqs):
    """(N,H,1) action_seqs -> (N,2)/(N,2) final (q,v) 编码，纯物理无渲染（Pendulum）。"""
    N, H, _ = action_seqs.shape
    theta0 = float(np.arctan2(q0[0], q0[1]))
    omega0 = float(v0[0] * np.cos(theta0) - v0[1] * np.sin(theta0))
    theta = np.full(N, theta0, dtype=np.float64)
    omega = np.full(N, omega0, dtype=np.float64)
    for t in range(H):
        torque = action_seqs[:, t, 0]
        omega = omega + torque
        omega = omega - cfg.gravity * np.sin(theta) * cfg.dt
        omega = omega * (1.0 - cfg.damping)
        theta = theta + omega * cfg.dt
    s, c = np.sin(theta), np.cos(theta)
    q_final = np.stack([s, c], axis=-1).astype(np.float32)
    v_final = np.stack([omega * c, -omega * s], axis=-1).astype(np.float32)
    return q_final, v_final


def cartpole_dynamics_batch(cfg, q0, v0, action_seqs):
    """(N,H,1) action_seqs -> (N,3)/(N,3) final (q,v) 编码，纯物理无渲染（CartPole）。"""
    N, H, _ = action_seqs.shape
    x0 = float(q0[0])
    theta0 = float(np.arctan2(q0[1], q0[2]))
    x_dot0 = float(v0[0])
    theta_dot0 = float(v0[1] * np.cos(theta0) - v0[2] * np.sin(theta0))
    x = np.full(N, x0, dtype=np.float64)
    x_dot = np.full(N, x_dot0, dtype=np.float64)
    theta = np.full(N, theta0, dtype=np.float64)
    theta_dot = np.full(N, theta_dot0, dtype=np.float64)
    total_mass = cfg.cart_mass + cfg.pole_mass
    pole_ml = cfg.pole_mass * cfg.half_length
    dt_sub = cfg.dt / cfg.n_substeps
    for t in range(H):
        force = action_seqs[:, t, 0] * cfg.force_scale
        for _ in range(cfg.n_substeps):
            st, ct = np.sin(theta), np.cos(theta)
            temp = (force + pole_ml * theta_dot ** 2 * st) / total_mass
            theta_acc = (-cfg.gravity * st - ct * temp) / (
                cfg.half_length * (4.0 / 3.0 - cfg.pole_mass * ct ** 2 / total_mass)
            )
            x_acc = temp - pole_ml * theta_acc * ct / total_mass
            x_dot = x_dot + x_acc * dt_sub
            x = x + x_dot * dt_sub
            theta_dot = theta_dot + theta_acc * dt_sub
            theta = theta + theta_dot * dt_sub
            lo = x < -cfg.x_range
            x[lo] = -cfg.x_range + (-cfg.x_range - x[lo])
            x_dot[lo] = -x_dot[lo] * cfg.x_restitution
            hi = x > cfg.x_range
            x[hi] = cfg.x_range - (x[hi] - cfg.x_range)
            x_dot[hi] = -x_dot[hi] * cfg.x_restitution
    s, c = np.sin(theta), np.cos(theta)
    q_final = np.stack([x, s, c], axis=-1).astype(np.float32)
    v_final = np.stack([x_dot, theta_dot * c, -theta_dot * s], axis=-1).astype(np.float32)
    return q_final, v_final


DYNAMICS_BATCH_FNS = {
    "inertia_ball": inertia_ball_dynamics_batch,
    "pendulum": pendulum_dynamics_batch,
    "cartpole": cartpole_dynamics_batch,
}


def oracle_dynamics_batch(env_name, cfg, q0, v0, action_seqs):
    return DYNAMICS_BATCH_FNS[env_name](cfg, q0, v0, action_seqs)


def build_context_generic(env, q, v, k: int, action_dim: int) -> np.ndarray:
    """反推最近 k 帧 context，最后一帧是当前帧 o_t=(q,v)。

    用"取反速度、正推 i 步、忽略速度只看渲染位置"的时间反演技巧：
    对二阶保守动力学（重力/弹性反弹都可逆，唯一要避开的是非弹性能量损失
    发生的边界反弹——所以外面的采样函数都留了 margin），(q,-v) 正推 i 步
    得到的位置，和 (q,v) 反推 i 步的真实位置轨迹一致；三个环境的 render()
    都只画位置不画速度，因此可以直接拿正推帧当反推帧用，不用为每个环境
    单独写"反向物理"。
    """
    env.set_state(q, -np.asarray(v))
    zero_action = np.zeros(action_dim, dtype=np.float32)
    frames = [env.render()]
    for _ in range(k - 1):
        frames.append(env.step(zero_action))
    frames.reverse()
    return np.stack(frames)
