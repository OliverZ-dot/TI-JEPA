"""环境注册表：把 data.py / train.py / eval/*.py 里所有"InertiaBall 专用"的
硬编码（action_dim=2、pos_dim=2、64x64 reshape 等）收敛成一处，让同一套
探针梯子 / kill experiment / CEM planning 代码可以在 InertiaBall / Pendulum /
CartPole 三个环境之间复用（§ 用户请求：做出多个真正涉及速度物理量的新
benchmark，而不是依赖单一玩具环境的偶然结果）。
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Type

from ti_jepa.envs.inertia_ball import InertiaBall, InertiaBallConfig, make_impulse_action_sequence
from ti_jepa.envs.pendulum import Pendulum, PendulumConfig, make_pendulum_action_sequence
from ti_jepa.envs.cartpole import CartPole, CartPoleConfig, make_cartpole_action_sequence


@dataclasses.dataclass
class EnvSpec:
    name: str
    env_cls: Type
    config_cls: Type
    action_seq_fn: Callable
    action_dim: int
    pos_dim: int
    vel_dim: int
    default_impulse_scale: float
    default_impulse_prob: float = 0.15


ENV_REGISTRY = {
    "inertia_ball": EnvSpec(
        name="inertia_ball", env_cls=InertiaBall, config_cls=InertiaBallConfig,
        action_seq_fn=make_impulse_action_sequence, action_dim=2, pos_dim=2, vel_dim=2,
        default_impulse_scale=0.035,
    ),
    "pendulum": EnvSpec(
        name="pendulum", env_cls=Pendulum, config_cls=PendulumConfig,
        action_seq_fn=make_pendulum_action_sequence, action_dim=1, pos_dim=2, vel_dim=2,
        default_impulse_scale=0.12,
    ),
    "cartpole": EnvSpec(
        name="cartpole", env_cls=CartPole, config_cls=CartPoleConfig,
        action_seq_fn=make_cartpole_action_sequence, action_dim=1, pos_dim=3, vel_dim=3,
        default_impulse_scale=0.35, default_impulse_prob=0.2,
    ),
}


def get_env_spec(name: str) -> EnvSpec:
    if name not in ENV_REGISTRY:
        raise ValueError(f"unknown env {name!r}, choices={list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]


# 三个环境里所有"固定像素"的装饰性字段（线宽/半径），按 image_size/64 等比缩放，
# 否则在官方规模 (image_size=224) 下渡渲染出来的杆/球会变得线细如发丝、
# 与画布不成比例（这是纯视觉缩放，不改变任何物理/坐标语义，只保证在更大分辨率
# 下渡渲染质量和 64x64 版本感知上一致，见 s5：官方规模验证）。
_PX_FIELDS = {
    "inertia_ball": ["ball_radius_px"],
    "pendulum": ["bob_radius_px", "rod_width_px", "pivot_radius_px"],
    "cartpole": ["pole_width_px", "pole_tip_radius_px"],
}


def make_scaled_config(env_name: str, image_size: int, seed=None, **extra):
    """按 image_size 构造环境 config，等比缩放装饰性像素字段（基准 64px）。"""
    spec = get_env_spec(env_name)
    scale = image_size / 64.0
    kwargs = dict(image_size=image_size, seed=seed)
    for field in _PX_FIELDS.get(env_name, []):
        default = spec.config_cls.__dataclass_fields__[field].default
        kwargs[field] = max(1, round(default * scale))
    kwargs.update(extra)
    return spec.config_cls(**kwargs)
