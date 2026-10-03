"""快速校准 Pendulum / CartPole 的物理常数：跑几条随机轨迹，检查数值稳定性、
角度/速度覆盖范围是否合理，并存一张montage图肉眼检查渲染。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from ti_jepa.envs.pendulum import Pendulum, PendulumConfig, make_pendulum_action_sequence
from ti_jepa.envs.cartpole import CartPole, CartPoleConfig, make_cartpole_action_sequence


def check_pendulum():
    print("=== Pendulum ===")
    cfg = PendulumConfig(image_size=64)
    env = Pendulum(cfg)
    rng = np.random.default_rng(0)
    frames_all = []
    thetas, omegas = [], []
    for ep in range(20):
        ep_rng = np.random.default_rng(ep)
        f0 = env.reset(ep_rng)
        acts = make_pendulum_action_sequence(ep_rng, 40)
        fs = [f0]
        for a in acts:
            fs.append(env.step(a))
            th, om = env.get_raw_state()
            thetas.append(th)
            omegas.append(om)
        if ep < 6:
            frames_all.append(fs[0])
            frames_all.append(fs[20])
            frames_all.append(fs[-1])
    thetas = np.array(thetas)
    omegas = np.array(omegas)
    print(f"theta range: [{thetas.min():.2f}, {thetas.max():.2f}]  std={thetas.std():.2f}")
    print(f"omega range: [{omegas.min():.3f}, {omegas.max():.3f}]  std={omegas.std():.3f}")
    print(f"any NaN: {np.any(np.isnan(thetas)) or np.any(np.isnan(omegas))}")
    montage = np.concatenate(frames_all, axis=1)
    Image.fromarray(montage).save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pendulum_sanity.png"))
    print("saved pendulum_sanity.png", montage.shape)


def check_cartpole():
    print("=== CartPole ===")
    cfg = CartPoleConfig(image_size=64)
    env = CartPole(cfg)
    frames_all = []
    xs, x_dots, thetas, theta_dots = [], [], [], []
    n_fallen = 0
    for ep in range(20):
        ep_rng = np.random.default_rng(ep)
        f0 = env.reset(ep_rng)
        acts = make_cartpole_action_sequence(ep_rng, 40)
        fs = [f0]
        for a in acts:
            fs.append(env.step(a))
            x, x_dot, th, om = env.get_raw_state()
            xs.append(x); x_dots.append(x_dot); thetas.append(th); theta_dots.append(om)
        if abs(env.theta) > np.pi / 2:
            n_fallen += 1
        if ep < 6:
            frames_all.append(fs[0])
            frames_all.append(fs[20])
            frames_all.append(fs[-1])
    xs, x_dots, thetas, theta_dots = map(np.array, (xs, x_dots, thetas, theta_dots))
    print(f"x range: [{xs.min():.2f}, {xs.max():.2f}]  std={xs.std():.2f}")
    print(f"x_dot range: [{x_dots.min():.2f}, {x_dots.max():.2f}]  std={x_dots.std():.2f}")
    print(f"theta range: [{thetas.min():.2f}, {thetas.max():.2f}]  std={thetas.std():.2f}")
    print(f"theta_dot range: [{theta_dots.min():.2f}, {theta_dots.max():.2f}]  std={theta_dots.std():.2f}")
    print(f"episodes fallen (|theta|>pi/2) out of 20: {n_fallen}")
    print(f"any NaN: {any(np.any(np.isnan(a)) for a in (xs, x_dots, thetas, theta_dots))}")
    montage = np.concatenate(frames_all, axis=1)
    Image.fromarray(montage).save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_sanity.png"))
    print("saved cartpole_sanity.png", montage.shape)


if __name__ == "__main__":
    check_pendulum()
    check_cartpole()
