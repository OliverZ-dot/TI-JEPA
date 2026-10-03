"""Calibration: how much does ignoring v0 cost (position error after H steps
if the planner just assumes v0=0), vs. how much positional correction can a
given action budget `a_max` (ctrl clip) buy in the same H steps? Mirrors the
audit the explore agent did for the default (undamped) config, but now
re-run under our chosen eval-only low-damping override, to pick a_max/damping
that make Protocol C actually test velocity-awareness rather than
brute-force override (exactly the InertiaBall a_max lesson from Protocol C's
first fix)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from reacher_env import make_env, oracle_rollout_batch, rollout_real


def main():
    rng = np.random.default_rng(0)
    for damping in [0.001, 0.005, 0.01]:
        env = make_env(damping=damping)
        for H in [5, 8, 10]:
            drift_ignore_v0 = []
            for _ in range(60):
                q0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)])
                shoulder_v = rng.normal(0, 1.35)
                wrist_v = rng.normal(0, 2.4)
                v0 = np.array([shoulder_v, wrist_v])
                zero_actions = np.zeros((H, 2))
                q_traj, _ = rollout_real(env, q0, v0, zero_actions)
                q_traj_novel, _ = rollout_real(env, q0, np.zeros(2), zero_actions)
                drift_ignore_v0.append(np.linalg.norm(q_traj[-1] - q_traj_novel[-1]))
            drift_ignore_v0 = np.array(drift_ignore_v0)

            for a_max in [0.02, 0.05, 0.1, 0.2]:
                max_swings = []
                for _ in range(20):
                    q0 = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0)])
                    v0 = np.zeros(2)
                    full_action = np.tile(np.array([a_max, a_max]), (H, 1))
                    q_traj, _ = rollout_real(env, q0, v0, full_action)
                    max_swings.append(np.linalg.norm(q_traj[-1] - q_traj[0]))
                max_swings = np.array(max_swings)
                ratio = np.median(max_swings) / (np.median(drift_ignore_v0) + 1e-6)
                print(f"damping={damping} H={H} a_max={a_max}: "
                      f"ignore_v0_drift(median)={np.median(drift_ignore_v0):.4f} "
                      f"action_swing(median)={np.median(max_swings):.4f} ratio={ratio:.2f}")
        env.close()


if __name__ == "__main__":
    main()
