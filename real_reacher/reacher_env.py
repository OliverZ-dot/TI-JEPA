"""Thin, planning-oriented wrapper around the official dm_control Reacher env
(`ReacherDMControlWrapper`), mirroring the role `InertiaBall`/`PushT` play for
the other two Protocol-C venues:

  - `make_env(damping)`: builds one instance and (optionally, once) overrides
    `dof_damping` -- a MODEL parameter, so unlike `qpos`/`qvel` it survives
    `env.reset()` (verified empirically: reset() only touches `data`, not
    `model`, unless `mark_dirty()` was called, which damping edits never do).
    This is eval-only and mirrors the exact same trick already used for real
    PushT's kill experiment (`damping=1.0` override there).
  - `set_state`/`get_state`: (qpos, qvel) <-> raw env state, thin renames of
    the wrapper's own `set_state`/`physics.data.{qpos,qvel}` for symmetry
    with `InertiaBall.set_state/get_state`.
  - `step_raw(action)`: advances physics directly via `physics.set_control` +
    `physics.step()` (bypassing the gym `.step()`'s obs/reward/info
    bookkeeping), matching the wrapper's own `action_repeat=2` semantics --
    this is what makes batched/serial CEM candidate evaluation fast enough
    (~40us/iter measured) to just loop the REAL simulator instead of hand-
    deriving vectorized dynamics (as InertiaBall's `oracle_dynamics_batch`
    had to).
  - `render_at(qpos)`: teleport-render a single frame at a given qpos (qvel
    is invisible to rendering by construction -- there is no motion blur --
    which is exactly the point: a single frame cannot carry velocity).

Needs `MUJOCO_GL=egl` in the environment for headless GPU rendering.
"""

from __future__ import annotations

import os

import sys

import numpy as np

LEWM_REPO = os.environ.get("LEWM_REPO", "./le-wm")
if LEWM_REPO not in sys.path:
    sys.path.insert(0, LEWM_REPO)

from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper  # noqa: E402

SHOULDER_RANGE = (-np.pi, np.pi)       # unlimited hinge, we just cap sampling
WRIST_RANGE = (-2.6, 2.6)              # true limit +-2.7925, small safety margin
ACTION_REPEAT = 2                       # matches DMControlWrapper.action_repeat
STEP_DT = 0.02 * ACTION_REPEAT          # 0.04s per wrapped env.step()


def make_env(damping: float = 0.001, task: str = "qpos_match", seed: int = 0):
    env = ReacherDMControlWrapper(task=task, seed=seed)
    env.reset(seed=seed)
    if damping is not None:
        env.env.physics.model.dof_damping[:] = damping
    return env


def set_state(env, qpos, qvel):
    env.env.physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    env.env.physics.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    env.env.physics.forward()


def get_state(env):
    return env.env.physics.data.qpos.copy(), env.env.physics.data.qvel.copy()


def step_raw(env, action):
    """Advance exactly one wrapped-env step (ACTION_REPEAT physics substeps),
    action in ctrl units [-1,1] (same convention CEM/model actions use)."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    for _ in range(ACTION_REPEAT):
        env.env.physics.set_control(a)
        env.env.physics.step()


def render_at(env, qpos, width=64, height=64):
    q0, v0 = get_state(env)
    set_state(env, qpos, np.zeros_like(v0))
    img = env.render(width=width, height=height)
    set_state(env, q0, v0)
    return img


def rollout_real(env, qpos0, qvel0, action_seq):
    """action_seq: (H,2) -> final (qpos,qvel), and full per-step trajectory."""
    set_state(env, qpos0, qvel0)
    qs = [np.array(qpos0, dtype=np.float64)]
    vs = [np.array(qvel0, dtype=np.float64)]
    for a in action_seq:
        step_raw(env, a)
        q, v = get_state(env)
        qs.append(q)
        vs.append(v)
    return np.stack(qs), np.stack(vs)


def oracle_rollout_batch(env, qpos0, qvel0, action_seqs):
    """action_seqs: (N,H,2) -> final (N,2) qpos, (N,2) qvel. Loops the REAL
    simulator serially per candidate (fast enough: ~40us/substep pair, see
    module docstring) -- exact ground truth, no hand-derived approximation."""
    N, H, _ = action_seqs.shape
    final_q = np.zeros((N, 2), dtype=np.float64)
    final_v = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        set_state(env, qpos0, qvel0)
        for t in range(H):
            step_raw(env, action_seqs[i, t])
        q, v = get_state(env)
        final_q[i], final_v[i] = q, v
    return final_q, final_v
