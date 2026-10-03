"""One-time preprocessing for the REAL dm_control Reacher dataset
(the dm_control Reacher HDF5 dataset, path set via the `REACHER_H5` env var; 2.01M frames,
10000 episodes of exactly 201 steps, native 224x224x3 uint8 pixels, random
torque actions).

Unlike `build_cache.py` (real PushT) we do NOT downsize pixels -- they are
already native 224x224, exactly the official ViT-Tiny/14 input resolution,
so no resize step is needed (one less lossy operation).

Decimates every `frameskip` raw steps (raw per-step |dqpos| ~ 0.05-0.06 rad,
barely visible at 224px; frameskip=3 gives ~0.15-0.2 rad/step, a clearly
visible arm rotation, while keeping >=60 decimated frames/episode for
plenty of k=3 windows).

Also carries `qpos`/`qvel` (2-dim each; qpos is RAW/unwrapped, see
reacher_data.py for the sin/cos periodic-safe pose target used downstream)
and `finger_pos`/`target_pos` (2-dim each, task-relevant) for eval-only
probes -- never touched by the training loop itself.
"""

from __future__ import annotations

import argparse
import os

import h5py
import hdf5plugin  # noqa: F401  -- registers the HDF5 compression filter plugins reacher_random.h5 needs
import numpy as np

H5_PATH = os.environ.get("REACHER_H5", os.path.expanduser(
    "~/.stable_worldmodel/datasets/dmc/reacher_random.h5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=4000)
    ap.add_argument("--frameskip", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="cache/reacher_cache.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    f = h5py.File(H5_PATH, "r")
    ep_len = f["ep_len"][:]         # all == 201
    ep_offset = f["ep_offset"][:]
    n_ep_total = len(ep_len)
    g = np.random.default_rng(args.seed)
    ep_ids = g.permutation(n_ep_total)[: args.n_episodes]
    ep_ids = np.sort(ep_ids)  # ascending file-offset order -> far more sequential h5 reads

    all_pixels, all_actions, all_qpos, all_qvel = [], [], [], []
    all_finger, all_target = [], []
    ep_starts, ep_lens_out = [], []
    ptr = 0
    for i, ep in enumerate(ep_ids):
        L = int(ep_len[ep])
        off = int(ep_offset[ep])
        n_dec = L // args.frameskip
        if n_dec < 5:
            continue
        # IMPORTANT perf note: h5py strided fancy-indexed reads (step>1) on this
        # dataset are ~10x slower than a single contiguous read of the same
        # span (measured: 1.6s vs 0.15-0.27s per 201-frame episode) -- read the
        # full contiguous block once, then decimate with plain numpy slicing.
        idx_end = off + n_dec * args.frameskip
        pix_full = f["pixels"][off:idx_end]              # (n_dec*frameskip,224,224,3) uint8
        act_full = f["action"][off:idx_end]               # (n_dec*frameskip, 2)
        qpos_full = f["qpos"][off:idx_end]
        qvel_full = f["qvel"][off:idx_end]
        finger_full = f["finger_pos"][off:idx_end]
        target_full = f["target_pos"][off:idx_end]

        pix = pix_full[:: args.frameskip]
        act = act_full.reshape(n_dec, args.frameskip, 2).mean(axis=1).astype(np.float32)
        qpos = qpos_full[:: args.frameskip].astype(np.float32)
        qvel = qvel_full[:: args.frameskip].astype(np.float32)
        finger = finger_full[:: args.frameskip].astype(np.float32)
        target = target_full[:: args.frameskip].astype(np.float32)

        all_pixels.append(pix)
        all_actions.append(act)
        all_qpos.append(qpos)
        all_qvel.append(qvel)
        all_finger.append(finger)
        all_target.append(target)
        ep_starts.append(ptr)
        ep_lens_out.append(n_dec)
        ptr += n_dec

        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(ep_ids)} episodes, {ptr} decimated frames so far", flush=True)

    pixels = np.concatenate(all_pixels, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    qpos = np.concatenate(all_qpos, axis=0)
    qvel = np.concatenate(all_qvel, axis=0)
    finger = np.concatenate(all_finger, axis=0)
    target = np.concatenate(all_target, axis=0)
    ep_starts = np.array(ep_starts, dtype=np.int64)
    ep_lens_out = np.array(ep_lens_out, dtype=np.int64)

    print(f"final cache: pixels={pixels.shape} actions={actions.shape} qpos={qpos.shape} "
          f"({pixels.nbytes/1e9:.2f} GB pixels)", flush=True)
    np.savez(args.out, pixels=pixels, actions=actions, qpos=qpos, qvel=qvel,
              finger_pos=finger, target_pos=target,
              ep_starts=ep_starts, ep_lens=ep_lens_out,
              frameskip=args.frameskip, image_size=224)
    print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
