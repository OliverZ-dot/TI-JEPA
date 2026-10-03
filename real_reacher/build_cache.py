"""One-time preprocessing for dm_control Reacher, mirroring
`real_pusht/build_cache.py`: sample N episodes from the official
`dmc/reacher_random.h5`, resize pixels, and dump a compact in-RAM cache so
training doesn't pay per-sample random-H5-access cost.

frameskip=1 here (unlike PushT's 5): the dataset is already recorded at the
wrapped env's resolution (physics timestep 0.02s x action_repeat=2 = 0.04s
per row), which is also the exact step granularity the Protocol-C planner
below uses, so keeping frameskip=1 keeps train-time and plan-time dynamics
resolution identical (no risk of the model learning a coarser-than-planning
notion of "one step").

state = concat(qpos, qvel) (4-dim) -- unlike PushT this is only used for
downstream probes/diagnostics, never for training (train_baseline /
train_tijepa in real_train.py only touch pixels+actions).
"""

from __future__ import annotations

import argparse

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

H5_PATH = os.environ.get("REACHER_H5", os.path.expanduser(
    "~/.stable_worldmodel/datasets/dmc/reacher_random.h5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=4000)
    ap.add_argument("--frameskip", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="cache/reacher_cache.npz")
    args = ap.parse_args()

    import os
    os.makedirs("cache", exist_ok=True)

    f = h5py.File(H5_PATH, "r")
    ep_len = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    n_ep_total = len(ep_len)
    g = np.random.default_rng(args.seed)
    ep_ids = g.permutation(n_ep_total)[: args.n_episodes]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_pixels, all_actions, all_state, ep_starts, ep_lens_out = [], [], [], [], []
    ptr = 0
    for i, ep in enumerate(ep_ids):
        L = int(ep_len[ep])
        off = int(ep_offset[ep])
        n_dec = L // args.frameskip
        if n_dec < 5:
            continue
        pix = f["pixels"][off : off + n_dec * args.frameskip : args.frameskip]  # (n_dec,224,224,3)
        pix_t = torch.from_numpy(pix).to(device).permute(0, 3, 1, 2).float() / 255.0
        pix_small = torch.nn.functional.interpolate(pix_t, size=(args.image_size, args.image_size),
                                                       mode="bilinear", align_corners=False)
        pix_u8 = (pix_small.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # (n_dec,S,S,3)

        if args.frameskip == 1:
            act = f["action"][off : off + n_dec]
        else:
            act = f["action"][off : off + n_dec * args.frameskip]
            act = act.reshape(n_dec, args.frameskip, 2).mean(axis=1)

        qpos = f["qpos"][off : off + n_dec * args.frameskip : args.frameskip]
        qvel = f["qvel"][off : off + n_dec * args.frameskip : args.frameskip]
        st = np.concatenate([qpos, qvel], axis=-1)  # (n_dec,4)

        all_pixels.append(pix_u8)
        all_actions.append(act.astype(np.float32))
        all_state.append(st.astype(np.float32))
        ep_starts.append(ptr)
        ep_lens_out.append(n_dec)
        ptr += n_dec

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ep_ids)} episodes, {ptr} decimated frames so far", flush=True)

    pixels = np.concatenate(all_pixels, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    state = np.concatenate(all_state, axis=0)
    ep_starts = np.array(ep_starts, dtype=np.int64)
    ep_lens_out = np.array(ep_lens_out, dtype=np.int64)

    print(f"final cache: pixels={pixels.shape} actions={actions.shape} state={state.shape} "
          f"({pixels.nbytes/1e9:.2f} GB pixels)", flush=True)
    np.savez(args.out, pixels=pixels, actions=actions, state=state,
              ep_starts=ep_starts, ep_lens=ep_lens_out, frameskip=args.frameskip, image_size=args.image_size)
    print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
