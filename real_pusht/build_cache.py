"""One-time preprocessing: sample N real pusht_expert_train episodes,
decimate every `frameskip` raw steps (matching the official model's temporal
resolution), resize pixels to `image_size`, and dump a compact in-RAM-sized
cache (.npz-like via np.savez, uint8 pixels) so `real_train.py` doesn't pay
per-sample random-H5-access + per-sample resize cost during training
(that path measured ~5s/step -- far too slow for an 8k-step run; this cache
turns it into a single sequential pass done once).
"""

from __future__ import annotations

import argparse

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

H5_PATH = os.environ.get("PUSHT_H5", os.path.expanduser(
    "~/.stable_worldmodel/datasets/pusht_expert_train.h5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=4000)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--image_size", type=int, default=84)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="cache/pusht_cache.npz")
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
        idx = off + np.arange(n_dec) * args.frameskip
        pix = f["pixels"][off : off + n_dec * args.frameskip : args.frameskip]  # (n_dec,224,224,3)
        pix_t = torch.from_numpy(pix).to(device).permute(0, 3, 1, 2).float() / 255.0
        pix_small = torch.nn.functional.interpolate(pix_t, size=(args.image_size, args.image_size),
                                                       mode="bilinear", align_corners=False)
        pix_u8 = (pix_small.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # (n_dec,S,S,3)

        act = f["action"][off : off + n_dec * args.frameskip]  # (n_dec*frameskip, 2)
        act = act.reshape(n_dec, args.frameskip, 2).mean(axis=1)  # (n_dec,2) mean action per decimated step

        st = f["state"][off : off + n_dec * args.frameskip : args.frameskip]  # (n_dec,7)

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
