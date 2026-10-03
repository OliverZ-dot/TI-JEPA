"""Checkpoint loaders that (unlike `ti_jepa.eval.common.load_baseline/
load_tijepa`, which hardcode `image_size=64`) read `image_size` from the
checkpoint's own saved args -- needed since we ended up training Reacher
models at 96x96 (64x64 pixel decodability turned out too low, see
the results log)."""

from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys

import torch

sys.path.insert(0, _REPO_ROOT)
from ti_jepa.models import HistoryPredictor, LeWMStyleEncoder, TIJEPAEncoder, TIJEPAPredictor  # noqa: E402


def load_baseline(path: str, device: str = "cuda"):
    state = torch.load(path, map_location=device)
    args = state["args"]
    image_size = args.get("image_size", 64)
    encoder = LeWMStyleEncoder(image_size=image_size, z_dim=args["z_dim"]).to(device)
    predictor = HistoryPredictor(per_step_dim=args["z_dim"], k=args["k"], action_dim=2,
                                  out_dim=args["z_dim"]).to(device)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor, args


def load_tijepa(path: str, device: str = "cuda"):
    state = torch.load(path, map_location=device)
    args = state["args"]
    image_size = args.get("image_size", 64)
    encoder = TIJEPAEncoder(image_size=image_size, q_dim=args["q_dim"], v_dim=args["v_dim"],
                             k=args["k"]).to(device)
    predictor = TIJEPAPredictor(q_dim=args["q_dim"], v_dim=args["v_dim"], action_dim=2).to(device)
    encoder.load_state_dict(state["encoder"])
    predictor.load_state_dict(state["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor, args
