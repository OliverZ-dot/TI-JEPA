"""官方规模 encoder：ViT-Tiny（复刻 `stable_pretraining.backbone.utils.vit_hf`，
见官方仓库 `config/train/model/lewm.yaml` + le-wm-main 里的
`stable_pretraining/backbone/utils.py::vit_hf`）。

官方配置（用户请求 §"扩大规模，比如尝试原本的LeWM的规模"要复刻的目标）：
    encoder:  size=tiny, patch_size=14, image_size=224, pretrained=false
    -> hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
       intermediate_size=hidden_size*4=768   (HuggingFace ViTConfig)

我们直接用 `transformers.ViTModel`（而不是 `stable_pretraining` 包本身）
复刻同一份 config，因为：
  1. `stable_pretraining` 在这台机器上没装，但可以从 PyPI 装；不过它还会拉
     lightning/wandb 等一整套依赖，风险和时间成本都比直接用 transformers 大。
  2. 官方 `vit_hf('tiny', patch_size=14, image_size=224, pretrained=False)`
     本质就是 `ViTModel(ViTConfig(hidden_size=192, num_hidden_layers=12,
     num_attention_heads=3, intermediate_size=768, image_size=224,
     patch_size=14), add_pooling_layer=False, use_mask_token=False)`——
     我们已经逐字核对过源码（checked against the official source），直接用 transformers 复刻是等价的。

输出：CLS token 的 last_hidden_state（(B, 192)），和官方 `jepa.py::JEPA.encode`
里 `output.last_hidden_state[:, 0]` 完全一致的取法。
"""

from __future__ import annotations

import torch
import torch.nn as nn


OFFICIAL_VIT_TINY_KWARGS = dict(
    hidden_size=192,
    num_hidden_layers=12,
    num_attention_heads=3,
    intermediate_size=192 * 4,
)


class ViTTinyBackbone(nn.Module):
    """official-scale 图像 encoder：ViT-Tiny/14, from-scratch（无 ImageNet 预训练，
    和官方 config 的 `pretrained: false` 一致——我们的自建环境渲染图和 ImageNet
    自然图像分布相差太远，加载预训练权重没有意义，官方自己也是从零训练）。
    """

    def __init__(self, image_size: int = 224, patch_size: int = 14, feat_dim: int = 192):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        assert feat_dim == OFFICIAL_VIT_TINY_KWARGS["hidden_size"], (
            "ViT-Tiny 的 hidden_size 固定是 192（官方 size='tiny' 预设），"
            f"传入的 feat_dim={feat_dim} 和官方规模不一致"
        )
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            use_mask_token=False,
            **OFFICIAL_VIT_TINY_KWARGS,
        )
        self.model = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
        self.feat_dim = feat_dim
        self.image_size = image_size

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """o: (B, 3, H, W) float in [0,1] -> (B, feat_dim) CLS token。"""
        out = self.model(o, interpolate_pos_encoding=True)
        return out.last_hidden_state[:, 0]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = ViTTinyBackbone()
    n = m.num_params()
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(f"ViTTinyBackbone params={n/1e6:.2f}M  out_shape={tuple(y.shape)}")
