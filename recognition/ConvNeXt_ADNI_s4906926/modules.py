"""
Model components for ADNI AD vs NC classification.

* BaselineCNN - a deliberately simple reference model.
* ConvNeXt    - "A ConvNet for the 2020s" (Liu et al., CVPR 2022), implemented
                from scratch below. ImageNet-1k pretrained weights can be loaded
                into this implementation for transfer learning.

All model code uses PyTorch only (no NumPy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2). Halves the spatial size."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """Four-stage plain CNN baseline.

    Input : (B, 1, 224, 224) grayscale MRI slice
    Output: (B, num_classes) logits (NC=0, AD=1)
    """

    def __init__(self, in_ch=1, num_classes=2, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, prev = [], in_ch
        for w in widths:                      # 224 -> 112 -> 56 -> 28 -> 14
            layers.append(ConvBlock(prev, w))
            prev = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling -> (B, C, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# ConvNeXt building blocks
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of a channels-first (B, C, H, W) tensor.

    nn.LayerNorm normalises the LAST dimension, so we move channels last,
    normalise, and move them back.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)                                   # (B, H, W, C)
        x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)                                # (B, C, H, W)


class DropPath(nn.Module):
    """Stochastic depth: randomly drop the whole residual branch per sample
    during training (regularisation for deep residual networks)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # One Bernoulli draw per sample, broadcast over (C, H, W)
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep                                      # rescale to keep expectation


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (inverted bottleneck with a large depthwise kernel).

        x ─► DWConv 7x7 ─► LayerNorm ─► Linear (C→4C) ─► GELU ─► Linear (4C→C)
          │                                                         │
          │                                          × gamma (Layer Scale)
          │                                                         │
          └──────────────────────── + ◄──── DropPath ◄──────────────┘

    The two Linear layers act as 1x1 convolutions on a channels-last tensor.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise: spatial mixing
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)                                   # expand (channel mixing)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)                                   # project back
        # Layer Scale: learnable per-channel scaling, initialised near zero so each
        # block starts close to an identity mapping (stabilises training).
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)                 # (B, C, H, W) -> (B, H, W, C)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)                 # back to (B, C, H, W)
        return shortcut + self.drop_path(x)       # residual connection


class ConvNeXt(nn.Module):
    """ConvNeXt classifier.

    Layout (ConvNeXt-T: depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)):

        Stem (patchify 4x4/4)     224 -> 56
        Stage 1  x3 blocks        56x56,  96 ch
        Downsample (2x2/2)        -> 28
        Stage 2  x3 blocks        28x28, 192 ch
        Downsample                -> 14
        Stage 3  x9 blocks        14x14, 384 ch
        Downsample                -> 7
        Stage 4  x3 blocks         7x7,  768 ch
        Global avg pool -> LayerNorm -> Linear -> logits

    Module names follow the official implementation so that published
    ImageNet checkpoints can be loaded directly.
    """

    def __init__(self, in_chans=1, num_classes=2, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0.0, layer_scale_init=1e-6):
        super().__init__()
        # downsample_layers[0] is the stem; [1..3] sit between stages.
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),   # patchify: non-overlapping 4x4 patches
            LayerNorm2d(dims[0]),
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # Drop-path probability increases linearly with depth.
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dims[i], rates[cur + j], layer_scale_init) for j in range(depths[i])
            ]))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x))
        return self.norm(x.mean(dim=(-2, -1)))    # global average pooling + LayerNorm -> (B, C)

    def forward(self, x):
        return self.head(self.forward_features(x))


# ---------------------------------------------------------------------------
# Pretrained weights
# ---------------------------------------------------------------------------
CONVNEXT_TINY_URL = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"


def _timm_to_official(sd):
    """Rename a timm ConvNeXt state dict to the official / this file's naming."""
    out = {}
    for k, v in sd.items():
        k = k.replace("stem.", "downsample_layers.0.")
        if ".downsample." in k:                                   # stages.i.downsample.j -> downsample_layers.i.j
            i, rest = k.split(".downsample.")
            k = f"downsample_layers.{i.split('.')[1]}.{rest}"
        k = (k.replace(".blocks.", ".").replace("conv_dw", "dwconv")
              .replace("mlp.fc1", "pwconv1").replace("mlp.fc2", "pwconv2")
              .replace("head.norm", "norm").replace("head.fc", "head"))
        out[k] = v
    return out


def _fetch_imagenet_state_dict():
    """Download ConvNeXt-T ImageNet-1k weights.

    First tries the official checkpoint (pure PyTorch download); if that is
    unreachable, falls back to the identical weights hosted by timm.
    Both are cached after the first download.
    """
    try:
        ckpt = torch.hub.load_state_dict_from_url(CONVNEXT_TINY_URL, map_location="cpu")
        return ckpt["model"]
    except Exception as e:
        print(f"Official checkpoint unavailable ({e}); falling back to timm.")
        import timm                                               # only used to fetch weights
        return _timm_to_official(timm.create_model("convnext_tiny.fb_in1k", pretrained=True).state_dict())


def load_imagenet_weights(model):
    """Load ImageNet weights into a 1-channel ConvNeXt-T.

    * Stem (1-channel model): RGB filters are summed over the colour axis.
      Feeding a grayscale image x to this filter equals feeding (x, x, x) to
      the original RGB filter. A 3-channel (2.5D) model uses the filters as-is.
    * Head: the 1000-class ImageNet classifier is discarded; ours stays random.
    """
    sd = _fetch_imagenet_state_dict()
    stem_w = sd["downsample_layers.0.0.weight"]                 # (96, 3, 4, 4)
    in_ch = model.downsample_layers[0][0].in_channels
    if stem_w.shape[1] != in_ch:
        sd["downsample_layers.0.0.weight"] = stem_w.sum(dim=1, keepdim=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"head.weight", "head.bias"} and not unexpected, (missing, unexpected)
    print("Loaded ImageNet-1k ConvNeXt-T weights (new classification head).")
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(name, num_classes=2, pretrained=False, drop_path=0.1, in_chans=1):
    """Select a model by name so train.py / predict.py stay model-agnostic.

    in_chans=1 for single slices, 3 for 2.5D input (three adjacent slices).
    """
    if name == "baseline":
        return BaselineCNN(in_ch=in_chans, num_classes=num_classes)
    if name == "convnext":
        model = ConvNeXt(in_chans=in_chans, num_classes=num_classes, drop_path_rate=drop_path)
        if pretrained:
            load_imagenet_weights(model)
        return model
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
