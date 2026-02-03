import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()

        # Patch embedding using 2d convolution with kernel_size=patch_size, stride=patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, D, H/P, W/P)
        out = self.proj(x)
        return out


class ConvMixerLayer(nn.Module):
    def __init__(self, input_dim, kernel_size=9):
        super().__init__()

        # Depthwise convolution (spatial mixing)
        self.dwconv = nn.Conv2d(input_dim, input_dim, kernel_size, groups=input_dim, padding=kernel_size // 2)
        # Pointwise convolution (channel mixing)
        self.pwconv = nn.Conv2d(input_dim, input_dim, kernel_size=1)

        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        # Depthwise convolution + Residual
        y = self.dwconv(x)
        y = self.gelu(y)
        y = self.bn1(y)
        x = x + y

        # Pointwise convolution
        x = self.pwconv(x)
        x = self.gelu(x)
        x = self.bn2(x)

        return x


class ConvMixer(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, depth=6, kernel_size=9):
        super().__init__()

        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ConvMixerLayer(embed_dim, kernel_size))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, H, W) -> (B, C, 1, 1)
        self.num_features = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        # Patch embedding
        out = self.patch_embed(x)

        # ConvMixer Layer
        for block in self.blocks:
            out = block(out)

        # Global average pooling + classifier
        out = self.pool(out)
        out = out.flatten(1)

        return out

    def forward(self, x):
        features = self.forward_features(x)
        out = self.classifier(features)

        return out


class ConvMixerEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = ConvMixer(**kwargs)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone.forward_features(x)


def convmixer_encoder():
    return ConvMixerEncoder(
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        depth=6,
        kernel_size=9,
        num_classes=10,
    )
