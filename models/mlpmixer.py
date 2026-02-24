import torch
import torch.nn as nn
import torch.nn.functional as F


# Patch Embedding: (B, C, H, W) -> (B, N, D) B: batch, N = num_patches, D = embed_dim
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        patch_size = self.patch_size
        assert height % patch_size == 0 and width % patch_size == 0, "H and W must be divisible by patch_size."

        grid_h = height // patch_size
        grid_w = width // patch_size
        num_patches = grid_h * grid_w

        # reshape (B, C, H, W) -> (B, C, H/P, P, W/P, P)
        x = x.reshape(batch_size, in_channels, grid_h, patch_size, grid_w, patch_size)
        # reshape (B, C, H/P, P, W/P, P) -> (B, H/P, W/P, C, P, P), with continuous tensors in memory
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # reshape (B, H/P, W/P, C, P, P) -> (B, N, C*P*P)
        x = x.reshape(batch_size, num_patches, in_channels * patch_size * patch_size)
        # reshape (B, N, C*P*P) -> (B, N, D)
        x = self.proj(x)

        return x


# 2-layer MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return out


# Mixer layer: Token mixing + Channel mixing
class MixerLayer(nn.Module):
    def __init__(self, num_patches, embed_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.token_mlp = MLP(num_patches, token_mlp_dim, num_patches)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.channel_mlp = MLP(embed_dim, channel_mlp_dim, embed_dim)

    def forward(self, x):
        # Token mixing
        y = self.ln1(x)
        y = y.transpose(1, 2)  # (B, D, N)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # (B, N, D)
        x = x + y

        # Channel mixing
        y = self.ln2(x)
        y = self.channel_mlp(y)
        x = x + y

        return x


class MLPMixer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, depth=6, token_mlp_dim=64, channel_mlp_dim=512, dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size."
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(MixerLayer(num_patches, embed_dim, token_mlp_dim, channel_mlp_dim))

        self.ln = nn.LayerNorm(embed_dim)
        self.num_features = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        out = self.patch_embed(x)

        for block in self.blocks:
            out = block(out)

        out = self.ln(out)
        out = out.mean(dim=1)  # Global average pooling for patches (B, N, D) -> (B, D)

        return out

    def forward(self, x):
        features = self.forward_features(x)
        out = self.classifier(features)

        return out


class MLPMixerEncoder(nn.Module):
    def __init__(self, **mixer_kwargs):
        super().__init__()
        self.backbone = MLPMixer(**mixer_kwargs)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone.forward_features(x)


def mlpmixer_encoder():
    return MLPMixerEncoder(
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        depth=6,
        token_mlp_dim=64,
        channel_mlp_dim=512,
        num_classes=10,
    )
