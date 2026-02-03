import torch
import torch.nn as nn
import torch.nn.functional as F


# Patch Embedding: (B, C, H, W) -> (B, N, D) B: batch, N = num_patches, D = embed_dim
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size."

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        patch_size = self.patch_size
        grid_size = self.grid_size
        num_patches = self.num_patches

        assert height == self.img_size and width == self.img_size, (
            f"Expected {(self.img_size, self.img_size)}, got {(height, width)}"
        )

        # reshape (B, C, H, W) -> (B, C, H/P, P, W/P, P)
        x = x.reshape(batch_size, in_channels, grid_size, patch_size, grid_size, patch_size)
        # reshape (B, C, H/P, P, W/P, P) -> (B, H/P, W/P, C, P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # reshape (B, H/P, W/P, C, P, P) -> (B, N, C*P*P)
        x = x.reshape(batch_size, num_patches, in_channels * patch_size * patch_size)
        # reshape (B, N, C*P*P) -> (B, N, D)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads=6, att_dropout=0.0, proj_dropout=0.0):
        super().__init__()

        # embed_dim = D, num_heads = H
        # each head has dimension d = D / H
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # scale factor: 1 / sqrt(d), which prevents dot-product from growing too large
        self.scale = self.head_dim ** -0.5

        # single linear layer to produce Q, K, V at once
        # (B, N, D) -> (B, N, 3D)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(att_dropout)

        # output projection W^O
        # (B, N, D) -> (B, N, D)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        # x: input token embeddings
        # shape: (B, N, D)
        batch_size, num_tokens, embed_dim = x.shape

        # (B, N, D) -> (B, N, 3D)
        qkv = self.qkv(x)

        # reshape for multi-head attention
        # (B, N, 3D) -> (B, N, 3, H, d)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (B, N, 3, H, d) -> (3, B, H, N, d) -> 3 X (B, H, N, d)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)

        # (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
        out = attn @ v

        # (B, H, N, d) -> (B, N, H, d)
        out = out.transpose(1, 2).contiguous()

        # (B, N, H, d) -> (B, N, D)
        out = out.reshape(batch_size, num_tokens, embed_dim)

        # Final linear layer (W^O)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=6, mlp_ratio=4.0, att_dropout=0.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MHA(embed_dim, num_heads, att_dropout, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        out = x + self.attn(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0, att_dropout=0.0, dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size."

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(EncoderBlock(embed_dim, num_heads, mlp_ratio, att_dropout, dropout))

        self.norm = nn.LayerNorm(embed_dim)
        self.num_features = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # [-0.04, 0.04] 범위 내에서 정규분포 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        batch_size = x.size(0)

        x = self.patch_embed(x)  # (B, C, H, W) -> (B, N, D)

        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (1, 1, D) -> (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1 + N, D)

        x = x + self.pos_embed  # (B, 1 + N, D) + (1, 1 + N, D) -> (B, 1 + N, D), using broadcasting
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        out = x[:, 0, :]  # (B, 1 + N, D) -> (B, D)

        return out

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)  # (B, D) -> (B, num_classes)

        return logits


class VisionTransformerEncoder(nn.Module):
    def __init__(self, **vit_kwargs):
        super().__init__()
        self.backbone = VisionTransformer(**vit_kwargs)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone.forward_features(x)


def vit_encoder():
    return VisionTransformerEncoder(
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=10,
    )
