import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    """Copy from vitlucidrains_mod_ver1."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    HAViT attention with per-head learnable alpha.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, alpha_init=0.9):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        logit_init = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.full((heads,), logit_init))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, h=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        h_prev = torch.randn_like(raw) if h is None else h

        alpha = torch.sigmoid(self.alpha_logit).view(1, self.heads, 1, 1)
        blended = alpha * raw + (1.0 - alpha) * h_prev

        attn = torch.softmax(blended, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), blended

    def flops(self, input_shape, layer_index):
        batch_size, seq_len, dim = input_shape
        return 3 * batch_size * seq_len * dim - 1


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        H = None
        for attn, ff in self.layers:
            out, H = attn(x, H)
            x = x + out
            x = x + ff(x)
        return self.norm(x)

    def flops(self, input_shape):
        return sum(attn.flops(input_shape, i) for i, (attn, _) in enumerate(self.layers))


class havit_learnable_alpha(nn.Module):
    """Drop-in for vitlucidrains_mod_ver1 - identical constructor signature."""

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.pool = pool
        self.dim = dim

        ih, iw = self.image_size
        ph, pw = self.patch_size
        assert pool in {"cls", "mean"}
        assert ih % ph == 0 and iw % pw == 0

        num_patches = (ih // ph) * (iw // pw)
        patch_dim = channels * ph * pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ph, p2=pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.mlp_head(x.mean(dim=1) if self.pool == "mean" else x[:, 0])

    def flops(self, batch_size):
        num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )
        return self.transformer.flops((batch_size, num_patches + 1, self.dim))
