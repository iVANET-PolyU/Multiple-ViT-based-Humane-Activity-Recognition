import torch
from torch import nn
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=300, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # b, 301, 90, heads = 8
        b, n, _, h = *x.shape, self.heads

        # self.to_qkv(x): b, 301, 300*8*3
        # qkv: b, 301, 300*8
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # b, 301, 300, 8
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # dots:b, 301, 300, 300
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # attn:b, 301, 300, 300
        attn = dots.softmax(dim=-1)

        # 使用einsum表示矩阵乘法：
        # out:b, 301, 300, 8
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # out:b, 301, 300*8
        out = rearrange(out, 'b h n d -> b n (h d)')

        # out:b, 301, 90
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=300, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pos_embedding = nn.Parameter(torch.randn(1, dim_head + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        # (b,N,d) = (b, 300, 90*3)
        b, n, _ = x.shape

        # 多一个可学习的x_class，与输入concat在一起，一起输入Transformer的Encoder。(b,1,d)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x: (b, 301, 90*3)
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Encoding：(b,N+1,d)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer的输入维度x的shape是：(b,N+1,d)
        x = self.transformer(x)

        # (b,1,d)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class mlp_head(nn.Module):
    def __init__(self, dim):
        super(mlp_head, self).__init__()
        self.layerNorm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 2)

    def forward(self, x):
        x = self.layerNorm(x)
        out = self.linear(x)
        return out


