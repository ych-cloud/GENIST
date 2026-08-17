import math
from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class GeneJointEmbedding(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        self.gene_name_ebd = nn.Parameter(
            torch.empty((input_size, hidden_dim)),
            requires_grad=True,
        )
        torch.nn.init.kaiming_uniform_(self.gene_name_ebd, a=math.sqrt(5))
        self.gene_count_ebd = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        torch.nn.init.xavier_uniform_(self.gene_count_ebd[0].weight)
        torch.nn.init.xavier_uniform_(self.gene_count_ebd[2].weight)

    def forward(self, x):
        gene_count_ebd = self.gene_count_ebd(x.squeeze(1).unsqueeze(2))
        gene_name_ebd = self.gene_name_ebd
        return torch.add(gene_count_ebd, gene_name_ebd)


class FinalLayer(nn.Module):
    """
    The final layer of the diffusion transformer.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 2, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return torch.permute(self.linear(x), (0, 2, 1))


class GeneralizedCausalAttention(nn.Module):
    """
    Multi-head attention that supports generalized causal masks.
    The mask should be boolean with shape [N, 1, Lq, Lk].
    """

    def __init__(self, dim, num_heads, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = norm_layer(self.head_dim)
        self.k_norm = norm_layer(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        n, length, dim = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            mask = torch.zeros_like(attn, dtype=attn.dtype)
            mask = mask.masked_fill(attn_mask.expand_as(attn), float("-inf"))
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(n, length, dim)
        return self.proj(x)


class CFBlock(nn.Module):
    """
    CausalFusion-style block with adaLN-Zero conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = GeneralizedCausalAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, attn_mask: Optional[torch.Tensor] = None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            attn_mask=attn_mask,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class GENISTModel(nn.Module):
    """
    GENIST uses gene joint embedding, timestep embedding, patch-condition
    embedding, and generalized causal attention blocks for autoregressive
    diffusion.
    """

    def __init__(
        self,
        input_size=200,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        label_size=512,
        learn_sigma=True,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.gene_joint_embed = GeneJointEmbedding(self.input_size, self.hidden_size)
        self.time_embed = TimestepEmbedder(self.hidden_size)
        self.label_embed = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.SiLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )
        self.blocks = nn.ModuleList(
            [CFBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(self.hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.label_embed[0].weight, std=0.02)
        nn.init.normal_(self.label_embed[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, attn_mask: Optional[torch.Tensor] = None, **kwargs):
        x = self.gene_joint_embed(x)
        t = self.time_embed(t)
        y = self.label_embed(y)
        c = t + y
        for block in self.blocks:
            x = block(x, c, attn_mask=attn_mask)
        return self.final_layer(x, c)


def GENIST(**kwargs):
    return GENISTModel(**kwargs)
