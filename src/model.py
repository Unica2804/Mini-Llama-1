import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import xformers.ops as xops
from dataclasses import dataclass

try:
    from utils.config import get_config
except ModuleNotFoundError:
    from config import get_config

default_config = get_config()

@dataclass
class LLamaConfig:
    d_model: int = default_config["d_model"]
    n_layers: int = default_config["n_layers"]
    n_heads: int = default_config["n_heads"]
    vocab_size: int = default_config["vocab_size"]
    multiple_of: int = 256
    norm_eps: float = 1e-6
    max_seq_len: int = default_config["max_seq_len"]
    seq_len: int = default_config["seq_len"]
    rope_base: float = 10000.0

    @property
    def d_ff(self) -> int:
        hidden_dim = int(2*(4*self.d_model)/3)
        return self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class Embedding(nn.Module):
    def __init__(self, config: LLamaConfig):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, config: LLamaConfig):
        super(RMSNorm, self).__init__()
        self.eps = config.norm_eps
        self.scale = nn.Parameter(torch.ones(config.d_model))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / norm)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, config: LLamaConfig):
        super().__init__()
        self.d_model = config.head_dim
        self.max_pos_emb = config.max_seq_len
        self.base = config.rope_base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d_model, 2).float() / self.d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=self.max_pos_emb)
    
    def _set_cos_sin_cache(self, seq_len, device=None, dtype=None):
        device = device or self.inv_freq.device
        dtype = dtype or self.inv_freq.dtype
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype), persistent=False)

    def forward(self,x,seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len,device=x.device,dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.device).to(x.dtype),
            self.sin_cached[:seq_len].to(x.device).to(x.dtype)
        )
    
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, pos_ids):
    cos = cos[pos_ids].unsqueeze(2)
    sin = sin[pos_ids].unsqueeze(2)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMultiHeadAttention(nn.Module):
    def __init__(self,config: LLamaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = config.head_dim

        self.wqkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary_pos_emb = RotaryPositionalEncoding(config)

    def forward(self, x, pos_ids=None):
        B, S, _ = x.shape

        if pos_ids is None:
            pos_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        
        qkv = self.wqkv(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2) #each is B,S,H,D contigious view
        cos, sin = self.rotary_pos_emb(v, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

        out = xops.memory_efficient_attention(
            q,k,v,
            attn_bias=xops.fmha.attn_bias.LowerTriangularMask(),
            p=0.0,
        )
        out = out.reshape(B, S, self.d_model)
        return self.wo(out)

class LlamaFeedForward(nn.Module):
    def __init__(self, config: LLamaConfig):
        super().__init__()
        self.wgate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wup = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wdown = nn.Linear(config.d_ff, config.d_model, bias=False)
    def forward(self, x):
        return self.wdown(F.silu(self.wgate(x)) * self.wup(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LLamaConfig):
        super().__init__()
        self.self_attn = LlamaMultiHeadAttention(config)
        self.norm1 = RMSNorm(config)
        self.ffn = LlamaFeedForward(config)
        self.norm2 = RMSNorm(config)

    def forward(self, x, pos_ids=None):
        x = x + self.self_attn(self.norm1(x), pos_ids)
        x = x + self.ffn(self.norm2(x))
        return x

class ProjectionLayer(nn.Module):
    def __init__(self,config: LLamaConfig):
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config)
        self.output = ProjectionLayer(config)

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, mean=0.0, std=0.02)

    def forward(self, tokens, pos_ids=None):
        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x, pos_ids)

        return self.output(self.norm(x))

def make_model(config: LLamaConfig | None = None):
    config = config or LLamaConfig()
    return LlamaModel(config)





