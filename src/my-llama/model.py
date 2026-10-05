import math
from typing import Any, Dict, List, Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from .config import ModelArgs
from .types import TransformerInput, TransformerOutput
from .ffn import FeedForward
from .moe import MoE

def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+eps)
    
    return _norm(x.float()).type_as(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super.__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return rmsnorm(x, self.eps)
    
def apply_scaling(freqs: torch.Tensor, scale_factor: float, high_freq_factor: float):
    low_freq_factor = 1
    old_context_len = 8192
    
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi /freq
        # 波长较小，高频，不需要放缩，否则相邻位置的token识别会模糊
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        # 波长较大，低频，需要放缩，扩大上下文容量
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq/scale_factor)
        # 中间地带需要smooth放缩
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype = freqs.dtype, device = freqs.device)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    use_scaled: bool,
    scale_factor: float,
    high_freq_factor: float,
):
    # RoPE核心公式，生成多组频率
    freqs = 1.0 /(theta **(torch.arange(0, dim, 2)[: (dim//2)].float() / dim))
    t = torch.arange(end, device = freqs.device, dtype = torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs,scale_factor,high_freq_factor)
    # 得到不同的角度
    freqs = torch.outer(t, freqs)
    
    # 生成复数组合代表角度
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis