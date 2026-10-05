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

from .args import ModelArgs
from .datatypes import TransformerInput, TransformerOutput
from .ffn import FeedForward
from .moe import MoE

from config import ModelArgs
from model import apply_rotary_emb
from model import rmsnorm
class Attention(nn.Module):
    def __init__(
        self,
        args: Model_Args,
        use_qk_norm: bool,
        use_rope: bool,
        add_bias: bool = False,
    ):
        super().__init__()
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        
        # Attention temperature tuning
        self.attn_temperature_tuning = args.attn_temperature_tuning # bool
        self.floor_scale = args.floor_scale # 温度调整的序列长度下界，小于下界时不调整，大于时进行调整
        self.attn_scale = args.attn_scale # QK/sqrt(d)后再次*scale
        
        self.n_heads = args.n_heads
        
        # MHA: 多头注意力： n_kv_heads= n_heads
        # GQA: 分组查询注意力：1 < n_kv_heads < n_heads Llama用这个
        # MQA: 多查询注意力： 1 = n_kv_heads < n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads 
        world_size = fs_init.get_model_parallel_world_size() # GPU数量
        self.n_local_heads = self.n_heads // world_size # 当前GPU上Q头数
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_kv_heads # 每个K/V头被多少个Q头共享
        self.head_dim = args.dim // self.n_heads
        
        # Wq按列拆，之后计算的时候是x @ Wq,各列独立
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias = add_bias,
            gather_output = False,
            init_method = lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias = add_bias,
            gather_output = False,
            init_method = lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias = add_bias,
            gather_output = False,
            init_method = lambda x: x,
        )
        
        # 按行拆，计算是 attn @ Wo，如果attn是按列拆的，那W按行拆可以让每张GPU上各个attn的分量和Wo的分量对应
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias = add_bias,
            input_is_parallel = True,
            init_method = lambda x: x,
        )
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.norm_eps = args.norm_eps
        
        # 加载权重时预处理
        self.register_load_state_dict_pre_hook(self.load_hook)
        
    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        # 把wqkv大矩阵拆成wq、wk和wv
        if prefix + "wqkv.weight" in state_dict:
            wqkv = state_dict.pop(prefix + "wqkv.weight")
            d, r = divmod(wqkv.shape[0], self.n_heads + 2 * self.n_kv_heads)
            if r != 0:
                raise ValueError(
                    f"shape={tuple(wqkv.shape)} is not divisible by "
                    f"n_heads ({self.n_heads}) + 2 * n_kv_heads ({self.n_kv_heads})"
                )
            wq, wk, wv = wqkv.split([d * self.n_heads, d * self.n_kv_heads, d * self.n_kv_heads], dim=0)
            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv
        
    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        if self.use_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        if self.use_qk_norm:
            xq, xk = rmsnorm(xq, self.norm_eps), rmsnorm(xk, self.norm_eps)
            
        if self.attn_temperature_tuning and not self.use_rope:
            seq_positions = torch.arange(start_pos, start_pos + seq_len, device = xq.device, dtype = torch.float32)
            attn_scales = torch.log(torch.floor((seq_positions + 1.0) / self.floor_scale) +1.0) * self.attn_scale + 1.0
            
            attn_scales = attn_scales.view(1, seq_len, 1, 1)
            xq = xq * attn_scales
        
        # 类型转换
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        
        # 写入本次的
        self.cache_k[:bsz, start_pos: seq_len+start_pos] = xk
        self.cache_v[:bsz, start_pos: seq_len+start_pos] = xv
        
        # 读全部
        xk = self.cache_k[:bsz, : seq_len+start_pos]
        xv = self.cache_v[:bsz, : seq_len+start_pos]
        
        xq, xk, xv = [t.transpose(1, 2) for t in (xq, xk, xv)]
        
        # K和V要在n_kv_heads上重复到n_heads次，才能和Q保持一致
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        
        attn_output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask = mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.wo(attn_output)
        return output