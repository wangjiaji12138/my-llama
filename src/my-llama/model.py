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
from .attention import Attention

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

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x: [B, seq_len, num_heads, head_dim]
    # freqs_cis: [seq_len, head_dim]
    # 需要把freqs_cis调整到x一样的维度才能进行矩阵乘法
    ndim = x.dim
    assert 0<= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # shape = [1, seq_len, 1, head_dim]
    shape = [d if i==1 or i == ndim - 1 else 1 for i,d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1],-1,2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # 复数乘法后view_as_real拆分最后一维为实部和虚部，flatten合并实部虚部
    # pytorch不支持直接将complex64转换为2*float
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq),xk_out.type_as(xk)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: ModelArgs
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // self.n_heads if args.head_dim is None else args.head_dim
        
        self.is_nope_layer = args.nope_layer_interval is not None and (layer_id+1) % args.nope_layer_interval==0
        
        use_rope = not self.is_nope_layer
        use_qk_norm = args.use_qk_norm and not self.is_nope_layer
        
        self.attention = Attention(args, use_qk_norm=use_qk_norm, use_rope=use_rope)
        
        if args.moe_args and (layer_id+1) % args.moe_args.interleave_moe_layer_step==0:
            self.feed_forward = MoE(
                dim = args.dim,
                hidden_dim = int(args.ffn_exp * args.dim),
                ffn_dim_multiplier = args.ffn_dim_multiplier,
                multiple_of = args.multiple_of,
                moe_args = args.moe_args                
            )
        else:
            hidden_dim = int(4*args.dim)
            hidden_dim = int(2*hidden_dim/3)
            if args.ffn_dim_multiplier is not None:
                hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
            
            self.feed_forward = FeedForward(
                dim = args.dim,
                hidden_dim = hidden_dim,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        
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
    ) -> None:
        if prefix + "attention.wqkv.layer_norm_weight" in state_dict:
            state_dict[prefix + "attention_norm.weight"] = state_dict.pop(prefix + "attention.wqkv.layer_norm_weight")

        if prefix + "feed_forward.mlp.layer_norm_weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(prefix + "feed_forward.mlp.layer_norm_weight")
        elif prefix + "feed_forward.norm.weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(prefix + "feed_forward.norm.weight")

        for k in (
            "feed_forward.experts.mlp",
            "feed_forward.mlp_shared",
            "attention.wo",
            "attention.wqkv",
        ):
            if prefix + k + "._extra_state" in state_dict:
                state_dict.pop(prefix + k + "._extra_state")
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        global_attn_mask: Optional[torch.Tensor],
        local_attn_mask: Optional[torch.Tensor],
    ):
        if self.is_nope_layer or local_attn_mask is None:
            mask = global_attn_mask
        else:
            mask = local_attn_mask
        
        # pre-norm & residual
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
        
class Transformer(nn.Module):
    def __init(self, args: ModelArgs, **kwargs) -> None:
        super().__init()
        self.args = args
        
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layerss
        
        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim, init_method = lambda x: x)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False, init_method = lambda x: x)
        
        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
            args.use_scaled_rope,
            args.rope_scaling_factor,
            args.rope_high_freq_factor,
        )
        
        vision_args = self.args.vision_args
        if vision_args:
            from .vision.embedding import VisionEmbeddings
            
            self.vision_embeddings = VisionEmbeddings(vision_args)
            self.vision_projection = ColumnParallelLinear(
                vision_args.output_dim,
                args.dim,
                bias=False,
                init_method = lambda x:x,
            )
        
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "rope.freqs" in state_dict:
            state_dict.pop(prefix + "rope.freqs")
        
    @torch.inference_mode()
    def forward(self, model_input: TransformerInput) -> TransformerOutput:
        tokens = model_input.tokens
        start_pos = model_input.tokens_positions
        assert isinstance(start_pos, int),(
            "This implementation does not support different start positions per batch item"
        )
        
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        if image_embedding :=model_input.image_embedding:
            h_image = self.vision_projection(image_embedding.embedding)
            h = h * ~image_embedding.mask + h_image * image_embedding.mask
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        self.freqs_cis = self.freqs_cis[start_pos: seqlen+start_pos]
        
        global_attn_mask, local_attn_mask = None, None
        if seqlen>1:
            global_attn_mask = torch.full((seqlen,seqlen), float("-inf"), device = tokens.device)
            global_attn_mask = torch.triu(global_attn_mask, diagonal=1).type_as(h)
        
            if global_attn_mask.device.type == torch.device("mps").type:
                global_attn_mask = torch.nan_to_num(global_attn_mask, nan=0.0)

            if chunk_size := self.args.attention_chunk_size:
                local_attn_mask = create_chunked_attention_mask(seqlen, chunk_size, tokens.device)

        for layer in self.layers:
            h = layer(h, start_pos, self.freqs_cis, global_attn_mask, local_attn_mask)
        h = self.norm(h)
        output = self.output(h).float()
        return TransformerOutput(logits=output)    
        
        
def create_chunked_attention_mask(seq_len: int, attention_chunk_size: int, device: torch.device) -> torch.Tensor:
    block_pos = torch.abs(
        (torch.arange(seq_len).unsqueeze(0) // attention_chunk_size)
        - (torch.arange(seq_len).unsqueeze(1) // attention_chunk_size)
    )
    token_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    mask = (block_pos == 0) & (token_pos <= 0)
    return mask.to(device)