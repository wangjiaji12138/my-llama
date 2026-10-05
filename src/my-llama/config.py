#参数权重

#LLM 推理优化里，量化就是降低数值精度，减少显存占用和内存带宽
from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator

class QuantizationScheme(Enum):
    """
    4bit整数存权重, 8bit整数存推理时动态计算的激活值
    z = Wx+b or Wx
    a = ReLU(z)
    其中W训练好后FP16->int4,b保持FP16(数量少没必要量化,而且很多大模型不选择加入bias)
    x是int8,会和W使用整数乘法相乘后到z=int32
    z是int32->FP16
    a是FP16->int8,传给下一层当x
    """
    int4_weight_int_8_dynamic_activation = "int4_weight_int_8_dynamic_activation"
    
"""
量化quantization: 用精度换开销,神经网络不需要太细微的精度, 而需要的是存储大量的权重
scale:模型精度下界
zero_point(0对应哪个整数,也就是偏差值)
例: 16bit浮点数x----(x/scale) + zero_point = y------4bit整数y
    y----(y-zero_point) * scale = x ------4bit整数y
    
旋转spinquant:通过学习一个正交的旋转矩阵R使得缩小原始量化group内的范围,保留更多精度
"""
class QuantizationArgs(BaseModel):
    scheme: Optional[QuantizationScheme] = None
    group_size: Optional[int] = None
    spinquant: bool = False
    
"""
https://zhuanlan.zhihu.com/p/1939468581650793792
LoRA核心思想:在得到预训练矩阵W后做微调时,冻结W,而是通过低秩分解学习增量W'= W + alpha/r * BA
原始的W = (d_out,d_in),低秩矩阵A = (r,d_din), 升维矩阵B = (d_out,r)
这里的r(rank)就是选择的秩, r << min(d_in,d_out)
alpha是缩放因子,也就是代码里的scale
"""

class LoRAArgs(BaseModel):
    rank: int
    scale: float

"""Llama4引进MoE架构"""
class MoEArgs(BaseModel):
    num_experts: int = -1
    capacity_factor: float = 1.0 # 相对平均分配的最大负载
    auto_scale_F: bool = True # 使每个MoE激活的参数量与dense层一样
    top_k: int = 1 # 选择的专家数
    interleave_moe_layer_step: int = 1  # 控制MOE层的间隔  
    
class Size(BaseModel):
    height: int
    width: int
    
class VisionArgs(BaseModel):
    image_size: Size 
    patch_size: Size # ViT 把image 切成的 patch 的尺寸
    
    # params for decoder
    dim: int
    n_layers: int
    n_heads: int
    mlp_ratio: float # 隐藏层扩展比率:如d->4d->4d->d
    output_dim: int
    
    pixel_shuffle_ratio: float # 像素重排比率：不增加计算量，只通过通道和空间维度的重排，改变特征图的分辨率
    
class ModelArgs(BaseModel):
    # 参数为-1表示自动推断
    dim: int = -1
    n_layers: int = -1
    n_heads: int = -1
    n_kv_heads: Optional[int] = None # 单独控制KV的头数，一般小于Q的n_heads
    head_dim: Optional[int] = None
    
    vocab_size: int = -1
    multiple_of: int = 256 # 隐藏层必须要向上取整到256的倍数，便于计算
    ffn_dim_multiplier: Optional[float] = None # 对FFN的hidden_dim再进行的缩放系数，因为hidden_dim一般为了SwiGLU调成8/3来等效一般的ffn，但精度不太够
    ffn_exp: Optional[float] = None
    norm_eps: float = 1e-5 # y = x / sqrt(mean(x^2) + eps) * gamma
    
    attention_chunk_size: Optional[int] = None # 长序列token 分块处理:4096->(512,512,...,512)
    rope_theta: float = 500000
    use_scaled_rope: bool = False # 将长序列的位置先映射到训练时见过的范围，再进行编码
    rope_scaling_factor: Optional[float] = None # 训练时的位置编码范围*rope_scaling_factor = 实际可处理的位置范围
    rope_high_freq_factor: Optional[float] = None # YaRN扩展，用于控制长序列位置压缩的高低频边界
    
    nope_layer_interval: Optional[int] = None # NoPE:不使用位置编码
    use_qk_norm: bool = False
    attn_temperature_tuning: bool = False
    floor_scale: float = 8192.0 # 温度调整的序列长度下界，小于下界时不调整，大于时进行调整
    attn_scale: float = 0.1 # QK/sqrt(d)后再次*scale
    
    vision_args: Optional[VisionArgs] = None
    moe_args: Optional[MoEArgs] = None
    quantization_args: Optional[QuantizationArgs] = None
    lora_args: Optional[LoRAArgs] = None
    
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    @model_validator(mode="after")
    def validate(self) -> "ModelArgs":
        assert self.n_kv_heads <= self.n_heads, f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"

        if self.use_scaled_rope:
            if self.rope_scaling_factor is None:
                self.rope_scaling_factor = 16 # 默认可以处理16倍的上下文,8k->128k
            if self.rope_high_freq_factor is None:
                self.rope_high_freq_factor = 1 # 几乎所有频率维度都被归为“高频”,默认不处理，靠iRoPE保留细节 
        
        return self