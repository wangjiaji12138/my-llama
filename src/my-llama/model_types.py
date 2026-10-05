from dataclasses import dataclass
from typing import List, Optional, Union

import torch

@dataclass
class MaskedEmbedding:
    # 不同的tokens需要对齐到同一序列长度才能作为同一批次数据
    embedding: torch.Tensor
    mask: torch.Tensor
    
@dataclass
class LLMInput:
    tokens: torch.Tensor
    images: Optional[List[torch.Tensor]] = None

@dataclass
class TransformerInput:
    tokens: torch.Tensor
    tokens_position: Union[torch.tensor, int]
    image_embedding: Optional[MaskedEmbedding] = None
  
@dataclass  
class LLMOutput:
    logits: torch.Tensor
    
TransformerOutput = LLMOutput