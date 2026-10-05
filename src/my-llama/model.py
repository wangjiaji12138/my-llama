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
