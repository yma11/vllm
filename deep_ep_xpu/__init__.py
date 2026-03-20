import torch
import deep_ep_cpp

from .utils import EventOverlap
from .buffer import Buffer

# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config

# TODO: original deepep topk_idx_t is configurable we currently set it as int64 for XPU, need to investigate other dtype when free
topk_idx_t = torch.int64
