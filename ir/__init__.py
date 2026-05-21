"""
IR (Intermediate Representation) Layer
----------------------------------------
模型结构定义层 — 零框架依赖。

这里的代码不 import torch / numpy / 任何计算框架。
模型只描述"有什么层、长什么样"，不关心"怎么算"。
计算由 backends/ 目录中的后端完成。
"""

from ir.config import ModelConfig
from ir.layers import (
    Layer, Linear, Embedding, LayerNormLayer, RMSNorm, DropoutLayer, Sequential,
    TransformerBlock, Attention, FeedForward, RoPELayer
)
from ir.transformer import TransformerModel
