"""
IR 层 — 完整 Transformer 模型结构定义

CodeSprite 模型 = Embedding + N×TransformerBlock + FinalNorm + LM Head

所有前向传播通过 backend 参数委托给具体后端实现。
零框架导入（不 import torch / numpy）。
"""

from typing import List, Dict, Any, Tuple, Optional
from ir.config import ModelConfig
from ir.layers import (
    Layer, Embedding, Linear, RMSNorm, DropoutLayer,
    TransformerBlock
)


class TransformerModel(Layer):
    """
    CodeSprite — 框架无关的 Decoder-only Transformer

    架构:
      Token Embedding
      → Dropout
      → N × TransformerBlock (RMSNorm + Attention + FFN)
      → Final RMSNorm
      → LM Head (Linear)
    """

    def __init__(self, config: ModelConfig, name: str = "codesprite"):
        super().__init__(name)
        self.config = config

        # Token 嵌入
        self.embedding = Embedding(
            config.vocab_size, config.hidden_size, name="embedding"
        )

        # 初始 Dropout
        self.embed_dropout = DropoutLayer(p=config.dropout, name="embed_dropout")

        # N 层 Transformer Block
        self.blocks = [
            TransformerBlock(config, name=f"block_{i}")
            for i in range(config.num_layers)
        ]

        # 最终归一化
        self.final_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, name="final_norm"
        )

        # LM Head (输出投影到 vocab_size)
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size,
            bias=False, name="lm_head"
        )

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """收集所有层的参数形状"""
        shapes = {}
        # Embedding
        for k, v in self.embedding.param_shapes().items():
            shapes[f"embedding.{k}"] = v
        # Blocks
        for block in self.blocks:
            for k, v in block.param_shapes().items():
                shapes[f"{block.name}.{k}"] = v
        # Final norm
        for k, v in self.final_norm.param_shapes().items():
            shapes[f"final_norm.{k}"] = v
        # LM head
        for k, v in self.lm_head.param_shapes().items():
            shapes[f"lm_head.{k}"] = v
        return shapes

    def forward(self, input_ids, backend, mask=None, past_key_values=None,
                use_cache=False, **kwargs):
        """
        完整前向传播

        Args:
            input_ids: (batch, seq_len) token IDs
            backend: Backend 实例（PyTorch / NumPy / ...）
            mask: 注意力掩码（可选，自动生成因果掩码）
            past_key_values: 可选列表 [(k0,v0), (k1,v1), ...] 每层的 KV-Cache
            use_cache: 是否返回新的 KV-Cache

        Returns:
            logits: (batch, seq_len, vocab_size)
            若 use_cache=True → (logits, new_key_values)
        """
        batch_size, seq_len = backend.shape(input_ids)

        # Token Embedding + Dropout
        x = self.embedding.forward(input_ids, backend)
        x = self.embed_dropout.forward(x, backend)

        # 自动生成因果注意力掩码
        if mask is None:
            if past_key_values is not None and len(past_key_values) > 0:
                # KV-Cache 模式：新 token 不需要因果掩码
                mask = None
            else:
                mask = backend.causal_mask(seq_len)

        # N 层 Transformer Block
        new_key_values = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_kv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None

            if use_cache:
                x, layer_cache = block.forward(x, backend, mask=mask,
                                               kv_cache=layer_kv, use_cache=True)
                new_key_values.append(layer_cache)
            else:
                x = block.forward(x, backend, mask=mask,
                                 kv_cache=layer_kv, use_cache=False)

        # 最终归一化 + LM Head
        x = self.final_norm.forward(x, backend)
        logits = self.lm_head.forward(x, backend)

        if use_cache:
            return logits, new_key_values
        return logits

    def forward_block(self, x, block_idx: int, backend, mask=None, **kwargs):
        """单独运行某一层（用于 KV-Cache 推理）"""
        return self.blocks[block_idx].forward(x, backend, mask=mask)

    def get_param_count(self) -> int:
        """计算总参数数量（基于形状）"""
        total = 0
        for shape in self.param_shapes().values():
            count = 1
            for dim in shape:
                count *= dim
            total += count
        return total

    def get_layer_names(self) -> List[str]:
        """返回所有层名称（用于调试和权重映射）"""
        return list(self.param_shapes().keys())

    def __repr__(self):
        params = self.get_param_count()
        return (
            f"TransformerModel(\n"
            f"  config={self.config},\n"
            f"  params={params:,} ({params/1e6:.1f}M),\n"
            f"  layers={len(self.blocks)}\n"
            f")"
        )
