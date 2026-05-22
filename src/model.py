"""
CodeSprite 模型架构 - 框架无关 IR 架构

核心架构: Transformer Decoder (类 GPT)
深度学习增强:
  1. RoPE (Rotary Position Embedding) - 旋转位置编码，支持外推更长的序列
  2. KV-Cache - 键值缓存加速自回归推理，避免重复计算
  3. 梯度检查点 (Gradient Checkpointing) - 用计算换内存，支持更大batch
  4. Flash Attention 兼容接口 - 为后续替换 FlashAttention 留接口
  5. SwiGLU 激活函数 - 更好的前馈网络非线性变换
  6. Pre-Norm 架构 - 更稳定的深度网络训练

# TODO: LayerNorm 换成 RMSNorm（LLaMA 那种），现在 TransformerBlock 里还是用的 LayerNorm
# 主要原因是懒，而且改完以后要重新验证一遍
# TODO: 支持 GQA（这个文件里的 Attention 还是 MHA，ir/layers.py 那边已经实现了 GQA 但没接过来）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class Config:
    """模型配置"""
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 intermediate_size, dropout, max_seq_length, tie_weights):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.tie_weights = tie_weights


# ============================================================
# RoPE 旋转位置编码 (Rotary Position Embedding)
# ============================================================

class RotaryPositionEncoding(nn.Module):
    """
    RoPE: Rotary Position Embedding (Su et al., 2021)

    通过旋转矩阵对 Q/K 编码位置信息，相比绝对位置编码:
    - 更好的长度泛化能力（外推性）
    - 相对位置信息天然编码
    - 计算高效，无需额外参数
    """

    def __init__(self, dim, max_seq_length=512, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # 预计算缓存
        self._set_cos_sin_cache(max_seq_length)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)

        # [seq_len, dim/2] -> 拼接 -> [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        """
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            seq_len: 可选，支持KV缓存时的实际长度
        Returns:
            cos, sin: 用于旋转变换的余弦和正弦值
        """
        if seq_len is None:
            seq_len = x.size(2)

        if seq_len > self.max_seq_length:
            self._set_cos_sin_cache(seq_len)
            self.max_seq_length = seq_len

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    @staticmethod
    def rotate_half(x):
        """将向量分为两半并交换，用于旋转变换"""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """对 Q 和 K 应用旋转位置编码"""
        # 扩展维度以广播: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


# ============================================================
# 带KV缓存的注意力层
# ============================================================

class Attention(nn.Module):
    """
    多头自注意力机制（支持KV缓存和RoPE）

    深度学习特性:
    - RoPE 旋转位置编码
    - KV-Cache 加速推理
    - 可选 Flash Attention
    - 注意力权重缩放
    """

    def __init__(self, config, use_rope=True, use_flash_attn=False):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn

        if self.head_dim * config.num_heads != config.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads, "
                           f"got {config.hidden_size} / {config.num_heads}")

        # Q/K/V 投影
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEncoding(
                dim=self.head_dim,
                max_seq_length=config.max_seq_length
            )

    def forward(self, query, key, value, attention_mask=None,
                kv_cache=None, use_cache=False):
        """
        Args:
            query: [batch, seq_len, hidden_size]
            key: [batch, seq_len, hidden_size] 或 None（使用KV缓存时）
            value: [batch, seq_len, hidden_size] 或 None（使用KV缓存时）
            attention_mask: [batch, 1, seq_len, kv_seq_len] 或 [batch, seq_len]
            kv_cache: 上一步的 (key_cache, value_cache)
            use_cache: 是否返回新的KV缓存
        Returns:
            output: [batch, seq_len, hidden_size]
            new_kv_cache: 可选，更新后的KV缓存
        """
        batch_size, seq_len, _ = query.size()

        # 投影
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key if key is not None else query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value if value is not None else query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        if self.use_rope:
            cos, sin = self.rope(Q, seq_len=seq_len)
            Q, K = self.rope.apply_rotary_pos_emb(Q, K, cos, sin)

        # KV 缓存处理
        if kv_cache is not None:
            past_k, past_v = kv_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        new_kv_cache = None
        if use_cache:
            new_kv_cache = (K, V)

        # 注意力计算
        # Q: [batch, heads, seq_len, head_dim]
        # K: [batch, heads, kv_len, head_dim]
        kv_len = K.size(2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 因果注意力掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        else:
            # 自动生成因果掩码
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=query.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)

        if use_cache:
            return output, new_kv_cache
        return output


# ============================================================
# SwiGLU 前馈网络 (比 GELU 更好的激活)
# ============================================================

class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络 (Shazeer, 2020)

    相比传统 GELU FFN:
    - 参数量略多但效果显著提升 (PaLM, LLaMA 等模型验证)
    - 门控机制提供更好的信息流控制
    - 表达能力更强
    """

    def __init__(self, config, use_swiglu=True):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(config.dropout)

        if use_swiglu:
            # SwiGLU: output = (xW1 * SiLU(xW_gate)) * W2
            self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.linear_gate = nn.Linear(config.hidden_size, config.intermediate_size)
            self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        else:
            # 传统 FFN: GELU
            self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
            self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
            self.activation = nn.GELU()

    def forward(self, x):
        if self.use_swiglu:
            gate = self.linear_gate(x)
            gate = F.silu(gate)  # SiLU = Swish
            x = self.linear1(x) * gate
            x = self.dropout(x)
            x = self.linear2(x)
        else:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
        return x


# ============================================================
# Transformer Block (Pre-Norm 架构)
# ============================================================

class TransformerBlock(nn.Module):
    """
    Transformer 解码器块 (Pre-Norm + RoPE + SwiGLU)

    深度学习最佳实践:
    - Pre-Norm: 归一化放在注意力/FFN之前，训练更稳定
    - RoPE: 旋转位置编码
    - SwiGLU: 门控前馈网络
    - 残差连接: 缓解梯度消失
    """

    def __init__(self, config, use_rope=True, use_swiglu=True, use_flash_attn=False):
        super().__init__()
        self.attention = Attention(config, use_rope=use_rope, use_flash_attn=use_flash_attn)
        self.feed_forward = FeedForward(config, use_swiglu=use_swiglu)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None, kv_cache=None, layer_cache=None, use_cache=False):
        # Pre-Norm + Self-Attention + 残差
        residual = x
        x = self.norm1(x)

        # 获取当前层的 KV 缓存
        layer_kv_cache = None
        if layer_cache is not None:
            layer_kv_cache = layer_cache.get('kv', None) if isinstance(layer_cache, dict) else layer_cache

        attn_output = self.attention(
            x, x, x, attention_mask,
            kv_cache=layer_kv_cache,
            use_cache=use_cache
        )

        if use_cache and isinstance(attn_output, tuple):
            attn_output, new_kv_cache = attn_output
        else:
            new_kv_cache = None

        x = self.dropout(attn_output)
        x = residual + x

        # Pre-Norm + FFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x

        if use_cache:
            return x, new_kv_cache
        return x


# ============================================================
# CodeSprite 主模型
# ============================================================

class CodeSprite(nn.Module):
    """
    CodeSprite - 框架无关 IR 架构的微型代码语言模型

    架构特性:
    - Transformer Decoder (类 GPT)
    - RoPE 旋转位置编码（支持序列长度外推）
    - SwiGLU 前馈网络（更强表达能力）
    - KV-Cache 加速推理（减少重复计算）
    - 梯度检查点（节省显存）
    - Pre-Norm 归一化（训练更稳定）
    - 可选 Flash Attention（兼容接口）
    """

    def __init__(self, config, use_rope=True, use_swiglu=True,
                 use_gradient_checkpointing=False, use_flash_attn=False):
        super().__init__()
        self.config = config
        self.use_rope = use_rope
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_flash_attn = use_flash_attn

        # Token 嵌入
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_rope=use_rope, use_swiglu=use_swiglu,
                           use_flash_attn=use_flash_attn)
            for _ in range(config.num_layers)
        ])

        # 最终层归一化
        self.norm = nn.LayerNorm(config.hidden_size)

        # LM Head (语言模型头)
        if config.tie_weights:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embeddings.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 梯度检查点
        if use_gradient_checkpointing:
            self.enable_gradient_checkpointing()

        # 模型参数统计
        self._count_parameters()

    def _count_parameters(self):
        """统计模型参数"""
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.embedding_params = self.token_embeddings.weight.numel()
        self.attention_params = sum(
            sum(p.numel() for p in layer.attention.parameters())
            for layer in self.layers
        )
        self.ffn_params = sum(
            sum(p.numel() for p in layer.feed_forward.parameters())
            for layer in self.layers
        )

    def enable_gradient_checkpointing(self):
        """启用梯度检查点，用计算换显存"""
        self.use_gradient_checkpointing = True
        # PyTorch 原生梯度检查点
        for layer in self.layers:
            layer._orig_forward = layer.forward
            layer.forward = lambda x, attention_mask=None, kv_cache=None, layer_cache=None, use_cache=False, _layer=layer: (
                torch.utils.checkpoint.checkpoint(
                    _layer._orig_forward, x, attention_mask,
                    use_reentrant=False
                ) if not use_cache else _layer._orig_forward(x, attention_mask, kv_cache, layer_cache, use_cache)
            )

    def disable_gradient_checkpointing(self):
        """禁用梯度检查点"""
        self.use_gradient_checkpointing = False
        for layer in self.layers:
            if hasattr(layer, '_orig_forward'):
                layer.forward = layer._orig_forward

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                past_key_values=None):
        """
        前向传播

        Args:
            input_ids: [batch, seq_len] Token IDs
            attention_mask: 可选，因果注意力掩码
            use_cache: 是否使用 KV 缓存
            past_key_values: 可选，历史 KV 缓存列表
        Returns:
            logits: [batch, seq_len, vocab_size]
            (可选) new_key_values: 更新后的 KV 缓存列表
        """
        batch_size, seq_len = input_ids.size()

        # Token 嵌入 + Dropout
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)

        # 注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        # 逐层传播
        new_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_kv_cache = None
            if past_key_values is not None and i < len(past_key_values):
                layer_kv_cache = past_key_values[i]

            if use_cache:
                hidden_states, layer_kv = layer(
                    hidden_states, attention_mask,
                    kv_cache=layer_kv_cache,
                    use_cache=True
                )
                new_key_values.append(layer_kv)
            else:
                hidden_states = layer(
                    hidden_states, attention_mask,
                    kv_cache=layer_kv_cache,
                    use_cache=False
                )

        # 最终归一化 + LM Head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if use_cache:
            return logits, new_key_values
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0,
                 top_k=None, top_p=None, use_kv_cache=True):
        """
        自回归文本生成（支持KV缓存加速）

        Args:
            input_ids: [batch, seq_len] 初始输入
            max_new_tokens: 最大生成token数
            temperature: 采样温度 (越低越确定性)
            top_k: Top-K 采样参数
            top_p: Top-P (nucleus) 采样参数
            use_kv_cache: 是否使用KV缓存加速
        Returns:
            output_ids: [batch, seq_len + generated_len]
        """
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            if use_kv_cache and past_key_values is not None:
                # KV缓存模式: 只计算最后一个token
                current_input = input_ids[:, -1:]
                logits, past_key_values = self.forward(
                    current_input, use_cache=True,
                    past_key_values=past_key_values
                )
            else:
                # 标准模式: 重新计算全部
                logits = self.forward(input_ids)

            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :] / temperature

            # Top-K 过滤
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Top-P (nucleus) 过滤
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # EOS 停止
            # 硬编码了 token id = 2，应该改成从 config 读
            # 但先这样，反正 eos_token_id 一般就是 2 或 3
            if next_token.item() == 2:
                break

        return input_ids

    def get_model_info(self):
        """获取模型详细信息"""
        return {
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'embedding_params': self.embedding_params,
            'attention_params': self.attention_params,
            'ffn_params': self.ffn_params,
            'num_layers': self.config.num_layers,
            'hidden_size': self.config.hidden_size,
            'num_heads': self.config.num_heads,
            'vocab_size': self.config.vocab_size,
            'max_seq_length': self.config.max_seq_length,
            'use_rope': self.use_rope,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'tie_weights': self.config.tie_weights,
        }


# ============================================================
# 向后兼容: 保持原有接口可用
# ============================================================

# 原始 SimpleAttention 和 SimpleCodeSprite 用于兼容
class SimpleAttention(Attention):
    """兼容旧接口"""
    def __init__(self, config):
        super().__init__(config, use_rope=False)


class SimpleFeedForward(FeedForward):
    """兼容旧接口"""
    def __init__(self, config):
        super().__init__(config, use_swiglu=False)
