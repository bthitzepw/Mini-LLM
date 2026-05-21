"""
ops/attention.py — 注意力算子接口
"""

# 这些接口由 backends/ 实现，此处仅作文档说明

# scaled_dot_product_attention(q, k, v, mask, num_kv_heads, scale, dropout_p, training):
#     """
#     缩放点积注意力
#     Q: (batch, num_heads, seq_q, head_dim)
#     K: (batch, num_kv_heads, seq_k, head_dim)
#     V: (batch, num_kv_heads, seq_k, head_dim)
#     返回: (batch, num_heads, seq_q, head_dim)
#     """

# reshape_for_heads(x, batch, seq, num_heads, head_dim):
#     """(batch, seq, hidden) → (batch, num_heads, seq, head_dim)"""

# reshape_from_heads(x, batch, seq, hidden):
#     """(batch, num_heads, seq, head_dim) → (batch, seq, hidden)"""

# causal_mask(seq_len):
#     """生成下三角掩码 (seq_len, seq_len)"""
