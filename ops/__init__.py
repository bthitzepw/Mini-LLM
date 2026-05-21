"""
ops/ 算子抽象层
---------------
定义核心数学运算的接口规范。
不包含具体实现 — 实现由 backends/ 提供。
"""

# 注意力算子接口文档：
#   scaled_dot_product_attention(q, k, v, mask, scale, dropout_p, training) → output
#   reshape_for_heads(x, batch, seq, heads, dim) → (batch, heads, seq, dim)
#   reshape_from_heads(x, batch, seq, hidden) → (batch, seq, hidden)

# 激活函数接口：
#   silu(x) → output
#   gelu(x) → output
#   softmax(x, dim) → output

# 归一化接口：
#   layer_norm(x, weight, bias, eps) → output
#   rms_norm(x, weight, eps) → output
