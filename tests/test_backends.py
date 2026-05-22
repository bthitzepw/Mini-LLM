"""
后端层测试 — 覆盖 PyTorch 和 NumPy 两个计算后端的关键算子
================================================================

测试策略：
  1. PyTorch 后端使用 CPU 设备（不依赖 GPU）
  2. NumPy 后端测试基础算子 + 推理链路
  3. 接口一致性：两个后端同一算子产生相同结构的结果
"""

import pytest
import math
import numpy as np
import torch


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def pt_backend():
    """PyTorch 后端（CPU）"""
    from backends.pytorch import PyTorchBackend
    return PyTorchBackend(device="cpu")


@pytest.fixture
def np_backend():
    """NumPy 后端"""
    from backends.numpy import NumPyBackend
    return NumPyBackend(seed=42)


@pytest.fixture
def sample_tensors():
    """创建测试用的 PyTorch/NumPy 张量对"""
    np_a = np.random.randn(2, 3).astype(np.float32)
    np_b = np.random.randn(2, 3).astype(np.float32)
    pt_a = torch.from_numpy(np_a)
    pt_b = torch.from_numpy(np_b)
    return {"np_a": np_a, "np_b": np_b, "pt_a": pt_a, "pt_b": pt_b}


# ============================================================
# PyTorch 后端 — 张量工具
# ============================================================

class TestPyTorchTensorOps:
    """PyTorch 后端张量基础操作"""

    def test_shape(self, pt_backend, sample_tensors):
        a = sample_tensors["pt_a"]
        assert pt_backend.shape(a) == (2, 3)

    def test_add(self, pt_backend, sample_tensors):
        a, b = sample_tensors["pt_a"], sample_tensors["pt_b"]
        result = pt_backend.add(a, b)
        assert pt_backend.shape(result) == (2, 3)
        torch.testing.assert_close(result, a + b)

    def test_multiply(self, pt_backend, sample_tensors):
        a, b = sample_tensors["pt_a"], sample_tensors["pt_b"]
        result = pt_backend.multiply(a, b)
        assert pt_backend.shape(result) == (2, 3)
        torch.testing.assert_close(result, a * b)

    def test_matmul(self, pt_backend):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        result = pt_backend.matmul(a, b)
        assert pt_backend.shape(result) == (2, 4)

    def test_transpose(self, pt_backend):
        a = torch.randn(2, 3, 4)
        result = pt_backend.transpose(a, 0, 2)
        assert pt_backend.shape(result) == (4, 3, 2)

    def test_reshape(self, pt_backend):
        a = torch.randn(2, 3, 4)
        result = pt_backend.reshape(a, 6, 4)
        assert pt_backend.shape(result) == (6, 4)

    def test_zeros_like(self, pt_backend, sample_tensors):
        a = sample_tensors["pt_a"]
        result = pt_backend.zeros_like(a)
        assert result.shape == a.shape
        assert torch.all(result == 0)

    def test_ones_like(self, pt_backend, sample_tensors):
        a = sample_tensors["pt_a"]
        result = pt_backend.ones_like(a)
        assert result.shape == a.shape
        assert torch.all(result == 1)

    def test_unsqueeze(self, pt_backend):
        a = torch.randn(2, 3)
        result = pt_backend.unsqueeze(a, 1)
        assert pt_backend.shape(result) == (2, 1, 3)

    def test_sqrt(self, pt_backend):
        a = torch.tensor([4.0, 9.0, 16.0])
        result = pt_backend.sqrt(a)
        torch.testing.assert_close(result, torch.sqrt(a))

    def test_log(self, pt_backend):
        a = torch.tensor([1.0, 2.718, 7.389])
        result = pt_backend.log(a)
        torch.testing.assert_close(result, torch.log(a))

    def test_exp(self, pt_backend):
        a = torch.tensor([0.0, 1.0, 2.0])
        result = pt_backend.exp(a)
        torch.testing.assert_close(result, torch.exp(a))

    def test_argmax(self, pt_backend):
        a = torch.tensor([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]])
        result = pt_backend.argmax(a, dim=-1)
        assert torch.equal(result, torch.tensor([1, 0]))

    def test_to_numpy(self, pt_backend):
        a = torch.randn(2, 3)
        result = pt_backend.to_numpy(a)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)


# ============================================================
# PyTorch 后端 — 神经网络算子
# ============================================================

class TestPyTorchNNOps:
    """PyTorch 后端神经网络基础算子"""

    def test_linear(self, pt_backend):
        x = torch.randn(4, 256)
        w = torch.randn(512, 256)
        result = pt_backend.linear(x, w)
        assert pt_backend.shape(result) == (4, 512)

    def test_linear_with_bias(self, pt_backend):
        x = torch.randn(4, 256)
        w = torch.randn(512, 256)
        b = torch.randn(512)
        result = pt_backend.linear(x, w, b)
        assert pt_backend.shape(result) == (4, 512)

    def test_embedding(self, pt_backend):
        input_ids = torch.tensor([[0, 2, 1], [3, 0, 2]])
        weight = torch.randn(100, 128)
        result = pt_backend.embedding(input_ids, weight)
        assert pt_backend.shape(result) == (2, 3, 128)
        # 对比手动查表
        torch.testing.assert_close(result[0, 1], weight[2])

    def test_layer_norm(self, pt_backend):
        x = torch.randn(2, 4, 256)
        w = torch.ones(256)
        b = torch.zeros(256)
        result = pt_backend.layer_norm(x, w, b, eps=1e-5)
        assert pt_backend.shape(result) == (2, 4, 256)

    def test_rms_norm_shape(self, pt_backend):
        x = torch.randn(2, 4, 512)
        w = torch.ones(512)
        result = pt_backend.rms_norm(x, w, eps=1e-5)
        assert pt_backend.shape(result) == (2, 4, 512)

    def test_rms_norm_normalizes(self, pt_backend):
        """RMSNorm 应使 RMS ≈ 1（乘以 weight=1 时）"""
        x = torch.randn(4, 8, 256) * 5.0 + 2.0  # 非零均值
        w = torch.ones(256)
        result = pt_backend.rms_norm(x, w, eps=1e-5)
        rms_after = torch.sqrt(torch.mean(result ** 2, dim=-1))
        assert torch.allclose(rms_after, torch.ones_like(rms_after), atol=1e-4)

    def test_dropout_training(self, pt_backend):
        x = torch.ones(100, 1000)
        result = pt_backend.dropout(x, p=0.5, training=True)
        # 训练模式应有部分值被置零
        assert torch.sum(result == 0.0) > 100
        assert result.shape == x.shape

    def test_dropout_eval(self, pt_backend):
        x = torch.randn(10, 20)
        result = pt_backend.dropout(x, p=0.5, training=False)
        # 推理模式不做 dropout
        torch.testing.assert_close(result, x)


# ============================================================
# PyTorch 后端 — 激活函数
# ============================================================

class TestPyTorchActivations:
    """PyTorch 后端激活函数"""

    def test_silu(self, pt_backend):
        x = torch.randn(2, 3, 64)
        result = pt_backend.silu(x)
        expected = x * torch.sigmoid(x)  # SiLU = x * sigmoid(x)
        torch.testing.assert_close(result, expected)

    def test_gelu(self, pt_backend):
        x = torch.randn(2, 3, 64)
        result = pt_backend.gelu(x)
        import torch.nn.functional as F
        torch.testing.assert_close(result, F.gelu(x))

    def test_softmax(self, pt_backend):
        x = torch.randn(2, 5)
        result = pt_backend.softmax(x, dim=-1)
        # 每行和应为 1
        row_sums = result.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones(2))


# ============================================================
# PyTorch 后端 — 注意力机制
# ============================================================

class TestPyTorchAttention:
    """PyTorch 后端注意力机制"""

    def test_reshape_for_heads(self, pt_backend):
        x = torch.randn(2, 16, 256)  # (batch, seq, hidden)
        result = pt_backend.reshape_for_heads(x, 2, 16, 8, 32)
        assert pt_backend.shape(result) == (2, 8, 16, 32)

    def test_reshape_from_heads(self, pt_backend):
        x = torch.randn(2, 8, 16, 32)  # (batch, heads, seq, head_dim)
        result = pt_backend.reshape_from_heads(x, 2, 16, 256)
        assert pt_backend.shape(result) == (2, 16, 256)

    def test_reshape_roundtrip(self, pt_backend):
        """reshape_for_heads → reshape_from_heads 应恢复原形状"""
        original = torch.randn(2, 16, 256)
        head_view = pt_backend.reshape_for_heads(original, 2, 16, 8, 32)
        back = pt_backend.reshape_from_heads(head_view, 2, 16, 256)
        assert pt_backend.shape(back) == pt_backend.shape(original)

    def test_causal_mask(self, pt_backend):
        mask = pt_backend.causal_mask(4)
        assert mask.shape == (4, 4)
        # causal mask: i<j 时为 False（不可见），i>=j 时为 True（可见）
        # mask 是 bool 类型，直接比对
        for i in range(4):
            for j in range(4):
                expected = (i >= j)
                assert mask[i, j].item() == expected, \
                    f"mask[{i},{j}] 期望 {expected}, 实际 {mask[i,j]}"

    def test_scaled_dot_product_attention_shape(self, pt_backend):
        q = torch.randn(2, 4, 8, 32)  # (batch, heads, seq, dim)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)
        result = pt_backend.scaled_dot_product_attention(q, k, v)
        assert pt_backend.shape(result) == (2, 4, 8, 32)

    def test_sdpa_with_mask(self, pt_backend):
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)
        mask = pt_backend.causal_mask(8)
        result = pt_backend.scaled_dot_product_attention(q, k, v, mask=mask)
        assert pt_backend.shape(result) == (2, 4, 8, 32)
        # 无 NaN/Inf
        assert torch.isfinite(result).all()

    def test_sdpa_gqa(self, pt_backend):
        """GQA: Q 的 heads 数 > K/V 的 heads 数"""
        q = torch.randn(2, 8, 4, 64)  # 8 Q heads
        k = torch.randn(2, 2, 4, 64)  # 2 KV heads
        v = torch.randn(2, 2, 4, 64)
        result = pt_backend.scaled_dot_product_attention(q, k, v, num_kv_heads=2)
        assert pt_backend.shape(result) == (2, 8, 4, 64)


# ============================================================
# PyTorch 后端 — RoPE
# ============================================================

class TestPyTorchRoPE:
    """PyTorch 后端旋转位置编码"""

    def test_rope_precompute_shape(self, pt_backend):
        cos, sin = pt_backend.rope_precompute(seq_len=64, dim=128, theta=10000.0)
        assert cos.shape == (64, 128)
        assert sin.shape == (64, 128)

    def test_rope_apply_shape(self, pt_backend):
        cos, sin = pt_backend.rope_precompute(64, 32, 10000.0)
        q = torch.randn(2, 4, 64, 32)
        k = torch.randn(2, 4, 64, 32)
        q_rope, k_rope = pt_backend.rope_apply(q, k, cos, sin)
        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape

    def test_rope_preserves_norm(self, pt_backend):
        """RoPE 应保持每个 token 的向量范数"""
        dim = 32
        cos, sin = pt_backend.rope_precompute(16, dim, 10000.0)
        q = torch.randn(2, 4, 16, dim)
        k = torch.randn(2, 4, 16, dim)
        q_norm_before = torch.norm(q.float(), dim=-1)
        k_norm_before = torch.norm(k.float(), dim=-1)

        q_rope, k_rope = pt_backend.rope_apply(q, k, cos, sin)
        q_norm_after = torch.norm(q_rope.float(), dim=-1)
        k_norm_after = torch.norm(k_rope.float(), dim=-1)

        torch.testing.assert_close(q_norm_after, q_norm_before, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_norm_after, k_norm_before, atol=1e-4, rtol=1e-4)


# ============================================================
# PyTorch 后端 — 损失函数
# ============================================================

class TestPyTorchLoss:
    """PyTorch 后端损失函数"""

    def test_cross_entropy(self, pt_backend):
        logits = torch.randn(4, 100)  # (batch, vocab)
        targets = torch.randint(0, 100, (4,))
        loss = pt_backend.cross_entropy(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_cross_entropy_label_smoothing(self, pt_backend):
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        loss_no_smooth = pt_backend.cross_entropy(logits, targets, label_smoothing=0.0)
        loss_smooth = pt_backend.cross_entropy(logits, targets, label_smoothing=0.1)
        # 标签平滑会改变损失值（不等于原值）
        assert loss_smooth.item() != pytest.approx(loss_no_smooth.item())
        # 两者都应 > 0
        assert loss_no_smooth.item() > 0
        assert loss_smooth.item() > 0


# ============================================================
# PyTorch 后端 — 权重管理
# ============================================================

class TestPyTorchWeights:
    """PyTorch 后端权重管理"""

    def test_init_weight_xavier(self, pt_backend):
        w = pt_backend.init_weight((64, 128), method="xavier")
        assert pt_backend.shape(w) == (64, 128)
        # Xavier 范围 (fan_in=128, fan_out=64)
        expected_limit = math.sqrt(6.0 / (128 + 64))
        assert torch.all(w >= -expected_limit * 1.1)
        assert torch.all(w <= expected_limit * 1.1)

    def test_init_weight_normal(self, pt_backend):
        w = pt_backend.init_weight((32, 64), method="normal")
        assert w.shape == (32, 64)

    def test_init_weight_zeros(self, pt_backend):
        w = pt_backend.init_weight((16, 32), method="zeros")
        assert torch.all(w == 0)

    def test_init_weight_ones(self, pt_backend):
        w = pt_backend.init_weight((16, 32), method="ones")
        assert torch.all(w == 1)

    def test_create_parameter(self, pt_backend):
        w = pt_backend.create_parameter((32, 64), method="xavier")
        assert w.requires_grad is True
        assert w.shape == (32, 64)

    def test_state_dict_roundtrip(self, pt_backend):
        """get_state_dict → load_state_dict 应无损"""
        from ir.config import ModelConfig
        from ir.transformer import TransformerModel
        from backends.pytorch import init_model_weights

        config = ModelConfig(hidden_size=128, num_layers=2, num_heads=4,
                             num_kv_heads=2, vocab_size=1000, max_seq_length=32)
        model = TransformerModel(config)
        init_model_weights(model, pt_backend)

        state = pt_backend.get_state_dict(model)
        assert len(state) > 0
        assert "embedding.weight" in state

        # 创建新模型加载权重
        model2 = TransformerModel(config)
        init_model_weights(model2, pt_backend)
        pt_backend.load_state_dict(model2, state)

        state2 = pt_backend.get_state_dict(model2)
        for key in state:
            torch.testing.assert_close(state2[key], state[key])


# ============================================================
# NumPy 后端 — 张量工具
# ============================================================

class TestNumPyTensorOps:
    """NumPy 后端张量基础操作"""

    def test_shape(self, np_backend, sample_tensors):
        a = sample_tensors["np_a"]
        assert np_backend.shape(a) == (2, 3)

    def test_add(self, np_backend, sample_tensors):
        a, b = sample_tensors["np_a"], sample_tensors["np_b"]
        result = np_backend.add(a, b)
        np.testing.assert_array_almost_equal(result, a + b)

    def test_multiply(self, np_backend, sample_tensors):
        a, b = sample_tensors["np_a"], sample_tensors["np_b"]
        result = np_backend.multiply(a, b)
        np.testing.assert_array_almost_equal(result, a * b)

    def test_matmul(self, np_backend):
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)
        result = np_backend.matmul(a, b)
        assert result.shape == (2, 4)

    def test_transpose(self, np_backend):
        a = np.random.randn(2, 3, 4).astype(np.float32)
        result = np_backend.transpose(a, 0, 2)
        assert result.shape == (4, 3, 2)

    def test_zeros_like(self, np_backend, sample_tensors):
        result = np_backend.zeros_like(sample_tensors["np_a"])
        assert np.all(result == 0)

    def test_ones_like(self, np_backend, sample_tensors):
        result = np_backend.ones_like(sample_tensors["np_a"])
        assert np.all(result == 1)

    def test_unsqueeze(self, np_backend):
        a = np.random.randn(2, 3).astype(np.float32)
        result = np_backend.unsqueeze(a, 0)
        assert result.shape == (1, 2, 3)

    def test_sqrt(self, np_backend):
        a = np.array([4.0, 9.0, 16.0], dtype=np.float32)
        result = np_backend.sqrt(a)
        np.testing.assert_array_almost_equal(result, np.sqrt(a))

    def test_to_numpy_is_identity(self, np_backend, sample_tensors):
        a = sample_tensors["np_a"]
        result = np_backend.to_numpy(a)
        assert result is a  # NumPy 后端 to_numpy 返回自身


# ============================================================
# NumPy 后端 — 神经网络算子
# ============================================================

class TestNumPyNNOps:
    """NumPy 后端神经网络基础算子"""

    def test_linear(self, np_backend):
        x = np.random.randn(4, 256).astype(np.float32)
        w = np.random.randn(512, 256).astype(np.float32)
        result = np_backend.linear(x, w)
        assert result.shape == (4, 512)

    def test_linear_with_bias(self, np_backend):
        x = np.random.randn(4, 256).astype(np.float32)
        w = np.random.randn(512, 256).astype(np.float32)
        b = np.random.randn(512).astype(np.float32)
        result = np_backend.linear(x, w, b)
        assert result.shape == (4, 512)

    def test_embedding(self, np_backend):
        input_ids = np.array([[0, 2, 1], [3, 0, 2]])
        weight = np.random.randn(100, 128).astype(np.float32)
        result = np_backend.embedding(input_ids, weight)
        assert result.shape == (2, 3, 128)

    def test_rms_norm(self, np_backend):
        x = np.random.randn(2, 4, 512).astype(np.float32)
        w = np.ones(512, dtype=np.float32)
        result = np_backend.rms_norm(x, w, eps=1e-5)
        assert result.shape == (2, 4, 512)

    def test_dropout_is_pass_through(self, np_backend):
        """NumPy 推理模式 dropout 应原样返回"""
        x = np.random.randn(10, 20).astype(np.float32)
        result = np_backend.dropout(x, p=0.5, training=True)
        np.testing.assert_array_equal(result, x)


# ============================================================
# NumPy 后端 — 激活函数
# ============================================================

class TestNumPyActivations:
    """NumPy 后端激活函数"""

    def test_silu(self, np_backend):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = np_backend.silu(x)
        # SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
        expected = x / (1.0 + np.exp(-x))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_gelu(self, np_backend):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = np_backend.gelu(x)
        assert x.shape == result.shape

    def test_softmax_sums_to_one(self, np_backend):
        x = np.random.randn(3, 10).astype(np.float32)
        result = np_backend.softmax(x, dim=-1)
        row_sums = result.sum(axis=-1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3))


# ============================================================
# NumPy 后端 — 注意力机制
# ============================================================

class TestNumPyAttention:
    """NumPy 后端注意力机制"""

    def test_reshape_for_heads(self, np_backend):
        x = np.random.randn(2, 16, 256).astype(np.float32)
        result = np_backend.reshape_for_heads(x, 2, 16, 8, 32)
        assert result.shape == (2, 8, 16, 32)

    def test_reshape_from_heads(self, np_backend):
        x = np.random.randn(2, 8, 16, 32).astype(np.float32)
        result = np_backend.reshape_from_heads(x, 2, 16, 256)
        assert result.shape == (2, 16, 256)

    def test_causal_mask(self, np_backend):
        mask = np_backend.causal_mask(4)
        assert mask.shape == (4, 4)
        for i in range(4):
            for j in range(4):
                assert mask[i, j] == (1.0 if i >= j else 0.0)

    def test_sdpa_no_mask(self, np_backend):
        q = np.random.randn(2, 4, 8, 32).astype(np.float32)
        k = np.random.randn(2, 4, 8, 32).astype(np.float32)
        v = np.random.randn(2, 4, 8, 32).astype(np.float32)
        result = np_backend.scaled_dot_product_attention(q, k, v)
        assert result.shape == (2, 4, 8, 32)

    def test_sdpa_gqa(self, np_backend):
        q = np.random.randn(2, 8, 4, 64).astype(np.float32)
        k = np.random.randn(2, 2, 4, 64).astype(np.float32)
        v = np.random.randn(2, 2, 4, 64).astype(np.float32)
        result = np_backend.scaled_dot_product_attention(q, k, v, num_kv_heads=2)
        assert result.shape == (2, 8, 4, 64)


# ============================================================
# NumPy 后端 — RoPE
# ============================================================

class TestNumPyRoPE:
    """NumPy 后端旋转位置编码"""

    def test_rope_precompute(self, np_backend):
        cos, sin = np_backend.rope_precompute(seq_len=32, dim=64, theta=10000.0)
        assert cos.shape == (32, 64)
        assert sin.shape == (32, 64)

    def test_rope_apply_preserves_shape(self, np_backend):
        cos, sin = np_backend.rope_precompute(32, 64, 10000.0)
        q = np.random.randn(2, 4, 32, 64).astype(np.float32)
        k = np.random.randn(2, 4, 32, 64).astype(np.float32)
        q_rope, k_rope = np_backend.rope_apply(q, k, cos, sin)
        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape


# ============================================================
# 接口一致性 — 两个后端同一算子结果对比
# ============================================================

class TestInterfaceConsistency:
    """验证两个后端接口一致性：同一输入应产生相同形状和类型的结果"""

    def test_shape_consistency(self, pt_backend, np_backend, sample_tensors):
        assert pt_backend.shape(sample_tensors["pt_a"]) == \
               np_backend.shape(sample_tensors["np_a"])

    def test_linear_consistency(self, pt_backend, np_backend):
        """相同输入→相同形状输出"""
        x_np = np.random.randn(4, 64).astype(np.float32)
        w_np = np.random.randn(128, 64).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        w_pt = torch.from_numpy(w_np)

        r_np = np_backend.linear(x_np, w_np)
        r_pt = pt_backend.linear(x_pt, w_pt)

        assert r_np.shape == tuple(r_pt.shape)

    def test_rms_norm_consistency(self, pt_backend, np_backend):
        x_np = np.random.randn(2, 4, 64).astype(np.float32)
        w_np = np.ones(64, dtype=np.float32)
        x_pt = torch.from_numpy(x_np)
        w_pt = torch.ones(64)

        r_np = np_backend.rms_norm(x_np, w_np, eps=1e-5)
        r_pt = pt_backend.rms_norm(x_pt, w_pt, eps=1e-5)

        assert r_np.shape == tuple(r_pt.shape)

    def test_softmax_consistency(self, pt_backend, np_backend):
        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x_pt = torch.tensor([[1.0, 2.0, 3.0]])

        r_np = np_backend.softmax(x_np, dim=-1)
        r_pt = pt_backend.softmax(x_pt, dim=-1)

        np.testing.assert_array_almost_equal(r_np, pt_backend.to_numpy(r_pt))

    def test_backend_names(self, pt_backend, np_backend):
        assert pt_backend.name == "pytorch"
        assert np_backend.name == "numpy"

    def test_invalid_init_method_raises(self, pt_backend):
        with pytest.raises(ValueError, match="未知初始化方法"):
            pt_backend.init_weight((4, 4), method="unknown")


# ============================================================
# IR 模型 + 后端集成（冒烟测试）
# ============================================================

class TestModelBackendIntegration:
    """IR 模型通过后端执行的端到端冒烟测试"""

    def test_forward_pytorch_no_error(self, pt_backend):
        """PyTorch 后端前向传播不报错"""
        from ir.config import ModelConfig
        from ir.transformer import TransformerModel
        from backends.pytorch import init_model_weights

        config = ModelConfig(hidden_size=128, num_layers=2, num_heads=4,
                             num_kv_heads=2, vocab_size=1000, max_seq_length=32)
        model = TransformerModel(config)
        init_model_weights(model, pt_backend)

        input_ids = torch.randint(0, 1000, (2, 16))
        logits = model.forward(input_ids, pt_backend)
        assert logits.shape == (2, 16, 1000)
        assert torch.isfinite(logits).all()

    def test_forward_numpy_no_error(self, np_backend):
        """NumPy 后端前向传播不报错"""
        from ir.config import ModelConfig
        from ir.transformer import TransformerModel

        config = ModelConfig(hidden_size=64, num_layers=2, num_heads=4,
                             num_kv_heads=2, vocab_size=500, max_seq_length=16)
        model = TransformerModel(config)

        # 使用 param_shapes() 构建权重字典，再通过 load_state_dict 加载
        shapes = model.param_shapes()
        state_dict = {}
        for key, shape in shapes.items():
            state_dict[key] = np_backend.init_weight(shape, "normal")
        np_backend.load_state_dict(model, state_dict)

        input_ids = np.random.randint(0, 500, (1, 8)).astype(np.int64)
        logits = model.forward(input_ids, np_backend)
        assert logits.shape == (1, 8, 500)
        assert np.isfinite(logits).all()

    def test_pytorch_training_step(self, pt_backend):
        """PyTorch 后端完整训练步（前向→loss→反向→优化器步）"""
        from ir.config import ModelConfig
        from ir.transformer import TransformerModel
        from backends.pytorch import init_model_weights

        config = ModelConfig(hidden_size=64, num_layers=2, num_heads=4,
                             num_kv_heads=2, vocab_size=500, max_seq_length=16)
        model = TransformerModel(config)
        init_model_weights(model, pt_backend)

        input_ids = torch.randint(0, 500, (2, 12))
        logits = model.forward(input_ids, pt_backend)

        # 下一个 token 预测：targets 偏移一位
        loss = pt_backend.cross_entropy(
            logits[:, :-1].reshape(-1, 500),
            input_ids[:, 1:].reshape(-1)
        )

        # 创建优化器并执行一步
        from backends.pytorch import collect_parameters
        params = collect_parameters(model)
        opt = pt_backend.create_optimizer(params, lr=1e-4)

        opt.zero_grad()
        loss.backward()
        pt_backend.clip_grad_norm([p for _, p in params], 1.0)
        opt.step()

        assert loss.item() > 0
