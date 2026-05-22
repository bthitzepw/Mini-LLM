"""
IR 层单元测试

覆盖 ir/config.py、ir/layers.py、ir/transformer.py 的核心逻辑。

运行方式：
    cd D:/mydemo/xin
    python -m pytest tests/test_ir.py -v
    或 python tests/test_ir.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# ModelConfig 测试
# ============================================================

def test_modelconfig_defaults():
    """默认配置创建"""
    from ir.config import ModelConfig

    cfg = ModelConfig()
    assert cfg.vocab_size == 4268
    assert cfg.hidden_size == 512
    assert cfg.num_layers == 8
    assert cfg.num_heads == 8
    assert cfg.num_kv_heads == 8       # 默认 0 → 自动设为 num_heads
    assert cfg.intermediate_size == 2048
    assert cfg.max_seq_length == 512
    assert cfg.dropout == 0.1
    assert cfg.activation == "swiglu"
    assert cfg.use_rope is True
    assert cfg.tie_weights is True
    assert cfg.use_bias is False


def test_modelconfig_head_dim():
    """head_dim 属性"""
    from ir.config import ModelConfig

    cfg = ModelConfig(hidden_size=512, num_heads=8)
    assert cfg.head_dim == 64


def test_modelconfig_gqa():
    """GQA 配置：num_kv_heads 少于 num_heads"""
    from ir.config import ModelConfig

    cfg = ModelConfig(hidden_size=512, num_heads=8, num_kv_heads=2)
    assert cfg.num_kv_heads == 2
    assert cfg.num_heads % cfg.num_kv_heads == 0


def test_modelconfig_validation_hidden_not_divisible():
    """hidden_size 不能被 num_heads 整除时抛错"""
    from ir.config import ModelConfig
    import pytest

    with pytest.raises(ValueError, match="必须能被"):
        ModelConfig(hidden_size=513, num_heads=8)


def test_modelconfig_validation_gqa_not_divisible():
    """num_heads 不能被 num_kv_heads 整除时抛错"""
    from ir.config import ModelConfig
    import pytest

    with pytest.raises(ValueError, match="必须能被"):
        ModelConfig(hidden_size=512, num_heads=8, num_kv_heads=3)


def test_modelconfig_custom():
    """自定义参数创建"""
    from ir.config import ModelConfig

    cfg = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=1024,
        max_seq_length=256,
        dropout=0.05,
        activation="gelu",
        use_rope=False,
        tie_weights=False,
        use_bias=True,
    )
    assert cfg.vocab_size == 1000
    assert cfg.hidden_size == 256
    assert cfg.activation == "gelu"
    assert cfg.use_rope is False
    assert cfg.use_bias is True


def test_modelconfig_from_yaml():
    """从 YAML 字典创建 ModelConfig"""
    from ir.config import ModelConfig

    data = {
        "model": {
            "vocab_size": 5000,
            "hidden_size": 384,
            "num_layers": 6,
            "num_heads": 6,
            "intermediate_size": 1536,
            "max_seq_length": 256,
            "dropout": 0.05,
            "activation": "gelu",
        }
    }
    cfg = ModelConfig.from_yaml(data)
    assert cfg.vocab_size == 5000
    assert cfg.hidden_size == 384
    assert cfg.num_layers == 6
    assert cfg.num_heads == 6
    assert cfg.activation == "gelu"


def test_modelconfig_from_yaml_partial():
    """不完整 YAML：缺失字段回退默认值"""
    from ir.config import ModelConfig

    data = {"model": {"vocab_size": 3000}}
    cfg = ModelConfig.from_yaml(data)
    assert cfg.vocab_size == 3000
    assert cfg.hidden_size == 512      # 默认值
    assert cfg.activation == "swiglu"  # 默认值


# ============================================================
# Layer 抽象层测试
# ============================================================

def test_layer_base():
    """Layer 基类基本行为"""
    from ir.layers import Layer

    layer = Layer(name="test")
    assert layer.name == "test"
    assert layer.training is True

    layer.eval()
    assert layer.training is False

    layer.train()
    assert layer.training is True


def test_embedding_param_shapes():
    """Embedding 层参数形状"""
    from ir.layers import Embedding

    emb = Embedding(vocab_size=1000, hidden_size=256, name="emb")
    shapes = emb.param_shapes()

    assert "weight" in shapes
    assert shapes["weight"] == (1000, 256)


def test_linear_param_shapes():
    """Linear 层参数形状"""
    from ir.layers import Linear

    linear = Linear(in_features=256, out_features=512, name="fc")
    shapes = linear.param_shapes()

    assert "weight" in shapes
    assert shapes["weight"] == (512, 256)

    # 默认 bias=False，不应有 bias
    assert "bias" not in shapes

    # 显式要求 bias
    linear_bias = Linear(in_features=256, out_features=512, bias=True, name="fc_bias")
    shapes_bias = linear_bias.param_shapes()
    assert "bias" in shapes_bias
    assert shapes_bias["bias"] == (512,)


def test_rmsnorm_param_shapes():
    """RMSNorm 层参数形状"""
    from ir.layers import RMSNorm

    norm = RMSNorm(normalized_shape=512, name="norm")
    shapes = norm.param_shapes()

    assert "weight" in shapes
    assert shapes["weight"] == (512,)


def test_dropout_no_params():
    """Dropout 层没有可训练参数"""
    from ir.layers import DropoutLayer

    dropout = DropoutLayer(p=0.1, name="drop")
    shapes = dropout.param_shapes()
    assert shapes == {}


def test_sequential():
    """Sequential 组合层"""
    from ir.layers import Sequential, Linear, RMSNorm

    seq = Sequential([
        Linear(256, 512, name="fc1"),
        RMSNorm(512, name="norm"),
        Linear(512, 256, name="fc2"),
    ], name="block")
    shapes = seq.param_shapes()

    # 应有 3 个参数：fc1.weight, norm.weight, fc2.weight
    assert len(shapes) == 3
    assert shapes["fc1.weight"] == (512, 256)
    assert shapes["norm.weight"] == (512,)
    assert shapes["fc2.weight"] == (256, 512)


# ============================================================
# TransformerModel 测试
# ============================================================

def test_transformer_model_build():
    """完整 Transformer 模型构建"""
    from ir.config import ModelConfig
    from ir.transformer import TransformerModel

    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=1024,
        max_seq_length=128,
    )
    model = TransformerModel(config, name="test_model")

    assert model.config.vocab_size == 1000
    assert model.name == "test_model"
    assert len(model.blocks) == 4       # num_layers


def test_transformer_model_param_shapes():
    """Transformer 模型参数完整性"""
    from ir.config import ModelConfig
    from ir.transformer import TransformerModel

    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
        max_seq_length=128,
        tie_weights=False,  # 不复用嵌入权重，确保 lm_head.weight 独立出现
    )
    model = TransformerModel(config)
    shapes = model.param_shapes()

    # 基本检查
    assert "embedding.weight" in shapes
    assert shapes["embedding.weight"] == (1000, 256)
    assert "lm_head.weight" in shapes

    # 每个 block 应有 attention 和 ffn 的参数
    # 实际命名：block_0.attn.q_proj.weight, block_0.ffn.w1.weight, block_0.ffn.wg.weight 等
    for i in range(2):
        prefix = f"block_{i}"
        assert f"{prefix}.attn.q_proj.weight" in shapes
        assert f"{prefix}.attn.o_proj.weight" in shapes
        assert f"{prefix}.ffn.w1.weight" in shapes      # FeedForward 用 w1/w2/wg 命名
        assert f"{prefix}.ffn.w2.weight" in shapes


def test_transformer_model_param_count():
    """模型参数量计算"""
    from ir.config import ModelConfig
    from ir.transformer import TransformerModel

    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        intermediate_size=1024,
        max_seq_length=128,
    )
    model = TransformerModel(config)
    count = model.get_param_count()

    # 参数量应 > 0 且合理（两层小模型大约几百万参数）
    assert count > 0
    assert isinstance(count, int)


def test_transformer_model_eval_mode():
    """模型 eval/train 模式切换"""
    from ir.config import ModelConfig
    from ir.transformer import TransformerModel

    config = ModelConfig()
    model = TransformerModel(config)

    model.eval()
    assert model.training is False

    model.train()
    assert model.training is True


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
