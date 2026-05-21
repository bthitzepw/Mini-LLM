"""
IR 层 — 模型配置

纯数据类，指定模型的所有结构参数。
零框架依赖（不 import torch / numpy / 任何计算库）。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """MiniLLM 模型配置 — 框架无关"""

    # === 词汇表 ===
    vocab_size: int = 4268
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    # === 模型尺寸 ===
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 0             # 0 = 与 num_heads 相同（MHA），>0 = GQA
    intermediate_size: int = 2048

    # === 序列 ===
    max_seq_length: int = 512

    # === 正则化 ===
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6

    # === 激活函数 ===
    activation: str = "swiglu"        # "swiglu" | "gelu" | "silu"

    # === 位置编码 ===
    use_rope: bool = True
    rope_theta: float = 10000.0

    # === 权重共享 ===
    tie_weights: bool = True           # 输入嵌入 = 输出投影

    # === 其他 ===
    use_bias: bool = False             # LLaMA 风格：大多数层不用 bias

    def __post_init__(self):
        """参数校验"""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须能被 num_heads ({self.num_heads}) 整除"
            )
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_head_dim(self) -> int:
        """GQA 场景下 KV 头的维度 = 总 KV 维度 / KV 头数"""
        return self.hidden_size // self.num_heads  # hidden_size 不变，KV投影到相同维度

    @classmethod
    def from_yaml(cls, config_dict: dict) -> "ModelConfig":
        """从 YAML 配置字典创建 ModelConfig"""
        m = config_dict.get("model", config_dict)
        return cls(
            vocab_size=m.get("vocab_size", 4268),
            hidden_size=m.get("hidden_size", 512),
            num_layers=m.get("num_layers", 8),
            num_heads=m.get("num_heads", 8),
            num_kv_heads=m.get("num_kv_heads", 0),
            intermediate_size=m.get("intermediate_size", 2048),
            max_seq_length=m.get("max_seq_length", 512),
            dropout=m.get("dropout", 0.1),
            rms_norm_eps=m.get("rms_norm_eps", 1e-6),
            activation=m.get("activation", "swiglu"),
            use_rope=m.get("use_rope", True),
            rope_theta=m.get("rope_theta", 10000.0),
            tie_weights=m.get("tie_weights", True),
            use_bias=m.get("use_bias", False),
        )
