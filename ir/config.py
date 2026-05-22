"""
IR 层 — 模型配置

纯数据类，指定模型的所有结构参数。
零框架依赖（不 import torch / numpy / 任何计算库）。

# 注：这个文件以前叫 model_config.py，后来重构 IR 的时候挪过来了
# 原来用 dict 传配置，改成 dataclass 之后方便多了
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class ModelConfig:
    """CodeSprite 模型配置 — 框架无关

    # 默认参数是 38M 参数量的小模型，跑起来不需要太大显存
    # 以前测过 hidden_size=256 num_layers=4 的更小版本，loss 降得很慢，放弃了
    """

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

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（保持 YAML key 命名一致性）"""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads if self.num_kv_heads != self.num_heads else 0,
            "intermediate_size": self.intermediate_size,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout,
            "rms_norm_eps": self.rms_norm_eps,
            "activation": self.activation,
            "use_rope": self.use_rope,
            "rope_theta": self.rope_theta,
            "tie_weights": self.tie_weights,
            "use_bias": self.use_bias,
        }


# ============================================================
# 训练配置
# ============================================================

@dataclass
class TrainingConfig:
    """训练超参 — 控制训练过程的所有参数"""

    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 10
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 200
    max_steps: int = 50000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    use_amp: bool = True
    label_smoothing: float = 0.05
    use_ema: bool = False
    ema_decay: float = 0.999
    early_stopping_patience: int = 5
    save_total_limit: int = 3

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，当前值: {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须 > 0，当前值: {self.learning_rate}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs 必须 > 0，当前值: {self.num_epochs}")

    @classmethod
    def from_yaml(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从 YAML 字典创建"""
        t = config_dict.get("training", config_dict)
        return cls(
            batch_size=t.get("batch_size", 16),
            learning_rate=t.get("learning_rate", 3e-4),
            num_epochs=t.get("num_epochs", 10),
            warmup_steps=t.get("warmup_steps", 500),
            save_steps=t.get("save_steps", 500),
            eval_steps=t.get("eval_steps", 200),
            max_steps=t.get("max_steps", 50000),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
            max_grad_norm=t.get("max_grad_norm", 1.0),
            weight_decay=t.get("weight_decay", 0.01),
            adam_beta1=t.get("adam_beta1", 0.9),
            adam_beta2=t.get("adam_beta2", 0.999),
            adam_epsilon=t.get("adam_epsilon", 1e-8),
            use_amp=t.get("use_amp", True),
            label_smoothing=t.get("label_smoothing", 0.05),
            use_ema=t.get("use_ema", False),
            ema_decay=t.get("ema_decay", 0.999),
            early_stopping_patience=t.get("early_stopping_patience", 5),
            save_total_limit=t.get("save_total_limit", 3),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "use_amp": self.use_amp,
            "label_smoothing": self.label_smoothing,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "save_total_limit": self.save_total_limit,
        }


# ============================================================
# 数据配置
# ============================================================

@dataclass
class DataConfig:
    """数据配置 — 训练/验证/测试文件路径和加载参数"""

    train_file: str = "data/raw/train.txt"
    val_file: str = "data/raw/val.txt"
    test_file: str = "data/raw/test.txt"
    num_workers: int = 2
    max_samples: Optional[int] = None  # 调试用：限制样本数

    def __post_init__(self):
        # 只在文件不存在时警告，不抛异常（允许配置先行）
        for label, path in [
            ("训练", self.train_file),
            ("验证", self.val_file),
            ("测试", self.test_file),
        ]:
            if not Path(path).exists():
                import warnings
                warnings.warn(f"{label}数据文件不存在: {path}")

    @classmethod
    def from_yaml(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        """从 YAML 字典创建"""
        d = config_dict.get("data", config_dict)
        return cls(
            train_file=d.get("train_file", "data/raw/train.txt"),
            val_file=d.get("val_file", "data/raw/val.txt"),
            test_file=d.get("test_file", "data/raw/test.txt"),
            num_workers=d.get("num_workers", 0),
            max_samples=d.get("max_samples", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_file": self.train_file,
            "val_file": self.val_file,
            "test_file": self.test_file,
            "num_workers": self.num_workers,
            "max_samples": self.max_samples,
        }


# ============================================================
# 系统配置
# ============================================================

@dataclass
class SystemConfig:
    """系统配置 — 设备、线程、精度、路径等运行时参数"""

    device: str = "auto"         # "auto" | "cuda" | "cpu"
    cpu_threads: Optional[int] = None
    seed: int = 42
    precision: str = "fp32"      # "fp32" | "fp16" | "bf16"
    log_level: str = "INFO"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def __post_init__(self):
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"device 必须是 'auto'/'cuda'/'cpu'，当前值: {self.device}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"precision 必须是 'fp32'/'fp16'/'bf16'，当前值: {self.precision}")

    @classmethod
    def from_yaml(cls, config_dict: Dict[str, Any]) -> "SystemConfig":
        """从 YAML 字典创建"""
        s = config_dict.get("system", config_dict)
        return cls(
            device=s.get("device", "auto"),
            cpu_threads=s.get("cpu_threads", None),
            seed=s.get("seed", 42),
            precision=s.get("precision", "fp32"),
            log_level=s.get("log_level", "INFO"),
            checkpoint_dir=s.get("checkpoint_dir", "checkpoints"),
            log_dir=s.get("log_dir", "logs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "cpu_threads": self.cpu_threads,
            "seed": self.seed,
            "precision": self.precision,
            "log_level": self.log_level,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
        }


# ============================================================
# 顶层配置聚合
# ============================================================

@dataclass
class Config:
    """CodeSprite 总配置 — 聚合所有子配置，支持 YAML 读写和 CLI 合并"""

    version: str = "2.0.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载完整配置"""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            version=raw.get("version", "2.0.0"),
            model=ModelConfig.from_yaml(raw.get("model", raw)),
            training=TrainingConfig.from_yaml(raw.get("training", raw)),
            data=DataConfig.from_yaml(raw.get("data", raw)),
            system=SystemConfig.from_yaml(raw.get("system", raw)),
        )

    def to_yaml(self, path: str):
        """保存配置到 YAML 文件"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "system": self.system.to_dict(),
        }

    def merge_from_args(self, args):
        """从命令行参数合并配置（覆盖 YAML 默认值）

        模型参数: --hidden_size, --num_layers, --num_heads
        训练参数: --lr, --epochs, --batch_size,
                  --no-amp, --use-ema, --label-smoothing
        模型开关: --no-rope, --no-swiglu
        系统参数: --device
        """
        # ---- 模型参数 ----
        if hasattr(args, "hidden_size") and args.hidden_size is not None:
            self.model.hidden_size = args.hidden_size
        if hasattr(args, "num_layers") and args.num_layers is not None:
            self.model.num_layers = args.num_layers
        if hasattr(args, "num_heads") and args.num_heads is not None:
            self.model.num_heads = args.num_heads

        # ---- 模型开关（布尔 flag）----
        if hasattr(args, "no_rope") and args.no_rope:
            self.model.use_rope = False
        if hasattr(args, "no_swiglu") and args.no_swiglu:
            self.model.activation = "gelu"

        # ---- 训练参数 ----
        if hasattr(args, "lr") and args.lr is not None:
            self.training.learning_rate = args.lr
        if hasattr(args, "epochs") and args.epochs is not None:
            self.training.num_epochs = args.epochs
        if hasattr(args, "batch_size") and args.batch_size is not None:
            self.training.batch_size = args.batch_size

        # ---- 训练开关（布尔 flag）----
        if hasattr(args, "no_amp") and args.no_amp:
            self.training.use_amp = False
        if hasattr(args, "use_ema") and args.use_ema:
            self.training.use_ema = True
        if hasattr(args, "label_smoothing") and args.label_smoothing is not None:
            self.training.label_smoothing = args.label_smoothing

        # ---- 系统参数 ----
        if hasattr(args, "device") and args.device is not None:
            self.system.device = args.device

        # 合并后重新校验
        self.model.__post_init__()
        self.training.__post_init__()
        self.system.__post_init__()
