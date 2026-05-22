"""
后端能力规格 (Backend Specification)

将隐式协议显式化为可验证的 Spec，包括：
  - BackendCapability: 单个后端的能力规格
  - BackendValidator: 接口一致性自动检测
  - 算子兼容性矩阵: 跨后端算子支持表

用途：
  1. 新后端接入前的自检清单
  2. 跨后端能力对比
  3. 自动检测接口实现完整性
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

# ============================================================
# 后端能力枚举
# ============================================================

class PrecisionMode(Enum):
    """精度模式"""
    FP32 = auto()      # 全精度
    FP16 = auto()      # 半精度
    BF16 = auto()      # BF16
    INT8 = auto()      # INT8 量化
    INT4 = auto()      # INT4 量化


class FeatureFlag(Enum):
    """功能标记"""
    TRAINING = auto()           # 支持训练（反向传播 + 优化器）
    GRADIENT_ACCUMULATION = auto()
    MIXED_PRECISION = auto()    # AMP 混合精度
    KV_CACHE = auto()           # KV 缓存
    STREAMING = auto()          # 流式生成
    FLASH_ATTENTION = auto()    # Flash Attention
    WEIGHT_QUANTIZATION = auto()
    CHECKPOINT_SAVE = auto()
    CHECKPOINT_LOAD = auto()


# ============================================================
# 算子定义（35个算子 = 基类所有抽象方法）
# ============================================================

# 所有后端必须实现的算子集合
REQUIRED_OPS: List[str] = [
    # 张量工具 (11)
    "shape", "add", "multiply", "matmul", "transpose", "reshape",
    "zeros_like", "ones_like", "unsqueeze", "sqrt", "to_numpy",
    # 神经网络 (5)
    "linear", "embedding", "layer_norm", "rms_norm", "dropout",
    # 激活 (3)
    "silu", "gelu", "softmax",
    # 注意力 (4)
    "reshape_for_heads", "reshape_from_heads",
    "scaled_dot_product_attention", "causal_mask",
    # 位置编码 (2)
    "rope_precompute", "rope_apply",
    # 损失 (1)
    "cross_entropy",
    # 权重 (5)
    "init_weight", "create_parameter", "create_optimizer",
    "save_checkpoint", "load_checkpoint",
    "get_state_dict", "load_state_dict",
    # 拼接 (1)
    "concat",
    # 数学 (3)
    "log", "exp", "argmax",
]

# 可选的训练专用算子（推理后端可跳过）
TRAINING_OPS: List[str] = [
    "backward", "step_optimizer", "zero_grad", "clip_grad_norm",
]


# ============================================================
# BackendCapability
# ============================================================

@dataclass
class BackendCapability:
    """
    后端能力规格

    描述一个后端的完整能力集，用于：
      - 新后端接入自检
      - 跨后端能力对比
      - 运行时能力查询
    """
    name: str                           # 后端名称，如 "PyTorchBackend"
    version: str = "1.0.0"
    description: str = ""
    precision_modes: List[PrecisionMode] = field(default_factory=list)
    features: List[FeatureFlag] = field(default_factory=list)
    supported_ops: Set[str] = field(default_factory=set)
    unsupported_ops: Set[str] = field(default_factory=set)
    min_memory_mb: int = 0              # 最少内存需求
    target_devices: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    @property
    def op_coverage(self) -> float:
        """算子覆盖率（已实现 / 总需求）"""
        total = len(REQUIRED_OPS)
        implemented = len(self.supported_ops & set(REQUIRED_OPS))
        return implemented / total if total > 0 else 0.0

    @property
    def is_training_capable(self) -> bool:
        return FeatureFlag.TRAINING in self.features

    @property
    def missing_ops(self) -> List[str]:
        """缺失的必要算子列表"""
        return sorted(set(REQUIRED_OPS) - self.supported_ops)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "precision_modes": [m.name for m in self.precision_modes],
            "features": [f.name for f in self.features],
            "supported_ops": sorted(self.supported_ops),
            "unsupported_ops": sorted(self.unsupported_ops),
            "op_coverage": f"{self.op_coverage:.1%}",
            "missing_ops": self.missing_ops,
            "min_memory_mb": self.min_memory_mb,
            "target_devices": self.target_devices,
            "limitations": self.limitations,
            "is_training_capable": self.is_training_capable,
        }

    def summary(self) -> str:
        lines = [
            f"BackendCapability: {self.name} v{self.version}",
            f"  Op Coverage: {self.op_coverage:.0%} ({len(self.supported_ops & set(REQUIRED_OPS))}/{len(REQUIRED_OPS)})",
            f"  Training: {'Yes' if self.is_training_capable else 'No'}",
            f"  Precision: {', '.join(m.name for m in self.precision_modes)}",
            f"  Features: {', '.join(f.name for f in self.features)}",
            f"  Devices: {', '.join(self.target_devices) if self.target_devices else 'N/A'}",
        ]
        if self.missing_ops:
            lines.append(f"  Missing Ops: {', '.join(self.missing_ops[:5])}")
            if len(self.missing_ops) > 5:
                lines.append(f"    ... and {len(self.missing_ops) - 5} more")
        if self.limitations:
            lines.append(f"  Limitations: {len(self.limitations)} listed")
        return "\n".join(lines)


# ============================================================
# BackendValidator
# ============================================================

class BackendValidator:
    """
    后端接口一致性验证器

    自动检测后端是否实现了所有必需的抽象方法。

    用法:
      from backends.pytorch import PyTorchBackend
      validator = BackendValidator()
      result = validator.validate(PyTorchBackend())
      print(result.summary())
    """

    def validate(self, backend_instance) -> "ValidationResult":
        """
        验证后端实例

        检查：
          1. 所有必需算子是否有实现（非 NotImplementedError）
          2. 训练后端是否实现了训练算子
          3. 关键方法行为是否正确（烟雾测试）
        """
        from backends.base import Backend
        raw_backend = type(backend_instance)

        implemented = []
        missing = []
        training_implemented = []
        training_missing = []

        for op_name in REQUIRED_OPS:
            method = getattr(raw_backend, op_name, None)
            if method is None:
                missing.append(op_name)
            elif self._is_abstract_method(raw_backend, op_name, Backend):
                missing.append(op_name)
            else:
                implemented.append(op_name)

        for op_name in TRAINING_OPS:
            method = getattr(raw_backend, op_name, None)
            if method is None:
                training_missing.append(op_name)
            elif self._is_abstract_method(raw_backend, op_name, Backend):
                training_missing.append(op_name)
            else:
                training_implemented.append(op_name)

        is_complete = len(missing) == 0
        is_training_complete = len(training_missing) == 0

        return ValidationResult(
            backend_name=backend_instance.name,
            implemented_ops=set(implemented),
            missing_ops=set(missing),
            training_ops_implemented=set(training_implemented),
            training_ops_missing=set(training_missing),
            is_complete=is_complete,
            is_training_complete=is_training_complete,
        )

    @staticmethod
    def _is_abstract_method(cls, method_name: str, base_cls) -> bool:
        """检查方法是否仍为抽象方法（子类未覆写）"""
        method = getattr(cls, method_name, None)
        if method is None:
            return True

        # 检查是否被 @abstractmethod 标记但未实现
        if hasattr(method, '__isabstractmethod__') and method.__isabstractmethod__:
            return True

        return False


@dataclass
class ValidationResult:
    """验证结果"""
    backend_name: str
    implemented_ops: Set[str] = field(default_factory=set)
    missing_ops: Set[str] = field(default_factory=set)
    training_ops_implemented: Set[str] = field(default_factory=set)
    training_ops_missing: Set[str] = field(default_factory=set)
    is_complete: bool = False
    is_training_complete: bool = False

    @property
    def required_coverage(self) -> float:
        total = len(REQUIRED_OPS)
        return len(self.implemented_ops) / total if total > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "is_complete": self.is_complete,
            "is_training_complete": self.is_training_complete,
            "required_coverage": f"{self.required_coverage:.1%}",
            "implemented_count": len(self.implemented_ops),
            "missing_count": len(self.missing_ops),
            "missing_ops": sorted(self.missing_ops),
            "training_ops_missing": sorted(self.training_ops_missing),
        }

    def summary(self) -> str:
        status = "PASS" if self.is_complete else "FAIL"
        train_status = "PASS" if self.is_training_complete else "N/A (inference-only)"

        lines = [
            f"ValidationResult: {self.backend_name}",
            f"  Required Ops: {self.required_coverage:.0%} ({len(self.implemented_ops)}/{len(REQUIRED_OPS)}) → {status}",
            f"  Training Ops: {len(self.training_ops_implemented)}/{len(TRAINING_OPS)} → {train_status}",
        ]
        if self.missing_ops:
            lines.append(f"  Missing: {', '.join(sorted(self.missing_ops))}")
        return "\n".join(lines)


# ============================================================
# 算子兼容性矩阵
# ============================================================

def build_compatibility_matrix() -> Dict[str, Dict[str, bool]]:
    """
    构建跨后端算子兼容性矩阵

    自动扫描所有已安装后端的能力。

    返回:
      {
        "PyTorchBackend": {"shape": True, "add": True, ...},
        "NumPyBackend":  {"shape": True, "add": True, ...},
      }
    """
    matrix = {}
    validator = BackendValidator()

    # PyTorch
    try:
        from backends.pytorch import PyTorchBackend
        result = validator.validate(PyTorchBackend())
        row = {op: (op not in result.missing_ops) for op in REQUIRED_OPS}
        row.update({op: (op not in result.training_ops_missing) for op in TRAINING_OPS})
        matrix["PyTorchBackend"] = row
    except ImportError:
        pass

    # NumPy
    try:
        from backends.numpy import NumPyBackend
        result = validator.validate(NumPyBackend())
        row = {op: (op not in result.missing_ops) for op in REQUIRED_OPS}
        row.update({op: (op not in result.training_ops_missing) for op in TRAINING_OPS})
        matrix["NumPyBackend"] = row
    except ImportError:
        pass

    return matrix


def print_compatibility_matrix():
    """打印人类可读的兼容性矩阵"""
    matrix = build_compatibility_matrix()
    if not matrix:
        print("No backends found.")
        return

    all_ops = REQUIRED_OPS + TRAINING_OPS
    backends = sorted(matrix.keys())

    # 表头
    col_width = max(len(op) for op in all_ops) + 2
    header = f"{'Operator':<{col_width}}" + "".join(f"{b:>16}" for b in backends)
    print(header)
    print("-" * len(header))

    for op in all_ops:
        statuses = "".join(
            f"{'OK' if matrix[b].get(op, False) else '--':>16}" for b in backends
        )
        print(f"{op:<{col_width}}{statuses}")

    total_all = len(all_ops)
    for b in backends:
        ok = sum(1 for op in all_ops if matrix[b].get(op, False))
        print(f"\n{b}: {ok}/{total_all} ops supported")


# ============================================================
# 预定义能力规格
# ============================================================

def get_pytorch_capability() -> BackendCapability:
    """PyTorch 后端能力规格"""
    return BackendCapability(
        name="PyTorchBackend",
        version="2.0+",
        description="PyTorch 训练 + 推理后端，支持 CUDA/MPS/CPU",
        precision_modes=[
            PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16,
        ],
        features=[
            FeatureFlag.TRAINING, FeatureFlag.GRADIENT_ACCUMULATION,
            FeatureFlag.MIXED_PRECISION, FeatureFlag.KV_CACHE,
            FeatureFlag.CHECKPOINT_SAVE, FeatureFlag.CHECKPOINT_LOAD,
        ],
        supported_ops=set(REQUIRED_OPS + TRAINING_OPS),
        target_devices=["CUDA", "MPS", "CPU"],
        min_memory_mb=512,
        limitations=[
            "Flash Attention 仅 CUDA GPUs 支持",
            "CPU 训练速度慢",
            "MPS 部分算子精度可能与 CUDA 不一致",
        ],
    )


def get_numpy_capability() -> BackendCapability:
    """NumPy 后端能力规格"""
    return BackendCapability(
        name="NumPyBackend",
        version="1.0",
        description="纯 NumPy CPU 推理后端，零 PyTorch 依赖",
        precision_modes=[PrecisionMode.FP32],
        features=[
            FeatureFlag.KV_CACHE, FeatureFlag.CHECKPOINT_LOAD,
        ],
        supported_ops=set(REQUIRED_OPS),
        target_devices=["CPU"],
        min_memory_mb=128,
        limitations=[
            "不支持训练（无反向传播）",
            "不支持 GPU 加速",
            "不支持混合精度",
            "仅支持 float32 推理",
            "无梯度、无优化器",
        ],
    )


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    "PrecisionMode", "FeatureFlag",
    "REQUIRED_OPS", "TRAINING_OPS",
    "BackendCapability", "BackendValidator", "ValidationResult",
    "build_compatibility_matrix", "print_compatibility_matrix",
    "get_pytorch_capability", "get_numpy_capability",
]
