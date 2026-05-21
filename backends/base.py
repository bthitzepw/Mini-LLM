"""
Backend 抽象接口
----------------
定义所有计算后端必须实现的方法。

每个 Backend 子类提供具体框架（PyTorch / NumPy / MLX 等）的计算实现。
IR 层的模型通过调用 backend.xxx() 来完成计算，不直接依赖任何框架。
"""

from typing import Any, Tuple, Optional, Dict
from abc import ABC, abstractmethod


class Backend(ABC):
    """计算后端抽象基类"""

    name: str = "base"

    # ============================================================
    # 张量工具方法
    # ============================================================

    @abstractmethod
    def shape(self, x: Any) -> Tuple[int, ...]:
        """返回张量的形状"""
        ...

    @abstractmethod
    def add(self, a, b) -> Any:
        """逐元素加法"""
        ...

    @abstractmethod
    def multiply(self, a, b) -> Any:
        """逐元素乘法"""
        ...

    @abstractmethod
    def matmul(self, a, b) -> Any:
        """矩阵乘法"""
        ...

    @abstractmethod
    def transpose(self, x, dim0: int, dim1: int) -> Any:
        """转置"""
        ...

    @abstractmethod
    def reshape(self, x, *shape) -> Any:
        """重塑张量形状"""
        ...

    @abstractmethod
    def zeros_like(self, x) -> Any:
        """创建与 x 形状相同的全零张量"""
        ...

    @abstractmethod
    def ones_like(self, x) -> Any:
        """创建与 x 形状相同的全一张量"""
        ...

    @abstractmethod
    def unsqueeze(self, x, dim: int) -> Any:
        """在指定维度插入一个维度"""
        ...

    @abstractmethod
    def sqrt(self, x) -> Any:
        """逐元素平方根"""
        ...

    @abstractmethod
    def to_numpy(self, x) -> Any:
        """将张量转换为 numpy 数组"""
        ...

    # ============================================================
    # 神经网络基础算子
    # ============================================================

    @abstractmethod
    def linear(self, x, weight, bias=None) -> Any:
        """线性变换: y = x @ weight.T + bias"""
        ...

    @abstractmethod
    def embedding(self, input_ids, weight) -> Any:
        """Token 嵌入查表"""
        ...

    @abstractmethod
    def layer_norm(self, x, weight, bias, eps: float) -> Any:
        """Layer Normalization"""
        ...

    @abstractmethod
    def rms_norm(self, x, weight, eps: float) -> Any:
        """RMS Normalization"""
        ...

    @abstractmethod
    def dropout(self, x, p: float, training: bool) -> Any:
        """Dropout（训练时随机置零，推理时通过）"""
        ...

    # ============================================================
    # 激活函数
    # ============================================================

    @abstractmethod
    def silu(self, x) -> Any:
        """SiLU (Swish) 激活"""
        ...

    @abstractmethod
    def gelu(self, x) -> Any:
        """GELU 激活"""
        ...

    @abstractmethod
    def softmax(self, x, dim: int = -1) -> Any:
        """Softmax"""
        ...

    # ============================================================
    # 注意力
    # ============================================================

    @abstractmethod
    def reshape_for_heads(self, x, batch, seq, num_heads, head_dim) -> Any:
        """
        将 (batch, seq, hidden) → (batch, num_heads, seq, head_dim)
        """
        ...

    @abstractmethod
    def reshape_from_heads(self, x, batch, seq, hidden) -> Any:
        """
        将 (batch, num_heads, seq, head_dim) → (batch, seq, hidden)
        """
        ...

    @abstractmethod
    def scaled_dot_product_attention(
        self, q, k, v, mask=None, num_kv_heads=None,
        scale=None, dropout_p=0.0, training=True
    ) -> Any:
        """
        缩放点积注意力

        Q: (batch, num_heads, seq_q, head_dim)
        K: (batch, num_kv_heads, seq_k, head_dim)
        V: (batch, num_kv_heads, seq_k, head_dim)
        """
        ...

    @abstractmethod
    def causal_mask(self, seq_len: int) -> Any:
        """生成因果注意力掩码 (seq_len, seq_len)"""
        ...

    # ============================================================
    # 位置编码
    # ============================================================

    @abstractmethod
    def rope_precompute(self, seq_len: int, dim: int, theta: float) -> Tuple[Any, Any]:
        """预计算 RoPE 的 cos/sin 表"""
        ...

    @abstractmethod
    def rope_apply(self, q, k, cos, sin) -> Tuple[Any, Any]:
        """对 Q/K 应用旋转位置编码"""
        ...

    # ============================================================
    # 损失函数
    # ============================================================

    @abstractmethod
    def cross_entropy(self, logits, targets, ignore_index: int = -100,
                      label_smoothing: float = 0.0) -> Any:
        """交叉熵损失"""
        ...

    # ============================================================
    # 训练专用
    # ============================================================

    def backward(self, loss) -> None:
        """反向传播（仅训练后端需要）"""
        raise NotImplementedError(f"{self.name} 后端不支持反向传播")

    def step_optimizer(self, optimizer: Any) -> None:
        """执行一步优化器更新（仅训练后端需要）"""
        raise NotImplementedError(f"{self.name} 后端不支持优化器更新")

    def zero_grad(self) -> None:
        """清零梯度（仅训练后端需要）"""
        raise NotImplementedError(f"{self.name} 后端不支持梯度清零")

    def clip_grad_norm(self, parameters, max_norm: float) -> float:
        """梯度裁剪（仅训练后端需要）"""
        raise NotImplementedError(f"{self.name} 后端不支持梯度裁剪")

    # ============================================================
    # 权重管理
    # ============================================================

    @abstractmethod
    def init_weight(self, shape: Tuple[int, ...], method: str = "xavier") -> Any:
        """
        初始化权重张量

        Args:
            shape: 张量形状
            method: 初始化方法 ("xavier", "normal", "zeros", "ones")
        """
        ...

    @abstractmethod
    def create_parameter(self, shape: Tuple[int, ...], method: str = "xavier") -> Any:
        """创建可训练参数（带梯度跟踪）"""
        ...

    @abstractmethod
    def create_optimizer(self, parameters, lr: float, **kwargs) -> Any:
        """创建优化器"""
        ...

    @abstractmethod
    def save_checkpoint(self, model, path: str, extra: Dict = None) -> None:
        """保存模型权重"""
        ...

    @abstractmethod
    def load_checkpoint(self, model, path: str) -> Dict:
        """加载模型权重"""
        ...

    @abstractmethod
    def get_state_dict(self, model) -> Dict[str, Any]:
        """获取模型权重字典"""
        ...

    @abstractmethod
    def load_state_dict(self, model, state_dict: Dict[str, Any]) -> None:
        """加载权重字典到模型"""
        ...

    # ============================================================
    # 工具
    # ============================================================

    def log(self, x) -> Any:
        """自然对数"""
        raise NotImplementedError

    def exp(self, x) -> Any:
        """指数函数"""
        raise NotImplementedError

    def argmax(self, x, dim: int = -1) -> Any:
        """返回最大值的索引"""
        raise NotImplementedError

    def multinomial(self, probs, num_samples: int = 1) -> Any:
        """从概率分布中采样"""
        return probs  # 默认实现，子类覆写
