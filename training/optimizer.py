"""
training/optimizer.py — 优化器创建工具
"""

from typing import Any, List, Tuple


def create_optimizer(backend, parameters: List[Tuple[str, Any]], lr: float,
                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01) -> Any:
    """
    创建优化器（委托给后端）

    Args:
        backend: PyTorchBackend 实例
        parameters: [(name, param), ...] 参数列表
        lr: 学习率
        betas: Adam betas
        eps: Adam epsilon
        weight_decay: 权重衰减

    Returns:
        优化器实例
    """
    return backend.create_optimizer(
        parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
