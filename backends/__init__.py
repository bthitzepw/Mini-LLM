"""
Backends — 计算后端
--------------------
提供不同框架的计算实现。

- PyTorchBackend: 用于训练（GPU + 混合精度）
- NumPyBackend: 用于纯 CPU 推理
"""

from backends.base import Backend
from backends.pytorch import PyTorchBackend, init_model_weights, collect_parameters
from backends.numpy import NumPyBackend, convert_torch_to_numpy
