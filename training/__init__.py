"""
training/ — 训练模块
--------------------
后端无关的训练器。

traineer 不关心模型是 Transformer 还是别的什么结构，
只知道：Layer + Graph + Backend
"""

from training.trainer import Trainer
from training.optimizer import create_optimizer
