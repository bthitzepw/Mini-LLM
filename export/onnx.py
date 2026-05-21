"""
ONNX 导出器
-----------
将 MiniLLM 模型导出为 ONNX 格式。

ONNX Runtime 可用于跨平台高效推理。
"""

import os
import numpy as np
from typing import Dict, Any


def export_onnx(model, output_path: str, input_shape: tuple = (1, 32)):
    """
    导出模型为 ONNX 格式

    Args:
        model: IR TransformerModel 实例
        output_path: 输出 .onnx 文件路径
        input_shape: 输入形状 (batch, seq_len)

    注意: ONNX 导出需要 PyTorch 后端。
    """
    try:
        import torch
    except ImportError:
        raise ImportError("ONNX 导出需要 PyTorch。请先安装: pip install torch onnx")

    # 确保使用 PyTorch 后端
    from backends.pytorch import PyTorchBackend, init_model_weights

    backend = PyTorchBackend(device="cpu")

    # 如果模型还没有权重，尝试初始化
    if model.embedding.params.get("weight") is None:
        print("模型未初始化权重，使用随机权重进行 ONNX 导出...")
        init_model_weights(model, backend)
    else:
        # 确保权重在 CPU 上
        for layer_name in model.get_layer_names():
            pass  # 权重已在加载时放到正确位置

    model.eval()

    # 创建 dummy 输入
    batch_size, seq_len = input_shape
    dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    # 导出 ONNX
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, ir_model, pt_backend):
            super().__init__()
            self.ir_model = ir_model
            self.pt_backend = pt_backend

        def forward(self, input_ids):
            return self.ir_model.forward(input_ids, self.pt_backend)

    wrapper = ONNXWrapper(model, backend)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path)
    print(f"ONNX exported to: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  Input shape: {input_shape}")
    print(f"  Dynamic axes: batch, sequence")

    return output_path
