#!/usr/bin/env python
"""
权重量化工具 (Weight Quantization)

支持 INT8 对称量化和 INT4 分组量化，为 GGUF 量化导出做准备。

量化方法:
  - INT8 对称量化 (Q8_0): per-tensor scale, [-127, 127]
  - INT4 分组量化 (Q4_K_M): per-group (32/128) scale + min, [0, 15]
  - 量化误差评估: L1/L2/余弦相似度

用法:
  from tools.quantize import quantize_weights_int8, quantize_error
  quantized = quantize_weights_int8(model, backend)
  error = quantize_error(original, quantized)
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


# ============================================================
# 数据类型
# ============================================================

@dataclass
class QuantizedTensor:
    """
    量化后的张量

    INT8: data 为 int8 值, scale 为 float32
    INT4: data 为 int4 (packed 2 per byte), scale + mins 为 float32
    """
    data: Any            # 量化后的整数值
    scale: Any           # 缩放因子 (float32)
    zero_point: Any = None  # 零点 (非对称量化)
    mins: Any = None     # INT4 分组量化专用: per-group minimum
    group_size: int = 0  # 分组大组
    method: str = "int8" # int8 / int4


# ============================================================
# INT8 对称量化
# ============================================================

def quantize_int8(tensor) -> QuantizedTensor:
    """
    INT8 对称量化: q = round(x / scale), scale = max(|x|) / 127

    Args:
        tensor: numpy array 或 torch tensor

    Returns:
        QuantizedTensor with int8 data and float32 scale
    """
    import numpy as np

    # 转为 numpy
    if hasattr(tensor, 'detach'):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)

    arr_f32 = arr.astype(np.float32)
    max_val = np.max(np.abs(arr_f32))

    if max_val == 0:
        scale = np.float32(1.0)
    else:
        scale = np.float32(max_val / 127.0)

    quant = np.clip(np.round(arr_f32 / scale), -127, 127).astype(np.int8)

    return QuantizedTensor(
        data=quant,
        scale=scale,
        method="int8",
    )


def dequantize_int8(qt: QuantizedTensor) -> Any:
    """
    INT8 反量化: x = q * scale

    Returns:
        float32 numpy array
    """
    import numpy as np
    data_f32 = qt.data.astype(np.float32)
    return data_f32 * qt.scale


def quantize_weights_int8(model, backend) -> Dict[str, QuantizedTensor]:
    """
    对模型所有权重进行 INT8 量化

    Args:
        model: IR TransformerModel
        backend: Backend 实例

    Returns:
        {param_name: QuantizedTensor} 量化后的权重字典
    """
    state_dict = backend.get_state_dict(model)
    quantized = {}

    for name, tensor in state_dict.items():
        qt = quantize_int8(tensor)
        quantized[name] = qt

    return quantized


# ============================================================
# INT4 分组量化 (GGUF Q4_K_M style)
# ============================================================

def quantize_int4_group(tensor, group_size: int = 32) -> QuantizedTensor:
    """
    INT4 分组量化 (Q4_K_M 风格)

    每组 group_size 个元素独立量化:
      - data: int4 值（每 2 个元素打包为 1 byte）
      - scale: 每组的缩放因子
      - mins: 每组的最小值

    Args:
        tensor: numpy array (1D or 2D, will be flattened per row)
        group_size: 分组大小 (默认 128 for Q4_K_M, 32 for fast)

    Returns:
        QuantizedTensor
    """
    import numpy as np

    if hasattr(tensor, 'detach'):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)

    arr_f32 = arr.astype(np.float32)

    # 填充到 group_size 的倍数
    original_size = arr_f32.size
    padded_size = ((original_size + group_size - 1) // group_size) * group_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[:original_size] = arr_f32.flatten()

    n_groups = padded_size // group_size
    scales = np.zeros(n_groups, dtype=np.float32)
    mins_arr = np.zeros(n_groups, dtype=np.float32)
    quant_data = np.zeros(padded_size // 2, dtype=np.uint8)  # 2 per byte

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_data = padded[start:end]

        g_min = np.min(group_data)
        g_max = np.max(group_data)
        g_range = g_max - g_min

        if g_range == 0:
            scale = 1.0
        else:
            scale = g_range / 15.0  # 4-bit: 0-15

        scales[g] = np.float32(scale)
        mins_arr[g] = np.float32(g_min)

        # 量化: q = round((x - min) / scale), clip to [0, 15]
        quant = np.clip(np.round((group_data - g_min) / scale), 0, 15).astype(np.uint8)

        # 打包: [q0, q1] → byte (q0 | q1<<4)
        for j in range(0, group_size, 2):
            byte_val = quant[j] | (quant[j + 1] << 4)
            quant_data[start // 2 + j // 2] = byte_val

    return QuantizedTensor(
        data=quant_data,
        scale=scales,
        mins=mins_arr,
        group_size=group_size,
        method="int4",
    )


def dequantize_int4_group(qt: QuantizedTensor, original_shape: tuple = None) -> Any:
    """
    INT4 分组反量化

    Returns:
        float32 numpy array
    """
    import numpy as np

    # 解包
    quant_unpacked = np.zeros(len(qt.data) * 2, dtype=np.uint8)
    for i, byte_val in enumerate(qt.data):
        quant_unpacked[i * 2] = byte_val & 0x0F
        quant_unpacked[i * 2 + 1] = (byte_val >> 4) & 0x0F

    group_size = qt.group_size
    n_groups = len(qt.scale)
    result = np.zeros(n_groups * group_size, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        result[start:end] = quant_unpacked[start:end].astype(np.float32) * qt.scale[g] + qt.mins[g]

    if original_shape:
        total = 1
        for dim in original_shape:
            total *= dim
        result = result[:total].reshape(original_shape)

    return result


# ============================================================
# 量化误差评估
# ============================================================

def compute_quantization_error(original, quantized: QuantizedTensor) -> Dict[str, float]:
    """
    计算量化误差

    Args:
        original: 原始 float32 张量
        quantized: QuantizedTensor

    Returns:
        {
            "l1_error": float,          # L1 平均误差
            "l2_error": float,          # L2 均方根误差
            "cosine_similarity": float, # 余弦相似度
            "max_abs_error": float,     # 最大绝对误差
            "relative_error": float,    # 相对误差 (L2_norm(error) / L2_norm(original))
        }
    """
    import numpy as np

    if hasattr(original, 'detach'):
        orig = original.detach().cpu().numpy().flatten().astype(np.float32)
    else:
        orig = np.array(original).flatten().astype(np.float32)

    # 反量化
    if quantized.method == "int8":
        recon = dequantize_int8(quantized).flatten()
    elif quantized.method == "int4":
        recon = dequantize_int4_group(quantized, None).flatten()
    else:
        recon = np.zeros_like(orig)

    # 对齐长度
    min_len = min(len(orig), len(recon))
    orig = orig[:min_len]
    recon = recon[:min_len]

    diff = orig - recon
    l1_error = np.mean(np.abs(diff))
    l2_error = np.sqrt(np.mean(diff ** 2))
    max_abs = np.max(np.abs(diff))

    # 余弦相似度
    orig_norm = np.linalg.norm(orig)
    recon_norm = np.linalg.norm(recon)
    if orig_norm > 1e-10 and recon_norm > 1e-10:
        cosine = np.dot(orig, recon) / (orig_norm * recon_norm)
    else:
        cosine = 1.0

    # 相对误差
    rel_error = np.linalg.norm(diff) / max(orig_norm, 1e-10)

    return {
        "l1_error": float(l1_error),
        "l2_error": float(l2_error),
        "cosine_similarity": float(cosine),
        "max_abs_error": float(max_abs),
        "relative_error": float(rel_error),
    }


def evaluate_model_quantization(model, backend) -> Dict:
    """
    对模型所有权重进行量化并评估误差

    Returns:
        {
            "method": str,
            "total_params": int,
            "per_layer": {name: {error_metrics}},
            "summary": {avg_l1, avg_l2, avg_cosine, ...}
        }
    """
    state_dict = backend.get_state_dict(model)
    total_params = 0
    per_layer = {}
    l1s = []
    l2s = []
    cosines = []

    for name, tensor in state_dict.items():
        shape = tensor.shape if hasattr(tensor, 'shape') else tensor.shape
        n_params = 1
        for dim in shape:
            n_params *= dim
        total_params += n_params

        qt = quantize_int8(tensor)
        error = compute_quantization_error(tensor, qt)

        per_layer[name] = {
            "shape": tuple(shape),
            "params": n_params,
            **error,
        }
        l1s.append(error["l1_error"])
        l2s.append(error["l2_error"])
        cosines.append(error["cosine_similarity"])

    return {
        "method": "int8",
        "total_params": total_params,
        "per_layer": per_layer,
        "summary": {
            "avg_l1_error": sum(l1s) / len(l1s) if l1s else 0,
            "avg_l2_error": sum(l2s) / len(l2s) if l2s else 0,
            "avg_cosine_similarity": sum(cosines) / len(cosines) if cosines else 0,
            "min_cosine": min(cosines) if cosines else 0,
            "max_l2_error": max(l2s) if l2s else 0,
        },
    }


def print_quantization_report(report: Dict):
    """打印量化报告"""
    print("=" * 60)
    print(f"权重量化报告 ({report['method'].upper()})")
    print("=" * 60)
    print(f"  总参数量: {report['total_params']:,}")
    print(f"  层数: {len(report['per_layer'])}")
    print(f"")
    print(f"  误差摘要:")
    s = report["summary"]
    print(f"    Avg L1 Error:       {s['avg_l1_error']:.6f}")
    print(f"    Avg L2 Error:       {s['avg_l2_error']:.6f}")
    print(f"    Avg Cosine Sim:     {s['avg_cosine_similarity']:.6f}")
    print(f"    Min Cosine Sim:     {s['min_cosine']:.6f}")
    print(f"    Max L2 Error:       {s['max_l2_error']:.6f}")
    print(f"")
    print(f"  各层误差 (sorted by L2):")
    sorted_layers = sorted(
        report["per_layer"].items(),
        key=lambda x: x[1]["l2_error"],
        reverse=True,
    )[:10]
    for name, info in sorted_layers:
        shape_str = "x".join(str(d) for d in info["shape"])
        print(f"    {name:<40} {shape_str:<20} "
              f"L2={info['l2_error']:.4f}  cos={info['cosine_similarity']:.4f}")
    print("=" * 60)


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    "QuantizedTensor",
    "quantize_int8", "dequantize_int8", "quantize_weights_int8",
    "quantize_int4_group", "dequantize_int4_group",
    "compute_quantization_error", "evaluate_model_quantization",
    "print_quantization_report",
]
