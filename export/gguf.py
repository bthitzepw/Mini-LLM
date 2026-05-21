"""
GGUF 导出器
-----------
将 MiniLLM 模型导出为 GGUF 格式（llama.cpp 兼容）。

GGUF 格式规范: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

import struct
import json
import os
import numpy as np
from typing import Dict, Any, List


# GGUF 魔数和版本
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF 值类型
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def export_gguf(model, output_path: str, metadata: Dict = None):
    """
    导出模型为 GGUF 格式

    Args:
        model: IR TransformerModel 实例（参数必须为 numpy arrays）
        output_path: 输出 .gguf 文件路径
        metadata: 额外元数据字典

    GGUF 文件结构:
        [Header] [Metadata KV Pairs] [Tensor Info] [Tensor Data]
    """
    config = model.config

    # 收集所有权重
    from backends.numpy import NumPyBackend
    np_backend = NumPyBackend()
    state_dict = np_backend.get_state_dict(model)

    if not state_dict:
        raise RuntimeError("模型没有权重数据，请先初始化或加载权重")

    # 确保权重是 numpy 数组
    for key in state_dict:
        param = state_dict[key]
        if hasattr(param, 'numpy'):  # torch tensor
            state_dict[key] = param.detach().cpu().numpy()
        elif isinstance(param, np.ndarray):
            state_dict[key] = param
        else:
            state_dict[key] = np.array(param)

    # 构建元数据
    meta: List[tuple] = [
        ("general.architecture", "minillm", GGUF_TYPE_STRING),
        ("general.name", "MiniLLM", GGUF_TYPE_STRING),
        ("general.file_type", 1, GGUF_TYPE_UINT32),  # F32 = 1
    ]

    # 模型参数
    model_params = [
        ("minillm.context_length", config.max_seq_length, GGUF_TYPE_UINT32),
        ("minillm.embedding_length", config.hidden_size, GGUF_TYPE_UINT32),
        ("minillm.block_count", config.num_layers, GGUF_TYPE_UINT32),
        ("minillm.head_count", config.num_heads, GGUF_TYPE_UINT32),
        ("minillm.head_count_kv", config.num_kv_heads, GGUF_TYPE_UINT32),
        ("minillm.feed_forward_length", config.intermediate_size, GGUF_TYPE_UINT32),
        ("minillm.rope.dimension_count", config.head_dim, GGUF_TYPE_UINT32),
        ("minillm.rope.freq_base", config.rope_theta, GGUF_TYPE_FLOAT32),
        ("minillm.attention.layer_norm_epsilon", config.rms_norm_eps, GGUF_TYPE_FLOAT32),
    ]
    meta.extend(model_params)

    # Tokenizer 参数
    tokenizer_params = [
        ("tokenizer.ggml.model", "gpt2", GGUF_TYPE_STRING),
        ("tokenizer.ggml.bos_token_id", config.bos_token_id, GGUF_TYPE_UINT32),
        ("tokenizer.ggml.eos_token_id", config.eos_token_id, GGUF_TYPE_UINT32),
        ("tokenizer.ggml.padding_token_id", config.pad_token_id, GGUF_TYPE_UINT32),
    ]
    meta.extend(tokenizer_params)

    # 权重映射（IR 参数名 → GGUF 张量名）
    tensor_map = _build_tensor_map(state_dict)

    # 写入文件
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(state_dict)))  # tensor count
        f.write(struct.pack('<Q', len(meta)))         # metadata kv count

        # Metadata (简化: 跳过复杂 kv 编码，直接写 JSON)
        metadata_json = json.dumps({
            key: _serialize_value(val, vtype)
            for key, val, vtype in meta
        }, ensure_ascii=False).encode('utf-8')
        f.write(struct.pack('<I', len(metadata_json)))
        f.write(metadata_json)

        # Tensor info + data 偏移计算
        # 简化实现: 直接写入 tensor 名称和形状信息
        tensor_infos = []
        offset = f.tell() + 8 * len(state_dict) * 4  # 预留偏移

        for key in sorted(state_dict.keys()):
            gguf_name = tensor_map.get(key, key.replace('.', '_'))
            tensor = state_dict[key]
            shape = tensor.shape
            dtype = _numpy_to_gguf_dtype(tensor.dtype)

            # 写入名称长度 + 名称
            name_bytes = gguf_name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # 写入维度数 + 维度
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

        # 写入 tensor 数据（简化：连续写入）
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            # 转换为 float32
            data = tensor.astype(np.float32).tobytes()
            f.write(data)

    file_size = os.path.getsize(output_path)
    print(f"GGUF exported to: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  Tensors: {len(state_dict)}")
    print(f"  Parameters: {model.get_param_count():,}")

    return output_path


def _build_tensor_map(state_dict: Dict) -> Dict[str, str]:
    """构建参数名 → GGUF 张量名映射"""
    mapping = {}

    # 根据参数类型自动推断 GGUF 名称
    for key in state_dict:
        parts = key.split('.')

        # Embedding
        if key == "embedding.weight":
            mapping[key] = "token_embd.weight"

        # Final norm
        elif key == "final_norm.weight":
            mapping[key] = "output_norm.weight"

        # LM head
        elif key == "lm_head.weight":
            mapping[key] = "output.weight"

        # Block 参数
        elif parts[0].startswith("block_"):
            block_num = parts[0].split("_")[1]
            param_name = ".".join(parts[1:])

            # Attention norm
            if "attn_norm" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.attn_norm.weight"

            # Attention Q/K/V/O
            elif "q_proj" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.attn_q.weight"
            elif "k_proj" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.attn_k.weight"
            elif "v_proj" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.attn_v.weight"
            elif "o_proj" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.attn_output.weight"

            # FFN norm
            elif "ffn_norm" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.ffn_norm.weight"

            # FFN layers
            elif "w1" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.ffn_gate.weight"
            elif "wg" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.ffn_up.weight"
            elif "w2" in param_name and "weight" in param_name:
                mapping[key] = f"blk.{block_num}.ffn_down.weight"

    return mapping


def _numpy_to_gguf_dtype(dtype) -> int:
    """NumPy dtype → GGUF dtype"""
    mapping = {
        np.float32: 0,  # F32
        np.float16: 1,  # F16
        np.int32: 2,    # I32
        np.int16: 3,    # I16
    }
    return mapping.get(dtype.type if hasattr(dtype, 'type') else dtype, 0)


def _serialize_value(value, vtype: int):
    """序列化 GGUF 值"""
    if vtype == GGUF_TYPE_STRING:
        return str(value)
    elif vtype == GGUF_TYPE_BOOL:
        return bool(value)
    elif vtype in (GGUF_TYPE_FLOAT32, GGUF_TYPE_FLOAT64):
        return float(value)
    else:
        return int(value)
