"""
检查点转换工具
--------------
将旧版 MiniLLM 的 .pt 权重转换为新版 IR 架构的格式。

用法:
    python tools/convert_checkpoint.py --old checkpoints/best_model.pt --new checkpoints/best_model_v2.pt
"""

import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_old_to_new(old_path: str, new_path: str, config_dict: dict = None):
    """
    将旧版 MiniLLM 检查点转换为新版 IR 格式

    旧版权重命名格式:
        token_embeddings.weight
        layers.{i}.attention.q_proj.weight
        layers.{i}.attention.k_proj.weight
        layers.{i}.attention.v_proj.weight
        layers.{i}.attention.out_proj.weight
        layers.{i}.feed_forward.linear1.weight
        layers.{i}.feed_forward.linear2.weight
        layers.{i}.feed_forward.linear_gate.weight  (SwiGLU)
        layers.{i}.norm1.weight
        layers.{i}.norm1.bias
        layers.{i}.norm2.weight
        layers.{i}.norm2.bias
        norm.weight
        norm.bias
        lm_head.weight

    新版权重命名格式:
        embedding.weight
        block_{i}.attn_norm.weight
        block_{i}.attn.q_proj.weight
        block_{i}.attn.k_proj.weight
        block_{i}.attn.v_proj.weight
        block_{i}.attn.o_proj.weight
        block_{i}.ffn_norm.weight
        block_{i}.ffn.w1.weight
        block_{i}.ffn.w2.weight
        block_{i}.ffn.wg.weight  (SwiGLU)
        final_norm.weight
        lm_head.weight
    """

    print(f"Loading old checkpoint: {old_path}")
    checkpoint = torch.load(old_path, map_location='cpu', weights_only=False)

    # 获取权重字典
    if 'model_state_dict' in checkpoint:
        old_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        old_state = checkpoint['state_dict']
    else:
        old_state = checkpoint

    new_state = {}

    # 映射规则
    key_map = {
        'token_embeddings.weight': 'embedding.weight',
        'norm.weight': 'final_norm.weight',
        'lm_head.weight': 'lm_head.weight',
    }

    for old_key, value in old_state.items():
        new_key = None

        # 直接映射
        if old_key in key_map:
            new_key = key_map[old_key]
        # Layer 参数
        elif old_key.startswith('layers.'):
            parts = old_key.split('.')
            layer_idx = parts[1]  # e.g., "0", "1"
            component = '.'.join(parts[2:])

            component_map = {
                'norm1.weight': f'block_{layer_idx}.attn_norm.weight',
                'attention.q_proj.weight': f'block_{layer_idx}.attn.q_proj.weight',
                'attention.k_proj.weight': f'block_{layer_idx}.attn.k_proj.weight',
                'attention.v_proj.weight': f'block_{layer_idx}.attn.v_proj.weight',
                'attention.out_proj.weight': f'block_{layer_idx}.attn.o_proj.weight',
                'norm2.weight': f'block_{layer_idx}.ffn_norm.weight',
                'feed_forward.linear1.weight': f'block_{layer_idx}.ffn.w1.weight',
                'feed_forward.linear2.weight': f'block_{layer_idx}.ffn.w2.weight',
                'feed_forward.linear_gate.weight': f'block_{layer_idx}.ffn.wg.weight',
            }

            if component in component_map:
                new_key = component_map[component]

        if new_key is not None:
            new_state[new_key] = value.clone()
            print(f"  {old_key} → {new_key}")
        else:
            print(f"  [SKIP] {old_key} (未找到映射)")

    # 保存新格式检查点
    new_checkpoint = {
        'state_dict': new_state,
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'best_perplexity': checkpoint.get('best_perplexity', float('inf')),
        'training_history': checkpoint.get('training_history', []),
        'format': 'ir_v2',
    }

    torch.save(new_checkpoint, new_path)
    print(f"\nConverted checkpoint saved to: {new_path}")
    print(f"  Mapped {len(new_state)} parameters")
    print(f"  Total tensors in source: {len(old_state)}")

    return new_path


def main():
    parser = argparse.ArgumentParser(description='Convert old MiniLLM checkpoint to new IR format')
    parser.add_argument('--old', type=str, required=True, help='Path to old checkpoint')
    parser.add_argument('--new', type=str, required=True, help='Path to save new checkpoint')
    args = parser.parse_args()

    convert_old_to_new(args.old, args.new)


if __name__ == '__main__':
    main()
