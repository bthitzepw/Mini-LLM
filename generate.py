"""
MiniLLM v2 交互式文本生成

用法:
  python generate.py                        # 交互模式
  python generate.py --prompt "def hello("  # 单次生成
  python generate.py --backend numpy        # 使用 NumPy 后端
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir.config import ModelConfig
from ir.transformer import TransformerModel
from inference.engine import InferenceEngine
from src.tokenizer import SimpleTokenizer


def load_config(config_path='config/config.yaml'):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='MiniLLM v2 Text Generation')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Input prompt (single-shot mode)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'pytorch', 'numpy'],
                       help='Backend selection')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-K sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-P (nucleus) sampling')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    # 加载配置
    config_dict = load_config()

    # 构建 IR 模型
    mc = ModelConfig.from_yaml(config_dict)
    model = TransformerModel(mc)

    # 选择后端
    if args.backend == 'numpy':
        from backends.numpy import NumPyBackend
        backend = NumPyBackend()
    elif args.backend == 'pytorch':
        from backends.pytorch import PyTorchBackend
        backend = PyTorchBackend(device=args.device)
    else:
        backend = None  # auto-select

    # 创建 tokenizer
    tokenizer = SimpleTokenizer(vocab_size=mc.vocab_size)

    # 创建推理引擎
    engine = InferenceEngine(
        model,
        backend=backend,
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        device=args.device
    )

    # 设置采样参数
    engine.temperature = args.temperature
    engine.top_k = args.top_k
    engine.top_p = args.top_p

    print(f"\nMiniLLM v2 — Inference Engine")
    print(f"  Backend: {engine.backend.name}")
    print(f"  Parameters: {model.get_param_count():,}")
    print(f"  Temperature: {engine.temperature}")
    print(f"  Top-K: {engine.top_k}")
    print(f"  Top-P: {engine.top_p}")
    print()

    # 单次生成模式
    if args.prompt:
        output = engine.generate(args.prompt, max_new_tokens=args.max_tokens)
        print(output)
        return

    # 交互模式
    print("Interactive generation mode (type ':quit' to exit)")
    print("Commands: :temp <val> | :topk <val> | :topp <val> | :len <val>")
    print("="*50)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt == ':quit':
            break

        # 命令处理
        if prompt.startswith(':temp'):
            try:
                engine.temperature = float(prompt.split()[1])
                print(f"Temperature set to {engine.temperature}")
            except:
                print("Usage: :temp <value>")
            continue

        if prompt.startswith(':topk'):
            try:
                engine.top_k = int(prompt.split()[1])
                print(f"Top-K set to {engine.top_k}")
            except:
                print("Usage: :topk <value>")
            continue

        if prompt.startswith(':topp'):
            try:
                engine.top_p = float(prompt.split()[1])
                print(f"Top-P set to {engine.top_p}")
            except:
                print("Usage: :topp <value>")
            continue

        if prompt.startswith(':len'):
            try:
                args.max_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {args.max_tokens}")
            except:
                print("Usage: :len <value>")
            continue

        if prompt.startswith(':info'):
            info = engine.info()
            for k, v in info.items():
                print(f"  {k}: {v}")
            continue

        # 生成
        output = engine.generate(prompt, max_new_tokens=args.max_tokens)
        print(f"\n{output}")


if __name__ == '__main__':
    main()
