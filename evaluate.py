"""
CodeSprite v2 模型评估

用法:
  python evaluate.py                              # 评估最佳模型
  python evaluate.py --checkpoint checkpoints/checkpoint_epoch_5.pt
  python evaluate.py --backend numpy              # NumPy 后端评估
"""

import os
import sys
import argparse
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir.config import ModelConfig
from ir.transformer import TransformerModel
from backends.pytorch import PyTorchBackend, init_model_weights
from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader


def load_config(config_path='config/config.yaml'):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def evaluate_model(model, dataloader, backend):
    """评估模型（计算 Loss 和 Perplexity）"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(backend.device)
            labels = batch['labels'].to(backend.device)

            logits = model.forward(input_ids, backend)

            # Shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))

    return {
        'val_loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description='CodeSprite v2 Evaluation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    config_dict = load_config()
    mc = ModelConfig.from_yaml(config_dict)
    model = TransformerModel(mc)

    # PyTorch 后端
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    backend = PyTorchBackend(device=str(device))

    # 初始化并加载权重
    init_model_weights(model, backend)

    if os.path.exists(args.checkpoint):
        backend.load_checkpoint(model, args.checkpoint)
    else:
        print(f"Warning: Checkpoint not found: {args.checkpoint}")
        print("Evaluating with random weights...")

    print(f"\nModel: {model.get_param_count():,} parameters")
    print(f"Device: {device}")
    print(f"Backend: {backend.name}")

    # 加载数据
    tokenizer = SimpleTokenizer(vocab_size=mc.vocab_size)
    test_dataset = TextDataset(
        config_dict['data']['test_file'], tokenizer, mc.max_seq_length
    )
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}\n")

    # 评估
    metrics = evaluate_model(model, test_loader, backend)

    print(f"Evaluation Results:")
    print(f"  Loss:       {metrics['val_loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Tokens:     {metrics['total_tokens']:,}")

    # 评级
    ppl = metrics['perplexity']
    if ppl < 10:
        rating = "Excellent"
    elif ppl < 30:
        rating = "Good"
    elif ppl < 100:
        rating = "Fair"
    else:
        rating = "Poor"
    print(f"  Rating:     {rating}")


if __name__ == '__main__':
    main()
