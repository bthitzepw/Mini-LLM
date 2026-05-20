"""
Mini LLM 评估脚本 - 深度学习增强版

评估指标:
  - Perplexity (困惑度) - 语言模型核心指标
  - Token-level Loss
  - 生成质量抽检
  - 模型信息报告
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import os
import sys
import time
from tqdm import tqdm

from src.model import MiniLLM, Config as ModelConfig
from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader


def calculate_perplexity(model, dataloader, device, use_amp=False):
    """计算 Perplexity（困惑度）"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids, attention_mask)

            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

            valid_tokens = (shift_labels != -100).sum().item()
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))

    return perplexity, avg_loss, total_tokens


def sample_generation(model, tokenizer, prompts, device, max_new_tokens=100):
    """对样本提示进行生成质量抽检"""
    results = []
    model.eval()

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, max_length=model.config.max_seq_length)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50
            )

        generated = tokenizer.decode(output_ids[0].cpu().numpy())
        results.append({
            'prompt': prompt[:50] + ('...' if len(prompt) > 50 else ''),
            'generated': generated[:200] + ('...' if len(generated) > 200 else ''),
            'output_length': output_ids.size(1) - len(input_ids)
        })

    return results


def evaluate_model(model, test_loader, device, tokenizer=None):
    """全面评估模型"""
    print("\n" + "="*50)
    print("Mini LLM Model Evaluation")
    print("="*50)

    # 基本信息
    info = model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"  Parameters:       {info['total_params']:,}")
    print(f"  Layers:           {info['num_layers']}")
    print(f"  Hidden Size:      {info['hidden_size']}")
    print(f"  Attention Heads:  {info['num_heads']}")
    print(f"  Vocab Size:       {info['vocab_size']}")
    print(f"  Max Sequence:     {info['max_seq_length']}")
    print(f"  RoPE:             {info['use_rope']}")
    print(f"  Device:           {device}")

    # Perplexity 评估
    print(f"\n{'─'*40}")
    print("Perplexity Evaluation:")
    print(f"{'─'*40}")

    use_amp = device.type == 'cuda'
    start_time = time.time()

    perplexity, avg_loss, total_tokens = calculate_perplexity(model, test_loader, device, use_amp)
    eval_time = time.time() - start_time

    print(f"\n  Results:")
    print(f"    Average Loss:    {avg_loss:.4f}")
    print(f"    Perplexity:      {perplexity:.2f}")
    print(f"    Total Tokens:    {total_tokens:,}")
    print(f"    Eval Time:       {eval_time:.1f}s")
    print(f"    Tokens/sec:      {total_tokens / max(eval_time, 0.01):,.0f}")

    # 质量评级
    if perplexity < 10:
        quality = "Excellent (PPL < 10)"
    elif perplexity < 30:
        quality = "Good (PPL < 30)"
    elif perplexity < 100:
        quality = "Fair (PPL < 100)"
    elif perplexity < 500:
        quality = "Poor (PPL < 500)"
    else:
        quality = "Untrained / Very Poor (PPL >= 500)"

    print(f"    Quality:         {quality}")

    # 生成质量抽检
    if tokenizer is not None:
        print(f"\n{'─'*40}")
        print("Generation Quality Check:")
        print(f"{'─'*40}")

        test_prompts = [
            "def fibonacci(n):",
            "# 快速排序",
            "function hello(",
            "class DataProcessor:",
            "SELECT * FROM",
            "// 二叉树遍历",
            "for i in range(",
            "public static void main",
        ]

        samples = sample_generation(model, tokenizer, test_prompts, device)
        for i, sample in enumerate(samples):
            print(f"\n  Sample {i+1}:")
            print(f"    Prompt: {sample['prompt']}")
            print(f"    Generated ({sample['output_length']} tokens): {sample['generated'][:150]}")

    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print("="*50)

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
        'quality': quality,
        'eval_time': eval_time,
        'model_info': info
    }


def main():
    import math
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    model_config = ModelConfig(
        vocab_size=config_dict['model']['vocab_size'],
        hidden_size=config_dict['model']['hidden_size'],
        num_layers=config_dict['model']['num_layers'],
        num_heads=config_dict['model']['num_heads'],
        intermediate_size=config_dict['model']['intermediate_size'],
        dropout=config_dict['model']['dropout'],
        max_seq_length=config_dict['model']['max_seq_length'],
        tie_weights=config_dict['model']['tie_weights']
    )

    use_rope = config_dict['model'].get('use_rope', True)
    use_swiglu = config_dict['model'].get('use_swiglu', True)

    print(f"Loading model (RoPE={use_rope}, SwiGLU={use_swiglu})...")
    model = MiniLLM(model_config, use_rope=use_rope, use_swiglu=use_swiglu).to(device)

    checkpoint_path = 'checkpoints/best_model.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        if 'best_val_loss' in checkpoint:
            print(f"  Best val loss from training: {checkpoint['best_val_loss']:.4f}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Evaluating untrained model (random weights)")

    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])

    test_dataset = TextDataset(config_dict['data']['test_file'], tokenizer, config_dict['model']['max_seq_length'])
    test_loader = create_dataloader(test_dataset, batch_size=config_dict['training']['batch_size'], shuffle=False)

    results = evaluate_model(model, test_loader, device, tokenizer)


if __name__ == '__main__':
    main()
