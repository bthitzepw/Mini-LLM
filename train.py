"""
Mini LLM 训练入口 - 深度学习增强版

支持:
  - 标准训练
  - 增量训练（从已有检查点继续）
  - 自动学习模式（从用户反馈数据中学习）
  - 学习率查找器
  - 混合精度训练
  - 标签平滑
  - EMA
  - 早停

用法:
  python train.py                  # 标准训练
  python train.py --mode auto      # 自动学习模式
  python train.py --find-lr        # 学习率查找
  python train.py --no-amp         # 禁用混合精度
"""

import torch
import yaml
import random
import numpy as np
import os
import sys
import argparse
import math

from src.model import MiniLLM, Config as ModelConfig
from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader
from src.trainer import Trainer


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


class ConfigWrapper:
    def __init__(self, config_dict):
        self.model = config_dict['model']
        self.training = type('TrainingConfig', (), config_dict['training'])()
        self.data = type('DataConfig', (), config_dict['data'])()
        self.system = type('SystemConfig', (), config_dict['system'])()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_data(config):
    print("Initializing tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config.model['vocab_size'])

    print(f"Loading datasets...")
    train_dataset = TextDataset(config.data['train_file'], tokenizer, config.model['max_seq_length'])
    val_dataset = TextDataset(config.data['val_file'], tokenizer, config.model['max_seq_length'])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=config.data['num_workers']
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=config.data['num_workers']
    )

    return tokenizer, train_loader, val_loader


def print_model_summary(model, device):
    """打印模型摘要"""
    info = model.get_model_info()
    print(f"\n{'='*50}")
    print(f"Model Summary")
    print(f"{'='*50}")
    print(f"  Total parameters:     {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    print(f"  Embedding parameters: {info['embedding_params']:,}")
    print(f"  Attention parameters: {info['attention_params']:,}")
    print(f"  FFN parameters:       {info['ffn_params']:,}")
    print(f"  Layers:               {info['num_layers']}")
    print(f"  Hidden size:          {info['hidden_size']}")
    print(f"  Heads:                {info['num_heads']}")
    print(f"  Vocab size:           {info['vocab_size']}")
    print(f"  Max sequence:         {info['max_seq_length']}")
    print(f"  RoPE:                 {info['use_rope']}")
    print(f"  Gradient Checkpoint:  {info['use_gradient_checkpointing']}")
    print(f"  Tie weights:          {info['tie_weights']}")
    print(f"  Device:               {device}")

    # 显存估算
    if device.type == 'cuda':
        mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
        mem_grads = mem_params  # 梯度大小约等于参数大小
        mem_opt = mem_params * 2  # AdamW 约为参数的2倍
        total_mem = (mem_params + mem_grads + mem_opt) / (1024 ** 3)
        print(f"\n  Estimated GPU memory: ~{total_mem:.1f} GB (params+grads+optimizer)")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Mini LLM Training')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'auto', 'find-lr'],
                       help='Training mode: standard, auto (from feedback), or find-lr')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no-rope', action='store_true', help='Disable RoPE')
    parser.add_argument('--no-swiglu', action='store_true', help='Disable SwiGLU')
    parser.add_argument('--use-ema', action='store_true', help='Enable EMA')
    parser.add_argument('--use-checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--label-smoothing', type=float, default=None, help='Label smoothing value')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()

    config_dict = load_config()

    # 命令行参数覆盖配置
    if args.no_amp:
        config_dict['training']['use_amp'] = False
    if args.no_rope:
        config_dict['model']['use_rope'] = False
    if args.no_swiglu:
        config_dict['model']['use_swiglu'] = False
    if args.use_ema:
        config_dict['training']['use_ema'] = True
    if args.use_checkpointing:
        config_dict['training']['use_gradient_checkpointing'] = True
    if args.label_smoothing is not None:
        config_dict['training']['label_smoothing'] = args.label_smoothing
    if args.lr is not None:
        config_dict['training']['learning_rate'] = args.lr
    if args.epochs is not None:
        config_dict['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config_dict['training']['batch_size'] = args.batch_size

    config = ConfigWrapper(config_dict)
    set_seed(config.system['seed'])

    device = torch.device(config.system['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # 构建模型
    model_config = ModelConfig(
        vocab_size=config.model['vocab_size'],
        hidden_size=config.model['hidden_size'],
        num_layers=config.model['num_layers'],
        num_heads=config.model['num_heads'],
        intermediate_size=config.model['intermediate_size'],
        dropout=config.model['dropout'],
        max_seq_length=config.model['max_seq_length'],
        tie_weights=config.model['tie_weights']
    )

    use_rope = config.model.get('use_rope', True)
    use_swiglu = config.model.get('use_swiglu', True)
    use_gradient_ckpt = getattr(config.training, 'use_gradient_checkpointing', False)

    print(f"\nBuilding model (RoPE={use_rope}, SwiGLU={use_swiglu})...")
    model = MiniLLM(model_config, use_rope=use_rope, use_swiglu=use_swiglu,
                    use_gradient_checkpointing=use_gradient_ckpt)

    print_model_summary(model, device)

    # 学习率查找器模式
    if args.mode == 'find-lr':
        print("\nRunning LR Finder...")
        tokenizer, train_loader, val_loader = prepare_data(config_dict)
        suggested_lr, lr_pairs = Trainer.find_learning_rate(
            model, train_loader, config, device
        )
        print(f"\nSuggested learning rate: {suggested_lr:.2e}")
        return

    # 准备数据
    tokenizer, train_loader, val_loader = prepare_data(config_dict)

    os.makedirs(config.system['checkpoint_dir'], exist_ok=True)
    os.makedirs(config.system['log_dir'], exist_ok=True)

    # 自动学习模式：合并用户反馈数据
    if args.mode == 'auto':
        print("\nAuto-learning mode: loading user feedback data...")
        try:
            from src.auto_learner import AutoLearner
            learner = AutoLearner()
            feedback_data = learner.prepare_training_data(include_augmented=True)
            if feedback_data:
                print(f"Found {len(feedback_data)} samples from user feedback.")

                # 合并到训练集
                class FeedbackDataset(torch.utils.data.Dataset):
                    def __init__(self, texts, tokenizer, max_length):
                        self.data = []
                        for text in texts:
                            if len(text) < 10:
                                continue
                            tokens = tokenizer.encode(text, max_length=max_length)
                            if len(tokens) < 5:
                                continue
                            self.data.append(tokens)
                    def __len__(self):
                        return len(self.data)
                    def __getitem__(self, idx):
                        tokens = self.data[idx]
                        input_ids = tokens[:-1]
                        labels = tokens[1:]
                        attention_mask = [1] * len(input_ids)
                        return {
                            'input_ids': torch.tensor(input_ids, dtype=torch.long),
                            'labels': torch.tensor(labels, dtype=torch.long),
                            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                        }

                feedback_dataset = FeedbackDataset(
                    feedback_data, tokenizer, config_dict['model']['max_seq_length']
                )
                # 合并数据集
                from torch.utils.data import ConcatDataset
                combined_dataset = ConcatDataset([train_loader.dataset, feedback_dataset])
                train_loader = torch.utils.data.DataLoader(
                    combined_dataset,
                    batch_size=config_dict['training']['batch_size'],
                    shuffle=True,
                    num_workers=0
                )
                print(f"Combined dataset size: {len(combined_dataset)}")

                # 自动学习模式使用更小的学习率
                if args.lr is None:
                    config_dict['training']['learning_rate'] = 0.00005
                    print("Auto-learning: using smaller learning rate (5e-5)")
        except Exception as e:
            print(f"Warning: Could not load feedback data: {e}")
            print("Falling back to standard training.")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 加载已有检查点
    checkpoint_path = os.path.join(config.system['checkpoint_dir'], 'best_model.pt')
    if os.path.exists(checkpoint_path):
        print("\nLoading existing checkpoint...")
        trainer.load_checkpoint('best_model.pt')

    print("\n" + "="*50)
    print(f"Starting {args.mode} training!")
    print(f"  Mode: {args.mode}")
    print(f"  AMP: {config_dict['training']['use_amp']}")
    print(f"  Label Smoothing: {config_dict['training'].get('label_smoothing', 0)}")
    print(f"  EMA: {config_dict['training'].get('use_ema', False)}")
    print(f"  LR: {config_dict['training']['learning_rate']}")
    print(f"  Epochs: {config_dict['training']['num_epochs']}")
    print("="*50 + "\n")

    trainer.train()

    print("\nTraining finished! Running final evaluation...")
    final_val_loss = trainer.evaluate()
    print(f"Final validation loss: {final_val_loss['val_loss']:.4f}")
    print(f"Final perplexity: {final_val_loss['perplexity']:.2f}")


if __name__ == '__main__':
    main()
