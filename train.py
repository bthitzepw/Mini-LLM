"""
CodeSprite v2 训练入口 — 框架无关 IR 架构

用法:
  python train.py                  # 标准训练（自动选择设备）
  python train.py --mode auto      # 自动学习模式
  python train.py --find-lr        # 学习率查找器
  python train.py --no-amp         # 禁用混合精度
  python train.py --device cpu     # 强制 CPU 训练
  python train.py --no-cpu-fallback  # GPU 不可用时直接报错（不静默回退）
  python train.py --convert-old checkpoints/best_model.pt  # 转换旧权重

设备策略:
  训练：优先 GPU，自动回退 CPU（可通过 --no-cpu-fallback / 环境变量禁用回退）
  推理：默认 CPU，可按需切换 GPU
"""

import torch
import yaml
import random
import numpy as np
import os
import sys
import argparse

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir.config import ModelConfig
from ir.transformer import TransformerModel
from ir.layers import Layer
from backends.pytorch import PyTorchBackend, init_model_weights, collect_parameters
from training.trainer import Trainer
from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader
from src.device import resolve_device, print_device_info, warn_cpu_training


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ConfigWrapper:
    def __init__(self, config_dict):
        self.model = type('ModelConfig', (), config_dict['model'])()
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


def prepare_data(config_dict):
    print("Initializing tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])

    print(f"Loading datasets...")
    train_dataset = TextDataset(
        config_dict['data']['train_file'], tokenizer,
        config_dict['model']['max_seq_length']
    )
    val_dataset = TextDataset(
        config_dict['data']['val_file'], tokenizer,
        config_dict['model']['max_seq_length']
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config_dict['training']['batch_size'],
        shuffle=True,
        num_workers=config_dict['data']['num_workers']
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config_dict['training']['batch_size'],
        shuffle=False,
        num_workers=config_dict['data']['num_workers']
    )

    return tokenizer, train_loader, val_loader


def print_model_info(model, backend):
    """打印模型信息"""
    device_display = getattr(backend, '_resolved_device', 'unknown')
    print(f"\n{'='*50}")
    print(f"CodeSprite v2 — Framework-Agnostic IR Architecture")
    print(f"{'='*50}")
    print(f"  Total parameters:     {model.get_param_count():,} "
          f"({model.get_param_count()/1e6:.1f}M)")
    print(f"  Hidden size:          {model.config.hidden_size}")
    print(f"  Layers:               {model.config.num_layers}")
    print(f"  Heads:                {model.config.num_heads}")
    print(f"  KV Heads:             {model.config.num_kv_heads}")
    print(f"  Vocab size:           {model.config.vocab_size}")
    print(f"  Max sequence:         {model.config.max_seq_length}")
    print(f"  Activation:           {model.config.activation}")
    print(f"  RoPE:                 {model.config.use_rope}")
    print(f"  Backend:              {backend.name} ({device_display})")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='CodeSprite v2 Training')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'auto', 'find-lr'],
                       help='Training mode')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no-rope', action='store_true', help='Disable RoPE')
    parser.add_argument('--no-swiglu', action='store_true', help='Disable SwiGLU')
    parser.add_argument('--use-ema', action='store_true', help='Enable EMA')
    parser.add_argument('--label-smoothing', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None,
                       help='Device: auto (default), cuda, cpu')
    parser.add_argument('--no-cpu-fallback', action='store_true',
                       help='Abort if GPU unavailable (sets CODESPRITE_ALLOW_CPU_FALLBACK=false)')
    parser.add_argument('--convert-old', type=str, default=None,
                       help='Convert old checkpoint to new format and exit')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    args = parser.parse_args()

    config_dict = load_config()

    # 命令行覆盖
    if args.no_amp:
        config_dict['training']['use_amp'] = False
    if args.no_rope:
        config_dict['model']['use_rope'] = False
    if args.no_swiglu:
        config_dict['model']['activation'] = 'gelu'
    if args.use_ema:
        config_dict['training']['use_ema'] = True
    if args.label_smoothing is not None:
        config_dict['training']['label_smoothing'] = args.label_smoothing
    if args.lr is not None:
        config_dict['training']['learning_rate'] = args.lr
    if args.epochs is not None:
        config_dict['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config_dict['training']['batch_size'] = args.batch_size

    config = ConfigWrapper(config_dict)
    set_seed(config_dict['system']['seed'])

    # 设备选择 — 统一设备管理模块
    if args.no_cpu_fallback:
        os.environ["CODESPRITE_ALLOW_CPU_FALLBACK"] = "false"

    device_str = args.device or config_dict['system']['device']
    cpu_threads = config_dict['system'].get('cpu_threads', None)
    resolved = resolve_device(device_str, cpu_threads=cpu_threads)
    device = torch.device(resolved)

    # 可观测性日志
    print_device_info(resolved)

    # CPU 训练风险警告
    if resolved == "cpu":
        warn_cpu_training()
    elif resolved == "cuda":
        print(f"  GPU: {torch._C._cuda_getDeviceProperties(0).name if hasattr(torch._C, '_cuda_getDeviceProperties') else 'available'}\n")

    # 构建 IR 模型
    mc = ModelConfig.from_yaml(config_dict)
    model = TransformerModel(mc)

    # 创建 PyTorch 后端（设备已在 resolve_device 中确定）
    backend = PyTorchBackend(device=resolved)

    # 初始化权重
    init_model_weights(model, backend)
    print_model_info(model, backend)

    # 旧权重转换模式
    if args.convert_old:
        from tools.convert_checkpoint import convert_old_to_new
        convert_old_to_new(args.convert_old, args.convert_old.replace('.pt', '_v2.pt'))
        return

    # 加载已有检查点
    best_path = os.path.join(config_dict['system']['checkpoint_dir'], 'best_model.pt')
    resume_path = args.resume or (best_path if os.path.exists(best_path) else None)

    if resume_path and os.path.exists(resume_path):
        print(f"\nLoading checkpoint: {resume_path}")
        backend.load_checkpoint(model, resume_path)

    # 准备数据
    tokenizer, train_loader, val_loader = prepare_data(config_dict)

    os.makedirs(config_dict['system']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config_dict['system']['log_dir'], exist_ok=True)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        backend=backend,
        config=config,
        tokenizer=tokenizer
    )

    print("\n" + "="*50)
    print(f"Starting {args.mode} training!")
    print(f"  AMP: {config_dict['training']['use_amp']}")
    print(f"  Label Smoothing: {config_dict['training'].get('label_smoothing', 0)}")
    print(f"  EMA: {config_dict['training'].get('use_ema', False)}")
    print(f"  LR: {config_dict['training']['learning_rate']}")
    print(f"  Epochs: {config_dict['training']['num_epochs']}")
    print("="*50 + "\n")

    trainer.train()

    print("\nTraining finished! Running final evaluation...")
    val_metrics = trainer.evaluate()
    print(f"Final validation loss: {val_metrics['val_loss']:.4f}")
    print(f"Final perplexity: {val_metrics['perplexity']:.2f}")


if __name__ == '__main__':
    main()
