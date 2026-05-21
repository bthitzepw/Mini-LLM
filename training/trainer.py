"""
训练器 — 后端无关的训练循环

设计原则：
  - 不 import torch（通过 backend 间接使用）
  - 不关心模型结构（只调用 model.forward(x, backend)）
  - 支持 AMP、EMA、标签平滑、早停、梯度累积
"""

import os
import math
import json
import time
from typing import Dict, Any, Optional


class Trainer:
    """
    后端无关的训练器

    用法:
        from backends.pytorch import PyTorchBackend, init_model_weights
        from training import Trainer

        backend = PyTorchBackend(device="cuda")
        init_model_weights(model, backend)
        trainer = Trainer(model, train_loader, val_loader, backend, config)
        trainer.train()
    """

    def __init__(self, model, train_loader, val_loader, backend, config,
                 tokenizer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.backend = backend
        self.config = config
        self.tokenizer = tokenizer

        # 训练配置
        self.num_epochs = config.training.num_epochs
        self.learning_rate = config.training.learning_rate
        self.batch_size = config.training.batch_size
        self.warmup_steps = config.training.warmup_steps
        self.max_steps = config.training.max_steps
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.max_grad_norm = config.training.max_grad_norm
        self.weight_decay = config.training.weight_decay
        self.label_smoothing = getattr(config.training, 'label_smoothing', 0.0)
        self.use_amp = getattr(config.training, 'use_amp', False)
        self.use_ema = getattr(config.training, 'use_ema', False)
        self.ema_decay = getattr(config.training, 'ema_decay', 0.999)
        self.early_stopping_patience = getattr(config.training, 'early_stopping_patience', 5)

        # 检查点目录
        self.checkpoint_dir = config.system.checkpoint_dir
        self.log_dir = config.system.log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 创建优化器
        from backends.pytorch import collect_parameters
        named_params = collect_parameters(model)
        self.optimizer = self.backend.create_optimizer(
            [(n, p) for n, p in named_params],
            lr=self.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_epsilon,
            weight_decay=self.weight_decay
        )

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.scaler = None
        if self.use_amp and hasattr(self.backend, 'device'):
            import torch
            self.scaler = torch.amp.GradScaler(self.backend.device.type)

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_perplexity = float('inf')
        self.patience_counter = 0
        self.training_history = []
        self.start_time = None

        # EMA (简化版)
        self.ema_shadow = {} if self.use_ema else None

    def _create_scheduler(self):
        """Cosine Annealing with Warmup"""
        import torch

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """主训练循环"""
        self.start_time = time.time()
        print(f"\n{'='*50}")
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"  Backend: {self.backend.name}")
        print(f"  Device: {getattr(self.backend, 'device', 'cpu')}")
        print(f"  Parameters: {self.model.get_param_count():,}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Label Smoothing: {self.label_smoothing}")
        print(f"  Gradient Accumulation: {self.grad_accum_steps} steps")
        print(f"{'='*50}\n")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(epoch)
            val_metrics = self.evaluate()

            # 记录
            record = {
                'epoch': epoch + 1,
                'train_loss': epoch_metrics['avg_loss'],
                'val_loss': val_metrics['val_loss'],
                'perplexity': val_metrics['perplexity'],
                'lr': self.scheduler.get_last_lr()[0],
                'epoch_time': epoch_metrics['time'],
            }
            self.training_history.append(record)

            total_time = time.time() - self.start_time
            print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {epoch_metrics['avg_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            print(f"  Epoch Time: {epoch_metrics['time']:.1f}s")

            # 保存检查点
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            # 检查最佳模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_perplexity = val_metrics['perplexity']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  >> New best model! Val Loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")

            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping! No improvement for {self.early_stopping_patience} epochs.")
                break

            if self.global_step >= self.max_steps:
                print(f"\nReached max steps ({self.max_steps}). Stopping.")
                break

        self._save_history()
        print("\nTraining completed!")

    def _train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()
        import torch

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.backend.device)
            labels = batch['labels'].to(self.backend.device)

            # 前向传播
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(self.backend.device.type):
                    logits = self.model.forward(input_ids, self.backend)
                    loss = self._compute_loss(logits, labels)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model.forward(input_ids, self.backend)
                loss = self._compute_loss(logits, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            # 梯度累积步
            if (self.global_step + 1) % self.grad_accum_steps == 0:
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for _, p in self._get_params()], self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for _, p in self._get_params()], self.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            if self.global_step >= self.max_steps:
                break

        return {
            'avg_loss': total_loss / max(num_batches, 1),
            'time': time.time() - epoch_start
        }

    def _compute_loss(self, logits, labels):
        """计算交叉熵损失"""
        import torch

        # shift: 预测位置 t 的 logits，目标位置 t+1 的 labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]

        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing
        )

    def _get_params(self):
        """获取所有可训练参数"""
        from backends.pytorch import collect_parameters
        return collect_parameters(self.model)

    def evaluate(self):
        """评估模型"""
        import torch

        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.backend.device)
                labels = batch['labels'].to(self.backend.device)

                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast(self.backend.device.type):
                        logits = self.model.forward(input_ids, self.backend)
                else:
                    logits = self.model.forward(input_ids, self.backend)

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

    def save_checkpoint(self, filename: str):
        """保存训练检查点"""
        import torch
        path = os.path.join(self.checkpoint_dir, filename)
        state_dict = self.backend.get_state_dict(self.model)
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_perplexity': self.best_perplexity,
            'training_history': self.training_history,
        }, path)

    def load_checkpoint(self, filename: str):
        """加载训练检查点"""
        import torch
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.backend.device, weights_only=False)
        self.backend.load_state_dict(self.model, checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_perplexity = checkpoint.get('best_perplexity', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        print(f"Checkpoint loaded: epoch={checkpoint.get('epoch', 0)}, step={self.global_step}")

    def _save_history(self):
        """保存训练历史"""
        path = os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
