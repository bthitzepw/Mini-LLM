"""
CodeSprite 训练器 - 框架无关 IR 架构

训练增强:
  1. 混合精度训练 (AMP) - FP16/FP32混合计算，加速训练+节省显存
  2. 标签平滑 (Label Smoothing) - 防止过拟合，提升泛化能力
  3. Perplexity 指标实时追踪 - 直观评估语言模型质量
  4. 学习率查找器 (LR Finder) - 帮助找到最优学习率
  5. 梯度累积 + 梯度裁剪 - 支持大batch训练
  6. Cosine Annealing + Warmup - 成熟的LR调度策略
  7. EMA (指数移动平均) - 训练稳定性提升
  8. 早停机制 (Early Stopping) - 防止过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import os
import math
import json
import time
from datetime import datetime
from tqdm import tqdm


class EMAModel:
    """指数移动平均模型权重，提升推理稳定性"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新 EMA 权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """临时应用 EMA 权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


class Trainer:
    """
    增强训练器

    支持:
    - 混合精度训练 (AMP)
    - 标签平滑
    - EMA
    - 早停
    - Perplexity 追踪
    """

    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 训练配置
        self.use_amp = getattr(config.training, 'use_amp', False) if hasattr(config.training, 'use_amp') else False
        self.label_smoothing = getattr(config.training, 'label_smoothing', 0.0) if hasattr(config.training, 'label_smoothing') else 0.0
        self.use_ema = getattr(config.training, 'use_ema', False) if hasattr(config.training, 'use_ema') else False
        self.ema_decay = getattr(config.training, 'ema_decay', 0.999) if hasattr(config.training, 'ema_decay') else 0.999
        self.early_stopping_patience = getattr(config.training, 'early_stopping_patience', 5) if hasattr(config.training, 'early_stopping_patience') else 5
        self.save_total_limit = getattr(config.training, 'save_total_limit', 3) if hasattr(config.training, 'save_total_limit') else 3

        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 损失函数（带标签平滑）
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing
        )

        # 混合精度缩放器
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # EMA
        self.ema = EMAModel(self.model, decay=self.ema_decay) if self.use_ema else None

        # 日志
        os.makedirs(config.system.checkpoint_dir, exist_ok=True)
        os.makedirs(config.system.log_dir, exist_ok=True)
        self.writer = SummaryWriter(config.system.log_dir)

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_perplexity = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        self.training_history = []
        self.start_time = None

    def _create_optimizer(self):
        """创建 AdamW 优化器（参数分组权重衰减）"""
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.training.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )

    def _create_scheduler(self):
        """创建 Cosine Annealing with Warmup 学习率调度器"""
        def lr_lambda(current_step):
            if current_step < self.config.training.warmup_steps:
                return float(current_step) / float(max(1, self.config.training.warmup_steps))
            progress = float(current_step - self.config.training.warmup_steps) / \
                      float(max(1, self.config.training.max_steps - self.config.training.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """主训练循环"""
        self.start_time = time.time()

        print(f"Starting training for {self.config.training.num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training features:")
        print(f"  - Mixed Precision (AMP): {self.use_amp}")
        print(f"  - Label Smoothing: {self.label_smoothing}")
        print(f"  - EMA: {self.use_ema} (decay={self.ema_decay})")
        print(f"  - Early Stopping: patience={self.early_stopping_patience}")
        print(f"  - Gradient Accumulation: {self.config.training.gradient_accumulation_steps} steps")
        print(f"  - Device: {self.device}")

        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(epoch)
            val_metrics = self.evaluate()

            # EMA 权重评估
            if self.use_ema:
                self.ema.apply_shadow()
                ema_metrics = self.evaluate()
                self.ema.restore()
                print(f"  EMA Val Loss: {ema_metrics['val_loss']:.4f}, "
                      f"EMA Perplexity: {ema_metrics['perplexity']:.2f}")

            # 记录训练历史
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': epoch_metrics['avg_loss'],
                'val_loss': val_metrics['val_loss'],
                'perplexity': val_metrics['perplexity'],
                'lr': self.scheduler.get_last_lr()[0],
                'epoch_time': epoch_metrics['time'],
            }
            self.training_history.append(epoch_record)

            # 打印总结
            total_time = time.time() - self.start_time
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs} Summary:")
            print(f"  Train Loss: {epoch_metrics['avg_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            print(f"  Epoch Time: {epoch_metrics['time']:.1f}s")
            print(f"  Total Time: {total_time:.1f}s")

            # 保存检查点
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            # 检查是否为最佳模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_perplexity = val_metrics['perplexity']
                self.patience_counter = 0

                # 保存最佳模型（使用EMA权重）
                if self.use_ema:
                    self.ema.apply_shadow()
                self.save_checkpoint('best_model.pt')
                if self.use_ema:
                    self.ema.restore()
                print(f"  >> New best model! Val Loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")

            # 早停检查
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                break

            if self.global_step >= self.config.training.max_steps:
                print(f"\nReached max steps ({self.config.training.max_steps}). Stopping.")
                break

        # 保存训练历史
        self._save_training_history()
        print("\nTraining completed!")
        self.writer.close()

    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            metrics = self._train_step(batch)
            total_loss += metrics['loss']
            num_batches += 1

            # 更新进度条
            if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f'{current_lr:.2e}'
                })

            # 定期评估
            if self.global_step % self.config.training.eval_steps == 0 and self.global_step > 0:
                val_metrics = self.evaluate()
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    if self.use_ema:
                        self.ema.apply_shadow()
                    self.save_checkpoint('best_model.pt')
                    if self.use_ema:
                        self.ema.restore()

                self.model.train()

            if self.global_step >= self.config.training.max_steps:
                break

        return {
            'avg_loss': total_loss / max(num_batches, 1),
            'time': time.time() - epoch_start
        }

    def _train_step(self, batch):
        """单步训练（支持混合精度）"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        if self.use_amp and self.scaler is not None:
            # 混合精度前向传播
            with torch.amp.autocast('cuda'):
                outputs = self.model(input_ids, attention_mask)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # 对齐长度
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss / self.config.training.gradient_accumulation_steps

            # 混合精度反向传播
            self.scaler.scale(loss).backward()

            if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()
        else:
            # 标准精度
            outputs = self.model(input_ids, attention_mask)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()

            if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

        raw_loss = loss.item() * self.config.training.gradient_accumulation_steps

        # 记录到 TensorBoard
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            self.writer.add_scalar('train/loss', raw_loss, self.global_step)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

            # 记录梯度范数（用于监控训练健康度）
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)

        self.global_step += 1

        return {'loss': raw_loss}

    @torch.no_grad()
    def evaluate(self):
        """评估模型（计算 Loss 和 Perplexity）"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0

        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)

        use_amp = self.use_amp and self.device.type == 'cuda'

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
            else:
                outputs = self.model(input_ids, attention_mask)

            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss += loss.item()

            valid_tokens = (shift_labels != -100).sum().item()
            total_tokens += valid_tokens
            num_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))  # 防止溢出

        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        self.writer.add_scalar('val/perplexity', perplexity, self.global_step)

        return {
            'val_loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
            'num_batches': num_batches
        }

    def save_checkpoint(self, filename):
        """保存训练检查点"""
        checkpoint_path = os.path.join(self.config.system.checkpoint_dir, filename)
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_perplexity': self.best_perplexity,
            'training_history': self.training_history,
            'config': {
                'use_amp': self.use_amp,
                'label_smoothing': self.label_smoothing,
                'use_ema': self.use_ema,
            }
        }, checkpoint_path)

        # 清理旧检查点（保留最近N个epoch检查点）
        if filename.startswith('checkpoint_epoch_') and self.save_total_limit > 0:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最近N个"""
        import glob
        checkpoint_dir = self.config.system.checkpoint_dir
        epoch_checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')),
            key=os.path.getmtime,
            reverse=True
        )
        # 删除超出限制的旧检查点
        for old_cp in epoch_checkpoints[self.save_total_limit:]:
            try:
                os.remove(old_cp)
            except OSError:
                pass

    def load_checkpoint(self, filename):
        """加载训练检查点"""
        checkpoint_path = os.path.join(self.config.system.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_perplexity = checkpoint.get('best_perplexity', float('inf'))
            self.training_history = checkpoint.get('training_history', [])
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"  Global step: {self.global_step}")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Best perplexity: {self.best_perplexity:.2f}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    def _save_training_history(self):
        """保存训练历史到JSON文件"""
        history_path = os.path.join(self.config.system.log_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)
        print(f"Training history saved to {history_path}")

    @staticmethod
    def find_learning_rate(model, train_loader, config, device='cuda',
                           lr_range=(1e-6, 1), num_iter=100):
        """
        学习率查找器 (LR Finder)

        快速遍历学习率范围，找到 loss 下降最快的学习率区间。

        Args:
            model: 待训练模型
            train_loader: 训练数据加载器
            config: 配置
            device: 计算设备
            lr_range: 学习率搜索范围
            num_iter: 搜索迭代次数

        Returns:
            suggested_lr: 建议的学习率
            lr_loss_pairs: [(lr, loss), ...]
        """
        print("Running LR Finder...")
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr_range[0])
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        lrs = []
        losses = []
        smooth_loss = float('inf')
        best_loss = float('inf')
        best_lr = lr_range[0]

        lr_factor = (lr_range[1] / lr_range[0]) ** (1 / num_iter)

        model.train()
        data_iter = iter(train_loader)

        for i in range(num_iter):
            # 获取下一个batch（循环）
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss.backward()
            optimizer.step()

            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr'] * lr_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # 平滑loss
            if smooth_loss == float('inf'):
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.98 * smooth_loss + 0.02 * loss.item()

            lrs.append(current_lr)
            losses.append(smooth_loss)

            if smooth_loss < best_loss:
                best_loss = smooth_loss
                best_lr = current_lr

            # 如果loss开始爆炸，停止
            if smooth_loss > 4 * best_loss:
                print(f"Loss exploded at lr={current_lr:.2e}, stopping.")
                break

        print(f"LR Finder complete. Suggested LR: {best_lr:.2e}")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Loss range: {min(losses):.4f} - {max(losses):.4f}")

        return best_lr, list(zip(lrs, losses))
