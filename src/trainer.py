import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import os
import math
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        os.makedirs(config.system.checkpoint_dir, exist_ok=True)
        os.makedirs(config.system.log_dir, exist_ok=True)
        self.writer = SummaryWriter(config.system.log_dir)
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.training.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
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
        def lr_lambda(current_step):
            if current_step < self.config.training.warmup_steps:
                return float(current_step) / float(max(1, self.config.training.warmup_steps))
            progress = float(current_step - self.config.training.warmup_steps) / float(max(1, self.config.training.max_steps - self.config.training.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        print(f"Starting training for {self.config.training.num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.training.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * self.config.training.gradient_accumulation_steps
                num_batches += 1
                
                if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({'loss': loss.item() * self.config.training.gradient_accumulation_steps, 'lr': f'{current_lr:.2e}'})
                    
                    self.writer.add_scalar('train/loss', loss.item() * self.config.training.gradient_accumulation_steps, self.global_step)
                    self.writer.add_scalar('train/lr', current_lr, self.global_step)
                
                self.global_step += 1
                
                if self.global_step % self.config.training.eval_steps == 0:
                    val_loss = self.evaluate()
                    self.model.train()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model.pt')
                        print(f"\nNew best model saved! Val loss: {val_loss:.4f}")
                
                if self.global_step >= self.config.training.max_steps:
                    print(f"\nReached max steps ({self.config.training.max_steps}). Stopping training.")
                    return
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average training loss: {avg_epoch_loss:.4f}")
            
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            val_loss = self.evaluate()
            print(f"Validation loss: {val_loss:.4f}")
        
        print("\nTraining completed!")
        self.writer.close()
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        
        return avg_loss
    
    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.config.system.checkpoint_dir, filename)
        torch.save({
            'epoch': self.current_epoch if hasattr(self, 'current_epoch') else 0,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.config.system.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
