import torch
import yaml
import random
import numpy as np
import os
import sys

from src.model import MiniLLM, Config as ModelConfig
from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader
from src.trainer import Trainer


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


class ConfigWrapper:
    def __init__(self, config_dict):
        self.model = config_dict['model']
        self.training = config_dict['training']
        self.data = config_dict['data']
        self.system = config_dict['system']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def main():
    config_dict = load_config()
    config = ConfigWrapper(config_dict)
    
    set_seed(config.system['seed'])
    
    device = torch.device(config.system['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    print("Building model...")
    model = MiniLLM(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    tokenizer, train_loader, val_loader = prepare_data(config)
    
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
    os.makedirs(config.system.log_dir, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    if os.path.exists(os.path.join(config.system.checkpoint_dir, 'best_model.pt')):
        print("\nLoading existing checkpoint...")
        trainer.load_checkpoint('best_model.pt')
    
    print("\n" + "="*50)
    print("Starting training!")
    print("="*50 + "\n")
    
    trainer.train()
    
    print("\nTraining finished! Running final evaluation...")
    final_val_loss = trainer.evaluate()
    print(f"Final validation loss: {final_val_loss:.4f}")


if __name__ == '__main__':
    main()
