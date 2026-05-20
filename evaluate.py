import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.model import MiniLLM, Config as ModelConfig
from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader


def calculate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            
            valid_tokens = (shift_labels != -100).sum().item()
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss


def evaluate_model(model, dataloader, device):
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    perplexity, avg_loss = calculate_perplexity(model, dataloader, device)
    
    print(f"\nEvaluation Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    
    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open('config/config.yaml', 'r') as f:
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
    
    print("Loading model...")
    model = MiniLLM(model_config).to(device)
    
    checkpoint_path = 'checkpoints/best_model.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Evaluating untrained model (random weights)")
    
    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])
    
    test_dataset = TextDataset(config_dict['data']['test_file'], tokenizer, config_dict['model']['max_seq_length'])
    test_loader = create_dataloader(test_dataset, batch_size=config_dict['training']['batch_size'], shuffle=False)
    
    results = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)


if __name__ == '__main__':
    main()
