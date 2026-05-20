import torch
import yaml
from src.model import MiniLLM, Config as ModelConfig


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_length=model.config.max_seq_length)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(next(model.parameters()).device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    generated_text = tokenizer.decode(output_ids[0].cpu().numpy())
    return generated_text


def interactive_mode(model, tokenizer):
    print("\n" + "="*50)
    print("Interactive Generation Mode")
    print("="*50)
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            generated = generate_text(model, tokenizer, prompt, max_new_tokens=100)
            print(f"Model: {generated}\n")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break


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
    if torch.cuda.is_available():
        checkpoint_path = 'checkpoints/best_model.pt'
    
    try:
        model = load_checkpoint(model, checkpoint_path, device)
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with untrained model (random weights)")
    
    from src.tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])
    
    interactive_mode(model, tokenizer)


if __name__ == '__main__':
    main()
