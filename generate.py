"""
Mini LLM 交互式生成 - 深度学习增强版

支持:
  - KV-Cache 加速推理
  - Top-P (nucleus) 采样
  - Temperature 采样
  - Top-K 采样
"""

import torch
import yaml
from src.model import MiniLLM, Config as ModelConfig


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=100,
                  temperature=0.8, top_k=50, top_p=None):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_length=model.config.max_seq_length)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_kv_cache=True
        )

    generated_text = tokenizer.decode(output_ids[0].cpu().numpy())
    return generated_text


def interactive_mode(model, tokenizer):
    print("\n" + "="*50)
    print("Interactive Generation Mode (Enhanced)")
    print("="*50)
    print("Commands:")
    print("  <text>     - Generate text")
    print("  :temp <n>  - Set temperature (0.1-2.0)")
    print("  :topk <n>  - Set top-k (1-200)")
    print("  :topp <n>  - Set top-p (0.0-1.0)")
    print("  :len <n>   - Set max length (10-500)")
    print("  :info      - Show model info")
    print("  quit/exit  - Exit")
    print("="*50 + "\n")

    temperature = 0.8
    top_k = 50
    top_p = None
    max_len = 100

    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt:
                continue

            # 命令模式
            if prompt.startswith(':'):
                parts = prompt.split()
                cmd = parts[0].lower()
                if cmd == ':temp' and len(parts) > 1:
                    temperature = max(0.1, min(2.0, float(parts[1])))
                    print(f"Temperature set to {temperature}")
                elif cmd == ':topk' and len(parts) > 1:
                    top_k = max(1, min(200, int(parts[1])))
                    print(f"Top-K set to {top_k}")
                elif cmd == ':topp' and len(parts) > 1:
                    val = float(parts[1])
                    top_p = val if val > 0 else None
                    print(f"Top-P set to {top_p}")
                elif cmd == ':len' and len(parts) > 1:
                    max_len = max(10, min(500, int(parts[1])))
                    print(f"Max length set to {max_len}")
                elif cmd == ':info':
                    info = model.get_model_info()
                    print(f"  Params: {info['total_params']:,}")
                    print(f"  RoPE: {info['use_rope']}")
                    print(f"  Layers: {info['num_layers']}")
                else:
                    print("Unknown command. Type 'quit' to exit.")
                continue

            generated = generate_text(model, tokenizer, prompt,
                                      max_new_tokens=max_len,
                                      temperature=temperature,
                                      top_k=top_k, top_p=top_p)
            print(f"Model: {generated}\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break


def main():
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
        model = load_checkpoint(model, checkpoint_path, device)
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running with untrained model (random weights)")

    info = model.get_model_info()
    print(f"  Parameters: {info['total_params']:,}")
    print(f"  RoPE: {info['use_rope']}")

    from src.tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])

    interactive_mode(model, tokenizer)


if __name__ == '__main__':
    main()
