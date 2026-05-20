from flask import Flask, render_template, request, jsonify
import torch
import yaml
import os
from datetime import datetime

from src.model import MiniLLM, Config as ModelConfig
from src.tokenizer import SimpleTokenizer
from src.moderator import moderator

app = Flask(__name__)

model = None
tokenizer = None
device = None


def load_model():
    global model, tokenizer, device
    
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
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    else:
        print(f"Warning: Checkpoint not found, using untrained model")
    
    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])


def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    if model is None or tokenizer is None:
        return "模型未加载，请先启动服务。"
    
    # 审核用户输入
    input_check = moderator.check_content(prompt)
    if not input_check['is_safe']:
        return "⚠️ 您的输入包含违规内容，请修改后重试。"
    
    model.eval()
    input_ids = tokenizer.encode(prompt, max_length=model.config.max_seq_length)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    generated_text = tokenizer.decode(output_ids[0].cpu().numpy())
    
    # 审核生成内容
    output_check = moderator.check_content(generated_text)
    if not output_check['is_safe']:
        filtered_text = moderator.filter_response(generated_text)
        generated_text = filtered_text
    
    # 添加AI生成标识
    ai_disclaimer = "\n\n---\n🤖 以上内容由人工智能生成，仅供参考。"
    generated_text += ai_disclaimer
    
    return generated_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/agreement')
def agreement():
    return render_template('agreement.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 100)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 50)
    
    try:
        result = generate_text(prompt, max_new_tokens, temperature, top_k)
        return jsonify({'success': True, 'text': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/info', methods=['GET'])
def api_info():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    return jsonify({
        'success': True,
        'model_name': 'Mini LLM',
        'params': f"{sum(p.numel() for p in model.parameters()):,}",
        'layers': model.config.num_layers,
        'hidden_size': model.config.hidden_size,
        'device': str(device)
    })


if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("Mini LLM Web Server Starting!")
    print("="*60)
    print("Open browser: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
