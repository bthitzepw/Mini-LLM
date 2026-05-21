"""
MiniLLM v2 Web 应用 — 框架无关架构版

安全加固:
  - 安全响应头（CSP/HSTS/X-Frame-Options等）
  - 请求速率限制
  - 输入校验与长度限制
  - 审计日志记录

API端点:
  - POST /api/generate        - 文本生成
  - GET  /api/info            - 模型信息
  - GET  /api/health          - 健康检查
  - POST /api/feedback        - 用户反馈
  - GET  /api/learning-status - 学习状态
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from collections import defaultdict

from flask import Flask, request, jsonify, render_template, make_response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir.config import ModelConfig
from ir.transformer import TransformerModel
from inference.engine import InferenceEngine
from src.tokenizer import SimpleTokenizer
from src.compliance import SecurityHeaders, RateLimiter, AuditLogger

# --- 初始化 ---

app = Flask(__name__)

# 安全配置: Flask secret_key（用于 session 签名等）
# 生产环境部署前请替换为随机生成的强密钥:
#   python -c "import secrets; print(secrets.token_hex(32))"
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'minillm-dev-key-change-in-production')

# 合规组件
security_headers = SecurityHeaders()
rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
audit_logger = AuditLogger(log_dir="logs")

# 加载配置
def load_config():
    import yaml
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config_dict = load_config()
model_config = ModelConfig.from_yaml(config_dict)

# 加载模型
tokenizer = SimpleTokenizer(vocab_size=model_config.vocab_size)
model = TransformerModel(model_config)
engine = InferenceEngine(
    model,
    checkpoint_path='checkpoints/best_model.pt',
    tokenizer=tokenizer,
    device='cpu'
)
engine.temperature = 0.8
engine.top_k = 50
engine.top_p = 0.9

print(f"MiniLLM v2 Web App ready")
print(f"  Backend: {engine.backend.name}")
print(f"  Parameters: {model.get_param_count():,}")

# --- 安全装饰器 ---

@app.before_request
def before_request():
    """请求前处理：速率限制"""
    client_ip = request.remote_addr
    if not rate_limiter.check(client_ip):
        audit_logger.log(client_ip, "RATE_LIMITED", request.path)
        response = jsonify({"error": "请求过于频繁，请稍后再试"})
        response.status_code = 429
        return response


@app.after_request
def after_request(response):
    """响应后处理：添加安全头"""
    return security_headers.apply(response)


# --- API 端点 ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """文本生成 API"""
    data = request.get_json(silent=True) or {}
    prompt = data.get('prompt', '').strip()
    max_tokens = min(data.get('max_tokens', 100), 500)
    temperature = data.get('temperature', engine.temperature)
    top_k = data.get('top_k', engine.top_k)
    top_p = data.get('top_p', engine.top_p)

    if not prompt:
        return jsonify({"error": "prompt 不能为空"}), 400
    if len(prompt) > 2000:
        return jsonify({"error": "prompt 长度不能超过 2000 字符"}), 400

    start_time = time.time()
    client_ip = request.remote_addr

    try:
        # 临时设置采样参数
        orig_temp = engine.temperature
        orig_topk = engine.top_k
        orig_topp = engine.top_p
        engine.temperature = temperature
        engine.top_k = top_k
        engine.top_p = top_p

        generated = engine.generate(prompt, max_new_tokens=max_tokens)

        # 恢复
        engine.temperature = orig_temp
        engine.top_k = orig_topk
        engine.top_p = orig_topp

        elapsed = time.time() - start_time
        audit_logger.log(client_ip, "GENERATE", f"len={len(prompt)}, tokens={max_tokens}, time={elapsed:.2f}s")

        return jsonify({
            "text": generated,
            "tokens_generated": len(tokenizer.encode(generated)) - len(tokenizer.encode(prompt)),
            "elapsed_seconds": round(elapsed, 2),
            "backend": engine.backend.name,
        })

    except Exception as e:
        logging.error(f"Generation error: {e}")
        return jsonify({"error": f"生成失败: {str(e)}"}), 500


@app.route('/api/info', methods=['GET'])
def model_info():
    """模型信息"""
    return jsonify(engine.info())


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "backend": engine.backend.name,
        "parameters": model.get_param_count(),
    })


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """用户反馈"""
    data = request.get_json(silent=True) or {}
    prompt = data.get('prompt', '')
    response_text = data.get('response', '')
    rating = data.get('rating', '')  # 'up' or 'down'

    if not prompt or not rating:
        return jsonify({"error": "缺少必要字段"}), 400

    # 记录反馈
    audit_logger.log(
        request.remote_addr, "FEEDBACK",
        f"rating={rating}, prompt_len={len(prompt)}, response_len={len(response_text)}"
    )

    return jsonify({"status": "ok", "message": "反馈已记录"})


@app.route('/api/learning-status', methods=['GET'])
def learning_status():
    """学习状态"""
    return jsonify({
        "auto_learning_enabled": False,
        "total_feedback": 0,
        "message": "自动学习功能在 v2 架构中暂时禁用"
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# --- 启动 ---

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  MiniLLM v2 Web Server")
    print("  Architecture: Framework-Agnostic IR")
    print(f"  Backend: {engine.backend.name}")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
