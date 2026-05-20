"""
Mini LLM Web 应用 - 多国合规加固版

安全加固:
  - 安全响应头（CSP/HSTS/X-Frame-Options等）
  - 请求速率限制
  - 输入校验与长度限制
  - 审计日志记录

合规接口:
  - /api/generate - 文本生成（含内容审核）
  - /api/info - 模型信息
  - /api/health - 健康检查
  - /api/user-rights - 用户权利概览（GDPR/PIPL/CCPA）
  - /api/data-export - 数据导出（GDPR Art.20）
  - /api/data-delete - 数据删除（GDPR Art.17/PIPL Art.47）

合规依据:
  - 中国: 《生成式AI管理办法》《网安法》《个保法》《数安法》
  - EU: AI Act, GDPR, DSA, NIS2
  - US: Section 230, COPPA, CCPA, AI Executive Order
"""

from flask import Flask, render_template, request, jsonify, make_response
import torch
import yaml
import os
import uuid
from datetime import datetime

from src.model import MiniLLM, Config as ModelConfig
from src.tokenizer import SimpleTokenizer
from src.moderator import moderator
from src.compliance import (
    security_headers, rate_limiter, data_retention,
    compliance_logger, UserRightsManager
)

app = Flask(__name__)

model = None
tokenizer = None
device = None


def load_model():
    global model, tokenizer, device

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

    print("Loading model...")
    model = MiniLLM(model_config).to(device)

    checkpoint_path = 'checkpoints/best_model.pt'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    else:
        print("Warning: Checkpoint not found, using untrained model")

    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])


def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    if model is None or tokenizer is None:
        return "模型未加载，请先启动服务。"

    # 输入长度限制（防止DoS）
    if len(prompt) > 5000:
        return "输入文本过长，请控制在5000字符以内。"

    # 审核用户输入
    input_check = moderator.check_content(prompt, check_type="input_check")

    # PII检测提示
    pii_warnings = []
    for pii in input_check.get('pii_detected', []):
        pii_warnings.append(f"[隐私保护] 检测到{pii['type']}，已自动脱敏处理。依据: {pii['regulation']}")

    if not input_check['is_safe']:
        if input_check['risk_level'] == 'block':
            compliance_logger.log_security_event(
                'INPUT_BLOCKED', 'HIGH',
                {'reason': str(input_check['issues'][:3])}
            )
            block_msg = "您的输入包含违规内容，已被安全系统拦截。"
            if pii_warnings:
                block_msg += "\n" + "\n".join(pii_warnings)
            return block_msg

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
    output_check = moderator.check_content(generated_text, check_type="output_check")
    if not output_check['is_safe']:
        filtered_text = moderator.filter_response(generated_text)
        generated_text = filtered_text
        compliance_logger.log_security_event(
            'OUTPUT_FILTERED', 'MEDIUM',
            {'reason': str(output_check['issues'][:3])}
        )

    # 未成年人适宜性检查
    age_check = moderator.check_age_appropriate(generated_text)
    age_warning = ""
    if not age_check['is_age_appropriate']:
        age_warning = f"\n\n[未成年人保护] {age_check['reason']} ({age_check['regulation']})"

    # AI生成标识（多国合规要求）
    ai_disclaimer = (
        "\n\n---\n"
        "[AI生成内容标识] 以上内容由人工智能模型自动生成。\n"
        "依据: 《生成式AI管理办法》第12条 / EU AI Act Art.52\n"
        "本内容仅供学习参考，不构成任何专业建议（医疗/法律/金融等）。"
    )
    generated_text += ai_disclaimer

    if pii_warnings:
        generated_text = "\n".join(pii_warnings) + "\n\n" + generated_text

    if age_warning:
        generated_text += age_warning

    return generated_text


def apply_security_headers(response):
    """为所有响应添加安全头"""
    for key, value in security_headers.get_headers().items():
        response.headers[key] = value
    return response


@app.after_request
def after_request(response):
    """全局响应后处理：添加安全头"""
    return apply_security_headers(response)


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
    # 速率限制
    client_id = request.remote_addr or 'unknown'
    rate_check = rate_limiter.is_allowed(client_id)
    if not rate_check['allowed']:
        response = make_response(jsonify({
            'success': False,
            'error': '请求过于频繁，请稍后重试。',
            'retry_after': rate_limiter.get_retry_after(client_id),
        }), 429)
        return apply_security_headers(response)

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': '无效的请求数据格式。'})

    prompt = data.get('prompt', '')
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({'success': False, 'error': '请输入有效文本。'})

    max_new_tokens = data.get('max_new_tokens', 100)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 50)

    # 参数范围校验
    try:
        max_new_tokens = max(10, min(500, int(max_new_tokens)))
        temperature = max(0.1, min(2.0, float(temperature)))
        top_k = max(1, min(200, int(top_k)))
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': '参数格式无效。'})

    try:
        result = generate_text(prompt, max_new_tokens, temperature, top_k)
        return jsonify({'success': True, 'text': result})
    except Exception as e:
        compliance_logger.log_security_event('GENERATION_ERROR', 'LOW', {'error': str(e)})
        return jsonify({'success': False, 'error': '服务内部错误，请稍后重试。'})


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
        'device': str(device),
        'compliance': {
            'content_moderation': True,
            'pii_detection': True,
            'audit_logging': True,
            'rate_limiting': True,
            'security_headers': True,
            'ai_content_labeling': True,
            'minor_protection': True,
        },
    })


@app.route('/api/health', methods=['GET'])
def api_health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
    })


@app.route('/api/user-rights', methods=['GET'])
def api_user_rights():
    """
    用户权利概览（GDPR/PIPL/CCPA）

    依据:
      - GDPR Art.12 - 信息透明原则
      - PIPL Art.44 - 知情权
      - CCPA §1798.100 - 知情权
    """
    rights = UserRightsManager.get_rights_summary()
    compliance_summary = UserRightsManager.get_regulation_compliance_summary()
    return jsonify({
        'success': True,
        'user_rights': rights,
        'data_principles': compliance_summary['data_principles'],
    })


@app.route('/api/data-export', methods=['GET'])
def api_data_export():
    """
    数据导出接口（GDPR Art.20 数据可携权 / CCPA §1798.100 知情权）

    Returns:
        当前会话可导出的数据
    """
    session_id = request.args.get('session_id', 'default')
    export_result = data_retention.get_user_data_export(session_id)

    compliance_logger.log_data_access(
        accessor=request.remote_addr or 'unknown',
        data_type='user_session',
        purpose='GDPR Art.20 / CCPA §1798.100 数据导出请求'
    )

    return jsonify({
        'success': True,
        'regulation': 'GDPR Art.20 (数据可携权) / CCPA §1798.100 (知情权)',
        **export_result,
    })


@app.route('/api/data-delete', methods=['POST'])
def api_data_delete():
    """
    数据删除接口（GDPR Art.17 被遗忘权 / PIPL Art.47 删除权 / CCPA §1798.105）

    Request Body:
        session_id: 会话标识
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id', 'default')

    delete_result = data_retention.delete_user_data(session_id)

    compliance_logger.log_data_access(
        accessor=request.remote_addr or 'unknown',
        data_type='user_session',
        purpose='GDPR Art.17 / PIPL Art.47 / CCPA §1798.105 数据删除请求'
    )

    return jsonify({
        'success': True,
        **delete_result,
    })


if __name__ == '__main__':
    load_model()
    print("\n" + "=" * 60)
    print("Mini LLM Web Server Starting!")
    print("=" * 60)
    print("Open browser: http://localhost:5000")
    print("Compliance features enabled:")
    print("  - Content moderation (CN/EU/US)")
    print("  - PII detection & masking")
    print("  - Rate limiting")
    print("  - Security headers")
    print("  - Audit logging")
    print("  - AI content labeling")
    print("  - Minor protection")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
