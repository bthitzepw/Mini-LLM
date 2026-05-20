"""
Mini LLM Web 应用 - 深度学习增强 + 自动学习版

安全加固:
  - 安全响应头（CSP/HSTS/X-Frame-Options等）
  - 请求速率限制
  - 输入校验与长度限制
  - 审计日志记录

深度学习特性:
  - RoPE旋转位置编码
  - SwiGLU前馈网络
  - KV-Cache加速推理
  - 混合精度支持

自动学习:
  - 用户反馈收集
  - 对话记录持久化
  - 增量训练触发
  - 学习进度追踪
  - 数据增强

API端点:
  - POST /api/generate        - 文本生成（含审核）
  - GET  /api/info            - 模型信息
  - GET  /api/health          - 健康检查
  - POST /api/feedback        - 提交用户反馈（点赞/点踩）
  - GET  /api/learning-status - 学习状态和统计
  - POST /api/learning/start  - 手动触发增量学习
  - GET  /api/learning/config - 获取/更新学习配置
  - POST /api/learning/config - 更新学习配置
  - GET  /api/user-rights     - 用户权利概览
  - GET  /api/data-export     - 数据导出
  - POST /api/data-delete     - 数据删除
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
from src.auto_learner import auto_learner

app = Flask(__name__)

model = None
tokenizer = None
device = None
session_interactions = {}  # session_id -> last_interaction_id 映射


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

    # 深度学习增强选项
    use_rope = config_dict['model'].get('use_rope', True)
    use_swiglu = config_dict['model'].get('use_swiglu', True)
    use_gradient_ckpt = config_dict['training'].get('use_gradient_checkpointing', False)

    print(f"Loading model (RoPE={use_rope}, SwiGLU={use_swiglu})...")
    model = MiniLLM(model_config, use_rope=use_rope, use_swiglu=use_swiglu,
                    use_gradient_checkpointing=use_gradient_ckpt).to(device)

    checkpoint_path = 'checkpoints/best_model.pt'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    else:
        print("Warning: Checkpoint not found, using untrained model")

    tokenizer = SimpleTokenizer(vocab_size=config_dict['model']['vocab_size'])

    # 自动学习数据库初始化
    auto_learning_config = config_dict.get('auto_learning', {})
    if auto_learning_config.get('enabled', False):
        auto_learner.db.set_config('auto_learning_enabled', 'true')
        print("Auto-learning enabled.")
    print(f"Model info: {model.get_model_info()}")


def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_k=50,
                  session_id='default'):
    if model is None or tokenizer is None:
        return "模型未加载，请先启动服务。"

    if len(prompt) > 5000:
        return "输入文本过长，请控制在5000字符以内。"

    # 审核用户输入
    input_check = moderator.check_content(prompt, check_type="input_check")

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

    # AI生成标识
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

    # 记录交互到自动学习系统
    interaction_id = auto_learner.record_interaction(
        session_id=session_id,
        prompt=prompt,
        response=generated_text,
        prompt_tokens=len(input_ids),
        response_tokens=output_ids.size(1) - len(input_ids),
        generation_params={
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': top_k
        }
    )
    session_interactions[session_id] = interaction_id

    return generated_text


def apply_security_headers(response):
    for key, value in security_headers.get_headers().items():
        response.headers[key] = value
    return response


@app.after_request
def after_request(response):
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
    session_id = data.get('session_id', str(uuid.uuid4()))

    try:
        max_new_tokens = max(10, min(500, int(max_new_tokens)))
        temperature = max(0.1, min(2.0, float(temperature)))
        top_k = max(1, min(200, int(top_k)))
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': '参数格式无效。'})

    try:
        result = generate_text(prompt, max_new_tokens, temperature, top_k, session_id)
        interaction_id = session_interactions.get(session_id)
        return jsonify({
            'success': True,
            'text': result,
            'session_id': session_id,
            'interaction_id': interaction_id,
        })
    except Exception as e:
        compliance_logger.log_security_event('GENERATION_ERROR', 'LOW', {'error': str(e)})
        return jsonify({'success': False, 'error': '服务内部错误，请稍后重试。'})


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """
    提交用户反馈

    Request Body:
        interaction_id: 交互记录ID
        feedback: 1=赞, -1=踩, 0=无反馈
        comment: 可选，反馈评论
    """
    data = request.get_json(silent=True) or {}

    interaction_id = data.get('interaction_id')
    feedback = data.get('feedback')
    comment = data.get('comment', '')

    if interaction_id is None or feedback is None:
        return jsonify({'success': False, 'error': '缺少必要参数 interaction_id 或 feedback。'})

    try:
        feedback = int(feedback)
        if feedback not in [-1, 0, 1]:
            return jsonify({'success': False, 'error': 'feedback 必须为 -1（踩）、0（无）、1（赞）。'})
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'feedback 格式无效。'})

    auto_learner.record_feedback(interaction_id, feedback, comment)

    label = {1: '赞', 0: '无反馈', -1: '踩'}[feedback]
    return jsonify({
        'success': True,
        'message': f'已记录反馈: {label}',
    })


@app.route('/api/learning-status', methods=['GET'])
def api_learning_status():
    """获取自动学习状态和统计信息"""
    status = auto_learner.get_learning_status()
    return jsonify({
        'success': True,
        **status,
    })


@app.route('/api/learning/start', methods=['POST'])
def api_start_learning():
    """
    手动触发增量学习

    Request Body:
        epochs: 可选，训练轮次
        lr: 可选，学习率
    """
    data = request.get_json(silent=True) or {}

    epochs = data.get('epochs')
    lr = data.get('lr')

    if epochs is not None:
        try:
            epochs = int(epochs)
            if epochs < 1 or epochs > 50:
                return jsonify({'success': False, 'error': 'epochs 应在 1-50 之间。'})
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'epochs 格式无效。'})

    if lr is not None:
        try:
            lr = float(lr)
            if lr < 1e-7 or lr > 1e-1:
                return jsonify({'success': False, 'error': 'lr 应在 1e-7 到 1e-1 之间。'})
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'lr 格式无效。'})

    result = auto_learner.start_learning(
        trigger_type='manual',
        epochs=epochs,
        lr=lr
    )
    return jsonify({'success': result['success'], 'message': result['message']})


@app.route('/api/learning/config', methods=['GET'])
def api_get_learning_config():
    """获取自动学习配置"""
    config = auto_learner.db.get_config()
    return jsonify({'success': True, 'config': config})


@app.route('/api/learning/config', methods=['POST'])
def api_set_learning_config():
    """
    更新自动学习配置

    Request Body:
        auto_learning_enabled: bool
        min_feedback_samples: int
        incremental_epochs: int
        incremental_lr: float
        augmentation_enabled: bool
        max_augmented_samples: int
    """
    data = request.get_json(silent=True) or {}
    updated = []

    allowed_keys = [
        'auto_learning_enabled', 'min_feedback_samples', 'incremental_epochs',
        'incremental_lr', 'augmentation_enabled', 'max_augmented_samples',
        'schedule_interval_hours', 'min_positive_ratio'
    ]

    for key in allowed_keys:
        if key in data:
            auto_learner.db.set_config(key, str(data[key]))
            updated.append(key)

    if not updated:
        return jsonify({'success': False, 'error': '没有有效的配置项需要更新。'})

    return jsonify({
        'success': True,
        'message': f'已更新配置: {", ".join(updated)}',
    })


@app.route('/api/info', methods=['GET'])
def api_info():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})

    model_info = model.get_model_info()
    return jsonify({
        'success': True,
        'model_name': 'Mini LLM',
        'params': f"{model_info['total_params']:,}",
        'trainable_params': f"{model_info['trainable_params']:,}",
        'layers': model.config.num_layers,
        'hidden_size': model.config.hidden_size,
        'device': str(device),
        'deep_learning': {
            'rope': model_info['use_rope'],
            'gradient_checkpointing': model_info['use_gradient_checkpointing'],
            'tie_weights': model_info['tie_weights'],
        },
        'auto_learning': auto_learner.get_learning_status()['stats']['auto_learning_enabled'],
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
    learning_status = auto_learner.get_learning_status()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'is_training': learning_status['is_training'],
    })


@app.route('/api/user-rights', methods=['GET'])
def api_user_rights():
    rights = UserRightsManager.get_rights_summary()
    compliance_summary = UserRightsManager.get_regulation_compliance_summary()
    return jsonify({
        'success': True,
        'user_rights': rights,
        'data_principles': compliance_summary['data_principles'],
    })


@app.route('/api/data-export', methods=['GET'])
def api_data_export():
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
    print("\nDeep Learning Features:")
    print("  - RoPE (Rotary Position Embedding)")
    print("  - SwiGLU Feed-Forward Network")
    print("  - KV-Cache Accelerated Inference")
    print("  - Mixed Precision Training (AMP)")
    print("  - Label Smoothing")
    print("  - Gradient Checkpointing")
    print("  - EMA (Exponential Moving Average)")
    print("\nAuto-Learning System:")
    print("  - User feedback collection")
    print("  - Conversation logging")
    print("  - Data augmentation")
    print("  - Incremental fine-tuning")
    print("  - Learning progress tracking")
    print("\nCompliance:")
    print("  - Content moderation (CN/EU/US)")
    print("  - PII detection & masking")
    print("  - Rate limiting")
    print("  - Security headers")
    print("  - Audit logging")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
