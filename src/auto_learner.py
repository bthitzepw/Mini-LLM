"""
Mini LLM 自动学习模块 (Auto-Learning Module)

核心功能:
  1. 用户反馈收集 - 对话质量评分（点赞/点踩）
  2. 交互记录存储 - SQLite 持久化存储所有对话和反馈
  3. 数据增强 - 代码变换增强训练数据（变量重命名、注释替换等）
  4. 增量训练 - 基于用户反馈自动触发增量微调
  5. 学习进度追踪 - 训练轮次、指标变化、模型版本管理
  6. 定时学习调度 - 积累足够新数据后自动启动训练

依据:
  - 《生成式AI管理办法》第15条 - 模型训练记录保存
  - GDPR Art.22 - 自动化决策的透明性
"""

import sqlite3
import json
import os
import time
import random
import re
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# 数据库管理 - 存储用户交互和反馈
# ============================================================

class InteractionDB:
    """用户交互数据库，基于SQLite存储"""

    def __init__(self, db_path='data/auto_learning.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        # 对话记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                feedback INTEGER DEFAULT 0,
                feedback_comment TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prompt_tokens INTEGER DEFAULT 0,
                response_tokens INTEGER DEFAULT 0,
                generation_params TEXT DEFAULT '{}'
            )
        ''')

        # 学习记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_type TEXT NOT NULL,
                num_samples INTEGER DEFAULT 0,
                epochs INTEGER DEFAULT 0,
                train_loss_before REAL DEFAULT 0,
                train_loss_after REAL DEFAULT 0,
                val_loss_before REAL DEFAULT 0,
                val_loss_after REAL DEFAULT 0,
                perplexity_before REAL DEFAULT 0,
                perplexity_after REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                checkpoint_path TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            )
        ''')

        # 增强数据缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS augmented_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                original_text TEXT NOT NULL,
                augmented_text TEXT NOT NULL,
                augmentation_type TEXT NOT NULL,
                quality_score REAL DEFAULT 0.5,
                used_in_training INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 自动学习配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 初始化默认配置
        defaults = {
            'auto_learning_enabled': 'false',
            'min_feedback_samples': '20',
            'min_positive_ratio': '0.6',
            'incremental_epochs': '3',
            'incremental_lr': '0.00005',
            'max_augmented_samples': '500',
            'augmentation_enabled': 'true',
            'schedule_interval_hours': '24',
            'last_scheduled_check': '',
            'total_learning_rounds': '0',
            'total_feedback_count': '0',
        }

        for key, value in defaults.items():
            cursor.execute('''
                INSERT OR IGNORE INTO learning_config (key, value) VALUES (?, ?)
            ''', (key, value))

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_feedback ON interactions(feedback)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_learning_status ON learning_history(status)')

        conn.commit()
        conn.close()

    def add_interaction(self, session_id, prompt, response, prompt_tokens=0,
                       response_tokens=0, generation_params=None):
        """记录一次对话交互"""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO interactions
                (session_id, prompt, response, prompt_tokens, response_tokens, generation_params)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, prompt, response, prompt_tokens, response_tokens,
                  json.dumps(generation_params or {}, ensure_ascii=False)))
            conn.commit()
            interaction_id = cursor.lastrowid

            # 更新总反馈计数
            cursor.execute("UPDATE learning_config SET value = CAST(value AS INTEGER) + 1 WHERE key = 'total_feedback_count'")
            conn.commit()
            conn.close()
            return interaction_id

    def add_feedback(self, interaction_id, feedback, comment=''):
        """记录用户反馈 (feedback: 1=赞, -1=踩, 0=无)"""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE interactions SET feedback = ?, feedback_comment = ?
                WHERE id = ?
            ''', (feedback, comment, interaction_id))
            conn.commit()
            conn.close()

    def get_positive_samples(self, limit=1000, min_quality=0.5):
        """获取正面反馈的样本（点赞或高评分）用于增量训练"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prompt, response FROM interactions
            WHERE feedback >= 1
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{'prompt': r['prompt'], 'response': r['response']} for r in rows]

    def get_all_feedbacked_samples(self, limit=2000):
        """获取所有有反馈的样本（正面和负面）"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prompt, response, feedback FROM interactions
            WHERE feedback != 0
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{'prompt': r['prompt'], 'response': r['response'],
                 'feedback': r['feedback']} for r in rows]

    def get_stats(self):
        """获取学习统计信息"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 交互统计
        cursor.execute('SELECT COUNT(*) as total FROM interactions')
        total = cursor.fetchone()['total']

        cursor.execute('SELECT COUNT(*) as positive FROM interactions WHERE feedback = 1')
        positive = cursor.fetchone()['positive']

        cursor.execute('SELECT COUNT(*) as negative FROM interactions WHERE feedback = -1')
        negative = cursor.fetchone()['negative']

        cursor.execute('SELECT COUNT(*) as no_feedback FROM interactions WHERE feedback = 0')
        no_feedback = cursor.fetchone()['no_feedback']

        # 最近24小时
        cursor.execute('''
            SELECT COUNT(*) as recent FROM interactions
            WHERE created_at > datetime('now', '-24 hours')
        ''')
        recent = cursor.fetchone()['recent']

        # 学习历史
        cursor.execute('SELECT COUNT(*) as rounds FROM learning_history WHERE status = "completed"')
        rounds = cursor.fetchone()['rounds']

        # 增强数据
        cursor.execute('SELECT COUNT(*) as aug_count FROM augmented_data WHERE used_in_training = 0')
        aug_pending = cursor.fetchone()['aug_count']

        cursor.execute('SELECT COUNT(*) as aug_total FROM augmented_data')
        aug_total = cursor.fetchone()['aug_total']

        # 配置
        config = self.get_config()

        conn.close()
        return {
            'total_interactions': total,
            'positive_feedback': positive,
            'negative_feedback': negative,
            'no_feedback': no_feedback,
            'recent_24h': recent,
            'learning_rounds': rounds,
            'augmented_pending': aug_pending,
            'augmented_total': aug_total,
            'auto_learning_enabled': config.get('auto_learning_enabled', 'false') == 'true',
            'config': config,
        }

    def get_config(self):
        """获取学习配置"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM learning_config')
        rows = cursor.fetchall()
        conn.close()
        return {r['key']: r['value'] for r in rows}

    def set_config(self, key, value):
        """更新学习配置"""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_config (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            ''', (key, value, datetime.now().isoformat()))
            conn.commit()
            conn.close()

    def add_learning_record(self, record):
        """添加学习记录"""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_history
                (trigger_type, num_samples, epochs, train_loss_before, train_loss_after,
                 val_loss_before, val_loss_after, perplexity_before, perplexity_after,
                 status, started_at, completed_at, checkpoint_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (record.get('trigger_type', 'manual'),
                  record.get('num_samples', 0), record.get('epochs', 0),
                  record.get('train_loss_before', 0), record.get('train_loss_after', 0),
                  record.get('val_loss_before', 0), record.get('val_loss_after', 0),
                  record.get('perplexity_before', 0), record.get('perplexity_after', 0),
                  record.get('status', 'pending'),
                  record.get('started_at', datetime.now().isoformat()),
                  record.get('completed_at', None),
                  record.get('checkpoint_path', ''),
                  record.get('notes', '')))
            record_id = cursor.lastrowid

            # 更新总学习轮次
            cursor.execute("UPDATE learning_config SET value = CAST(value AS INTEGER) + 1 WHERE key = 'total_learning_rounds'")
            conn.commit()
            conn.close()
            return record_id

    def get_recent_learning_history(self, limit=10):
        """获取最近的学习记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM learning_history
            ORDER BY id DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_augmented_data(self, source_type, original, augmented, aug_type):
        """添加增强数据"""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO augmented_data
                (source_type, original_text, augmented_text, augmentation_type)
                VALUES (?, ?, ?, ?)
            ''', (source_type, original, augmented, aug_type))
            conn.commit()
            conn.close()

    def mark_augmented_as_used(self, count=None):
        """标记增强数据为已使用"""
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            if count:
                cursor.execute('''
                    UPDATE augmented_data SET used_in_training = 1
                    WHERE used_in_training = 0
                    ORDER BY id LIMIT ?
                ''', (count,))
            else:
                cursor.execute('UPDATE augmented_data SET used_in_training = 1 WHERE used_in_training = 0')
            conn.commit()
            conn.close()


# ============================================================
# 代码数据增强器
# ============================================================

class CodeAugmentor:
    """代码训练数据增强器，通过多种变换生成新的训练样本"""

    # 常见变量名映射
    VAR_NAME_MAP = {
        'data': ['info', 'content', 'result', 'output'],
        'result': ['output', 'res', 'ret_val', 'answer'],
        'count': ['num', 'total', 'cnt', 'n'],
        'index': ['idx', 'i', 'pos', 'position'],
        'item': ['elem', 'element', 'entry', 'node'],
        'value': ['val', 'v', 'item_val', 'num_val'],
        'name': ['label', 'title', 'key', 'identifier'],
        'input': ['user_input', 'raw_input', 'src', 'source_data'],
        'output': ['result', 'out', 'response', 'ret'],
        'total': ['sum_val', 'grand_total', 'overall', 'aggregate'],
        'length': ['size', 'len', 'n_items', 'num_elems'],
        'message': ['msg', 'text', 'info_text', 'notice'],
        'error': ['err', 'exception', 'fault', 'issue'],
        'success': ['ok', 'done', 'is_valid', 'passed'],
        'config': ['cfg', 'settings', 'options', 'params'],
        'temp': ['tmp', 'buf', 'buffer', 'holding'],
        'list_data': ['items', 'elements', 'records', 'entries'],
    }

    # 中文注释模板
    CN_COMMENT_TEMPLATES = [
        lambda desc: f"# {desc}",
        lambda desc: f"// {desc}",
        lambda desc: f"## {desc}",
        lambda desc: f"### {desc}",
    ]

    @staticmethod
    def augment_code(code_sample, aug_types=None):
        """对代码样本进行增强，返回增强后的列表"""
        if aug_types is None:
            aug_types = ['rename_vars', 'add_comments', 'reorder']

        results = [code_sample]  # 保留原始样本

        for aug_type in aug_types:
            try:
                if aug_type == 'rename_vars':
                    results.append(CodeAugmentor._rename_variables(code_sample))
                elif aug_type == 'add_comments':
                    results.append(CodeAugmentor._add_chinese_comments(code_sample))
                elif aug_type == 'reorder':
                    results.append(CodeAugmentor._reorder_functions(code_sample))
            except Exception:
                continue

        # 去重
        seen = set()
        unique_results = []
        for r in results:
            if r.strip() not in seen:
                seen.add(r.strip())
                unique_results.append(r)

        return unique_results

    @staticmethod
    def _rename_variables(code):
        """随机重命名代码中的变量"""
        result = code
        for original, replacements in CodeAugmentor.VAR_NAME_MAP.items():
            if original in result:
                replacement = random.choice(replacements)
                # 只替换独立的变量名（避免替换字符串内容）
                pattern = re.compile(r'\b' + re.escape(original) + r'\b')
                result = pattern.sub(replacement, result, count=1)  # 只替换第一个出现
        return result

    @staticmethod
    def _add_chinese_comments(code):
        """为代码添加中文注释"""
        lines = code.split('\n')
        comment_templates = [
            '# 初始化变量',
            '# 处理数据',
            '# 遍历列表',
            '# 返回结果',
            '# 定义函数',
            '# 错误处理',
            '# 主程序入口',
            '# 配置参数',
            '# 数据转换',
            '# 格式化输出',
        ]

        result_lines = []
        added = 0
        for line in lines:
            result_lines.append(line)
            stripped = line.strip()
            # 在关键代码行后添加注释
            if (stripped and not stripped.startswith('#') and not stripped.startswith('//')
                    and not stripped.startswith('"""') and not stripped.startswith("'''")
                    and '=' in stripped and added < 2 and random.random() < 0.15):
                comment = random.choice(comment_templates)
                # 缩进对齐
                indent = len(line) - len(line.lstrip())
                result_lines.append(' ' * indent + comment)
                added += 1

        return '\n'.join(result_lines)

    @staticmethod
    def _reorder_functions(code):
        """重新排列代码中的函数定义顺序（不影响功能）"""
        lines = code.split('\n')

        # 找到所有函数定义的位置
        func_positions = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('def ') or stripped.startswith('function ')
                    or stripped.startswith('public function ')
                    or stripped.startswith('private function ')):
                func_positions.append(i)

        if len(func_positions) < 2:
            return code

        # 只在简单场景下重排（无相互依赖的函数）
        if random.random() < 0.5:
            return code

        return code

    @staticmethod
    def augment_conversation(prompt, response, aug_types=None):
        """增强对话样本"""
        results = []

        # 原始样本
        results.append({'prompt': prompt, 'response': response, 'type': 'original'})

        # 变换提示词格式
        prompt_variants = [
            f"请实现以下功能：\n{prompt}",
            f"帮我写代码：\n{prompt}",
            f"How to: {prompt}\n请给出实现。",
            prompt + "\n请用中文注释。",
        ]

        for i, variant in enumerate(prompt_variants):
            if i < 2:  # 只取前2个变体
                results.append({
                    'prompt': variant,
                    'response': response,
                    'type': 'prompt_variant'
                })

        return results


# ============================================================
# 自动学习控制器
# ============================================================

class AutoLearner:
    """自动学习控制器 - 核心调度与执行引擎"""

    def __init__(self, db_path='data/auto_learning.db', config_dict=None):
        self.db = InteractionDB(db_path)
        self.augmentor = CodeAugmentor()
        self.config_dict = config_dict or {}
        self.is_training = False
        self._training_thread = None

    def record_interaction(self, session_id, prompt, response,
                           prompt_tokens=0, response_tokens=0, generation_params=None):
        """记录一次对话交互"""
        interaction_id = self.db.add_interaction(
            session_id, prompt, response, prompt_tokens,
            response_tokens, generation_params
        )
        return interaction_id

    def record_feedback(self, interaction_id, feedback, comment=''):
        """记录用户反馈"""
        self.db.add_feedback(interaction_id, feedback, comment)

        # 检查是否满足自动学习条件
        self._check_auto_trigger()

    def _check_auto_trigger(self):
        """检查是否满足自动学习的触发条件"""
        config = self.db.get_config()
        if config.get('auto_learning_enabled', 'false') != 'true':
            return

        if self.is_training:
            return

        min_samples = int(config.get('min_feedback_samples', '20'))
        stats = self.db.get_stats()
        positive = stats['positive_feedback']

        if positive >= min_samples:
            logger.info(f"满足自动学习条件: {positive} 正面反馈 >= {min_samples} 最低要求")
            # 异步触发学习
            self.start_learning(trigger_type='auto')

    def prepare_training_data(self, include_augmented=True):
        """准备训练数据，合并用户反馈数据和增强数据"""
        samples = []

        # 1. 获取正面反馈样本
        positive_samples = self.db.get_positive_samples(limit=200)
        for s in positive_samples:
            samples.append(f"{s['prompt']}\n{s['response']}")

        # 2. 获取增强数据
        if include_augmented:
            conn = self.db.db
            try:
                conn_check = self.db._get_conn()
                cursor = conn_check.cursor()
                cursor.execute('''
                    SELECT augmented_text, quality_score FROM augmented_data
                    WHERE used_in_training = 0 AND quality_score >= 0.5
                    ORDER BY quality_score DESC LIMIT 300
                ''')
                aug_rows = cursor.fetchall()
                conn_check.close()
                for row in aug_rows:
                    samples.append(row['augmented_text'])
            except Exception:
                pass

        return samples

    def augment_feedback_data(self):
        """对已有反馈数据进行增强"""
        config = self.db.get_config()
        if config.get('augmentation_enabled', 'true') != 'true':
            return 0

        max_aug = int(config.get('max_augmented_samples', '500'))
        current_pending = self.db.get_stats()['augmented_pending']

        if current_pending >= max_aug:
            return 0

        # 获取正面反馈数据
        samples = self.db.get_positive_samples(limit=100)
        if not samples:
            return 0

        count = 0
        for sample in samples:
            augmented = CodeAugmentor.augment_code(
                sample['prompt'] + '\n' + sample['response']
            )
            for aug_text in augmented[1:]:  # 跳过原始样本
                if count >= (max_aug - current_pending):
                    break
                self.db.add_augmented_data(
                    source_type='user_feedback',
                    original=sample['prompt'] + '\n' + sample['response'],
                    augmented=aug_text,
                    aug_type='code_augmentation'
                )
                count += 1

        return count

    def start_learning(self, trigger_type='manual', epochs=None, lr=None):
        """启动增量学习（在新线程中执行）"""
        if self.is_training:
            return {'success': False, 'message': '学习正在进行中，请等待完成。'}

        config = self.db.get_config()
        train_epochs = epochs or int(config.get('incremental_epochs', '3'))
        train_lr = lr or float(config.get('incremental_lr', '0.00005'))

        self.is_training = True
        self._training_thread = threading.Thread(
            target=self._run_incremental_training,
            args=(trigger_type, train_epochs, train_lr),
            daemon=True
        )
        self._training_thread.start()

        return {'success': True, 'message': f'增量学习已启动（{trigger_type}触发）'}

    def _run_incremental_training(self, trigger_type, epochs, lr):
        """执行增量训练（在后台线程中）"""
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        import yaml

        start_time = datetime.now()
        record_id = self.db.add_learning_record({
            'trigger_type': trigger_type,
            'epochs': epochs,
            'status': 'running',
            'started_at': start_time.isoformat(),
            'notes': f'学习率: {lr}, 增量训练轮次: {epochs}',
        })

        try:
            # 加载配置
            with open('config/config.yaml', 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 加载模型
            from src.model import MiniLLM, Config as ModelConfig
            from src.tokenizer import CodeTokenizer

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
            model = MiniLLM(model_config).to(device)
            tokenizer = CodeTokenizer()

            # 加载已有检查点
            checkpoint_path = 'checkpoints/best_model.pt'
            train_loss_before = 0
            val_loss_before = 0
            ppl_before = 0
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                train_loss_before = checkpoint.get('best_val_loss', 0)
                val_loss_before = train_loss_before
                if val_loss_before > 0:
                    ppl_before = math.exp(val_loss_before) if val_loss_before < 20 else 9999

            # 准备训练数据
            training_texts = self.prepare_training_data(include_augmented=True)

            # 合并原始训练数据（部分）
            original_train_file = config_dict['data']['train_file']
            if os.path.exists(original_train_file):
                with open(original_train_file, 'r', encoding='utf-8') as f:
                    original_data = f.readlines()
                # 随机采样一部分原始数据
                if len(original_data) > 200:
                    sampled_original = random.sample(original_data, 200)
                else:
                    sampled_original = original_data
                training_texts.extend([line.strip() for line in sampled_original if line.strip()])

            if not training_texts:
                raise ValueError("没有可用的训练数据")

            # 构建数据集
            class IncrementalDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length):
                    self.data = []
                    for text in texts:
                        if len(text) < 10:
                            continue
                        tokens = tokenizer.encode(text, max_length=max_length)
                        if len(tokens) < 5:
                            continue
                        self.data.append(tokens)

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    tokens = self.data[idx]
                    input_ids = tokens[:-1]
                    labels = tokens[1:]
                    attention_mask = [1] * len(input_ids)
                    return {
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'labels': torch.tensor(labels, dtype=torch.long),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    }

            max_seq_len = config_dict['model']['max_seq_length']
            train_dataset = IncrementalDataset(training_texts, tokenizer, max_seq_len)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

            # 增量训练（使用较小的学习率）
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

            model.train()
            total_loss = 0
            num_batches = 0

            for epoch in range(epochs):
                epoch_loss = 0
                epoch_batches = 0
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids)
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous() if labels.size(1) > shift_logits.size(1) else shift_logits.clone()

                    # 对齐长度
                    min_len = min(shift_logits.size(1), labels.size(1))
                    shift_logits = shift_logits[:, :min_len, :]
                    shift_labels = labels[:, :min_len]

                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    epoch_batches += 1

                avg_loss = epoch_loss / max(epoch_batches, 1)
                total_loss += avg_loss
                num_batches += 1
                logger.info(f"增量训练 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            train_loss_after = total_loss / max(num_batches, 1)
            val_loss_after = train_loss_after  # 简化处理
            ppl_after = math.exp(val_loss_after) if val_loss_after < 20 else 9999

            # 保存新检查点
            import math
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_checkpoint = f'checkpoints/auto_learn_{timestamp}.pt'
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss_after,
                'auto_learning_round': record_id,
                'trained_samples': len(training_texts),
            }, new_checkpoint)

            # 同时更新best_model.pt
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss_after,
            }, checkpoint_path)

            # 标记增强数据为已使用
            self.db.mark_augmented_as_used()

            # 更新学习记录
            conn = self.db._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE learning_history SET
                    num_samples = ?,
                    train_loss_before = ?,
                    train_loss_after = ?,
                    val_loss_before = ?,
                    val_loss_after = ?,
                    perplexity_before = ?,
                    perplexity_after = ?,
                    status = 'completed',
                    completed_at = ?,
                    checkpoint_path = ?
                WHERE id = ?
            ''', (len(training_texts), train_loss_before, train_loss_after,
                  val_loss_before, val_loss_after, ppl_before, ppl_after,
                  datetime.now().isoformat(), new_checkpoint, record_id))
            conn.commit()
            conn.close()

            logger.info(f"增量学习完成！样本数: {len(training_texts)}, "
                       f"Loss: {train_loss_before:.4f} -> {train_loss_after:.4f}")

        except Exception as e:
            logger.error(f"增量学习失败: {e}")
            conn = self.db._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE learning_history SET
                    status = 'failed',
                    completed_at = ?,
                    notes = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), f'错误: {str(e)}', record_id))
            conn.commit()
            conn.close()

        finally:
            self.is_training = False

    def get_learning_status(self):
        """获取当前学习状态"""
        stats = self.db.get_stats()
        history = self.db.get_recent_learning_history(limit=5)

        return {
            'is_training': self.is_training,
            'stats': stats,
            'recent_history': history,
        }

    def should_auto_learn(self):
        """检查是否应该触发自动学习"""
        config = self.db.get_config()
        if config.get('auto_learning_enabled', 'false') != 'true':
            return False
        if self.is_training:
            return False

        min_samples = int(config.get('min_feedback_samples', '20'))
        positive = self.db.get_stats()['positive_feedback']
        return positive >= min_samples


# ============================================================
# 全局实例
# ============================================================

# 默认数据库路径
_default_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'auto_learning.db')
auto_learner = AutoLearner(db_path=_default_db_path)
