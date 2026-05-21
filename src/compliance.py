"""
合规基础设施模块

提供:
  - 安全响应头（CSP, HSTS等）- 符合EU NIS2 Directive / 中国网络安全等级保护
  - 速率限制 - 防止滥用，符合各平台责任法规
  - 审计日志接口
  - 数据保留策略 - GDPR Art.5(1)(e) / PIPL Art.22
  - 用户权利管理 - GDPR Art.15-22 / CCPA §1798.100-125
"""

import time
import os
import json
import logging
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
from typing import Dict, Optional, Callable, List


class SecurityHeaders:
    """安全响应头生成器"""

    @staticmethod
    def get_headers() -> Dict[str, str]:
        """
        生成符合多国法规的安全响应头

        参考:
          - EU NIS2 Directive (网络安全)
          - OWASP Secure Headers Project
          - 中国网络安全等级保护2.0
          - US CISA Binding Operational Directive 18-01
        """
        return {
            # Content Security Policy - 防止XSS注入
            'Content-Security-Policy': (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "frame-ancestors 'none';"
            ),
            # 防止点击劫持 - 同源策略
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            # HSTS - 强制HTTPS (RFC 6797)
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            # Referrer Policy - 隐私保护 (GDPR Art.5)
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            # Permissions Policy - 限制浏览器功能
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            # CORS - 跨域访问控制
            # 本地开发默认允许 localhost；生产环境请改为你的域名
            'Access-Control-Allow-Origin': 'http://localhost:5000',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        }


class RateLimiter:
    """
    速率限制器

    符合:
      - EU DSA Art.16 (平台义务 - 防止系统性滥用)
      - 中国《网络安全法》第22条
      - US CFAA (计算机欺诈和滥用法)
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> Dict:
        """检查是否允许请求"""
        now = time.time()
        window_start = now - self.window_seconds

        # 清理过期记录
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        is_allowed = len(self._requests[client_id]) < self.max_requests
        if is_allowed:
            self._requests[client_id].append(now)

        return {
            'allowed': is_allowed,
            'remaining': max(0, self.max_requests - len(self._requests[client_id])),
            'reset_at': int(now + self.window_seconds),
            'limit': self.max_requests,
        }

    def get_retry_after(self, client_id: str) -> Optional[int]:
        """获取需要等待的秒数"""
        record = self.is_allowed(client_id)
        if not record['allowed']:
            return record['reset_at'] - int(time.time())
        return None


class DataRetentionManager:
    """
    数据保留管理器

    符合:
      - GDPR Art.5(1)(e) - 存储限制原则
      - GDPR Art.17 - 被遗忘权(删除权)
      - PIPL Art.22 - 个人信息删除权
      - CCPA §1798.105 - 消费者删除权请求
    """

    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days
        self.data_dir = "data/user_sessions"

    def cleanup_expired_data(self) -> Dict:
        """清理过期数据"""
        if not os.path.exists(self.data_dir):
            return {'cleaned': 0, 'message': 'No user data directory'}

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        cleaned = 0

        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            if os.path.isfile(filepath):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_mtime < cutoff:
                    os.remove(filepath)
                    cleaned += 1

        return {
            'cleaned': cleaned,
            'retention_days': self.retention_days,
            'regulation': 'GDPR Art.5(1)(e) / PIPL Art.22 / CCPA §1798.105',
        }

    def delete_user_data(self, session_id: str) -> Dict:
        """
        删除用户数据（GDPR被遗忘权 / PIPL删除权）

        Args:
            session_id: 用户会话标识
        """
        filepath = os.path.join(self.data_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return {
                'deleted': True,
                'session_id': session_id,
                'regulation': 'GDPR Art.17 / PIPL Art.22 / CCPA §1798.105',
            }
        return {'deleted': False, 'message': 'No data found for this session'}

    def get_user_data_export(self, session_id: str) -> Dict:
        """
        导出用户数据（GDPR数据可携权 Art.20 / CCPA §1798.100）

        Args:
            session_id: 用户会话标识
        """
        filepath = os.path.join(self.data_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                'found': True,
                'data': data,
                'regulation': 'GDPR Art.20 (数据可携权) / CCPA §1798.100 (知情权)',
            }
        return {'found': False, 'message': 'No data found for this session'}


class UserRightsManager:
    """
    用户权利管理器

    实现:
      - GDPR Art.15 - 访问权
      - GDPR Art.16 - 更正权
      - GDPR Art.17 - 删除权(被遗忘权)
      - GDPR Art.20 - 数据可携权
      - GDPR Art.21 - 反对权
      - GDPR Art.22 - 自动化决策权
      - PIPL Art.44-49 - 个人信息主体权利
      - CCPA §1798.100-125 - 加州消费者隐私权
    """

    AVAILABLE_RIGHTS = {
        'GDPR': {
            'access': {'article': 'Art.15', 'name': '访问权', 'description': '获取你的个人数据副本'},
            'rectification': {'article': 'Art.16', 'name': '更正权', 'description': '更正不准确的个人数据'},
            'erasure': {'article': 'Art.17', 'name': '删除权(被遗忘权)', 'description': '请求删除你的个人数据'},
            'portability': {'article': 'Art.20', 'name': '数据可携权', 'description': '以结构化格式获取数据'},
            'object': {'article': 'Art.21', 'name': '反对权', 'description': '反对处理你的个人数据'},
            'automated_decision': {'article': 'Art.22', 'name': '自动化决策权', 'description': '不接受纯自动化决策'},
        },
        'PIPL': {
            'access': {'article': 'Art.44', 'name': '知情权', 'description': '查阅你的个人信息'},
            'copy': {'article': 'Art.45', 'name': '复制权', 'description': '复制你的个人信息'},
            'correction': {'article': 'Art.46', 'name': '更正权', 'description': '请求更正个人信息'},
            'deletion': {'article': 'Art.47', 'name': '删除权', 'description': '请求删除个人信息'},
            'portability': {'article': 'Art.45', 'name': '可携权', 'description': '将信息转移至指定处理者'},
        },
        'CCPA': {
            'know': {'article': '§1798.100', 'name': '知情权', 'description': '了解收集的个人信息'},
            'delete': {'article': '§1798.105', 'name': '删除权', 'description': '请求删除个人信息'},
            'correct': {'article': '§1798.106', 'name': '更正权', 'description': '更正不准确的个人信息'},
            'opt_out': {'article': '§1798.120', 'name': '退出权', 'description': '选择退出个人信息出售'},
            'non_discrimination': {'article': '§1798.125', 'name': '不歧视权', 'description': '行使权利不受到歧视待遇'},
        },
    }

    @classmethod
    def get_rights_summary(cls) -> Dict:
        """获取所有支持的用户权利概览"""
        return cls.AVAILABLE_RIGHTS

    @classmethod
    def get_regulation_compliance_summary(cls) -> Dict:
        """获取合规概览"""
        return {
            'data_principles': [
                {
                    'principle': '合法性、公正性、透明性',
                    'gdpr': 'Art.5(1)(a)',
                    'pipl': 'Art.5-7',
                    'ccpa': '§1798.100(b)',
                },
                {
                    'principle': '目的限制',
                    'gdpr': 'Art.5(1)(b)',
                    'pipl': 'Art.6',
                },
                {
                    'principle': '数据最小化',
                    'gdpr': 'Art.5(1)(c)',
                    'pipl': 'Art.6',
                },
                {
                    'principle': '准确性',
                    'gdpr': 'Art.5(1)(d)',
                    'pipl': 'Art.8',
                },
                {
                    'principle': '存储限制',
                    'gdpr': 'Art.5(1)(e)',
                    'pipl': 'Art.22',
                },
                {
                    'principle': '完整性和保密性',
                    'gdpr': 'Art.5(1)(f)',
                    'pipl': 'Art.9, 51',
                },
            ],
        }


class ComplianceAuditLogger:
    """合规审计日志器"""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._logger = logging.getLogger('compliance')
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            os.path.join(log_dir, 'compliance.log'), encoding='utf-8'
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        )
        self._logger.addHandler(handler)

    def log_event(self, event_type: str, details: Dict):
        """记录合规事件"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))

    def log_data_access(self, accessor: str, data_type: str, purpose: str):
        """记录数据访问"""
        self.log_event('DATA_ACCESS', {
            'accessor': accessor,
            'data_type': data_type,
            'purpose': purpose,
        })

    def log_security_event(self, event: str, severity: str, details: Dict = None):
        """记录安全事件"""
        self.log_event('SECURITY', {
            'event': event,
            'severity': severity,
            'details': details or {},
        })


# 全局实例
security_headers = SecurityHeaders()
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
data_retention = DataRetentionManager(retention_days=90)
compliance_logger = ComplianceAuditLogger()

# ==================== 代码安全合规常量 ====================

# 代码模型允许生成的代码类别（白名单）
CODE_GENERATION_WHITELIST = {
    'algorithm': True,         # 算法实现
    'data_structure': True,    # 数据结构
    'web_development': True,   # Web开发
    'api': True,               # API开发
    'database': True,          # 数据库操作
    'testing': True,           # 测试代码
    'utility': True,           # 工具函数
    'automation': True,        # 自动化脚本（非恶意）
    'data_analysis': True,     # 数据分析
    'machine_learning': True,  # 机器学习
    'game_development': True,  # 游戏开发
    'system_admin': True,      # 系统管理（非攻击性）
}

# 代码安全分类标签（用于输出标注）
CODE_SECURITY_LABELS = {
    'malicious_code': {
        'label': 'MALICIOUS_CODE_BLOCKED',
        'description': '请求涉及恶意代码生成，已被拦截',
        'regulation': 'CFAA (US) / 网络安全法 (CN) / Computer Misuse Act (UK)',
    },
    'exploit_code': {
        'label': 'EXPLOIT_CODE_BLOCKED',
        'description': '请求涉及漏洞利用代码，已被拦截',
        'regulation': 'CFAA (US) / 网络安全法 (CN) / 各国反计算机犯罪法',
    },
    'attack_script': {
        'label': 'ATTACK_SCRIPT_BLOCKED',
        'description': '请求涉及网络攻击脚本，已被拦截',
        'regulation': 'CFAA (US) / 网络安全法 (CN) / Budapest Convention on Cybercrime',
    },
    'security_education': {
        'label': 'SECURITY_EDUCATION',
        'description': '防御性安全代码，允许生成',
        'regulation': '网络安全教育豁免',
    },
}
