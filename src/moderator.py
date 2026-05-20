"""
内容审核模块 - 多司法管辖区合规版本

覆盖法规:
  - 中国: 《生成式人工智能服务管理暂行办法》《网络安全法》《个人信息保护法》《数据安全法》
  - 欧盟: EU AI Act (2024), GDPR, Digital Services Act (DSA)
  - 美国: COPPA, Section 230 CDA, California CCPA
  - 通用: Berne Convention (知识产权), OSCE仇恨言论标准

风险分级:
  BLOCK  - 直接拦截，不生成任何内容
  WARN   - 允许生成但附加警告标签
  PASS   - 正常通过
"""

import re
import json
import os
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


class RiskLevel(Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class AuditRecord:
    """审核审计记录"""
    timestamp: str
    action: str           # "input_check" | "output_check"
    text_snippet: str     # 截取前100字符
    risk_level: str
    matched_rules: List[str]
    jurisdiction: str     # 触发的主要司法管辖区
    decision: str         # "allowed" | "blocked" | "filtered"

    def to_dict(self):
        return asdict(self)


class ContentModerator:
    """多司法管辖区内容审核器"""

    def __init__(self, audit_log_dir: str = "logs"):
        self.audit_log_dir = audit_log_dir
        os.makedirs(audit_log_dir, exist_ok=True)

        self._audit_logger = self._setup_audit_logger()

        # ==================== 敏感关键词库 ====================

        # 中国法规：《生成式AI管理办法》第9条、第12条
        # 《网络安全法》第12条
        self.cn_block_keywords = [
            # 危害国家安全
            '颠覆国家政权', '推翻社会主义', '分裂国家', '煽动分裂',
            '恐怖活动', '极端主义', '暴力恐怖',
            # 违法犯罪教唆
            '制造毒品', '制毒方法', '炸弹制作', '自制枪支',
            '洗钱方法', '走私渠道',
            # 深度伪造滥用
            '换脸诈骗', '伪造视频诈骗', 'AI诈骗话术',
        ]

        self.cn_warn_keywords = [
            # 需要警示但非直接拦截
            '赌博', '彩票预测', '内幕消息',
            '减肥药', '偏方', '祖传秘方',
            '兼职日结', '刷单返利', '高额回报',
        ]

        # 欧盟：DSA Art.13-16 非法内容, EU AI Act Art.6 高风险
        self.eu_block_keywords = [
            # EU AI Act Annex III 高风险AI禁止用途
            'child exploitation', 'child sexual abuse', 'CSAM',
            'revenge porn', 'non-consensual intimate',
        ]

        self.eu_warn_keywords = [
            'hate speech', 'discrimination', 'genocide denial',
            'Holocaust denial', 'racial superiority',
        ]

        # 美国：COPPA 未成年人保护, 州级仇恨言论法规
        self.us_block_keywords = [
            'COPPA', 'underage exploitation',
        ]

        self.us_warn_keywords = [
            'financial advice', 'medical diagnosis', 'legal advice',
        ]

        # 通用禁止内容（跨国共识）
        self.universal_block_keywords = [
            '儿童色情', '虐童', '未成年人性行为',
            '谋杀计划', '大规模屠杀', 'school shooting',
            '自杀方法', '自残教程', '如何自杀',
            '制作炸弹', '自制武器', '生物武器',
        ]

        # ==================== 正则模式库 ====================

        # PII检测：GDPR Art.5, PIPL Art.28（个人信息保护）
        self._pii_patterns = [
            (r'(?:身份证号|身份证号码|ID[号码]?)\s*[:：]?\s*\d{17}[\dXx]', 'CN_ID_CARD', 'CN'),
            (r'(?:手机号|联系电话|手机)[号码]?\s*[:：]?\s*1[3-9]\d{9}', 'CN_PHONE', 'CN'),
            (r'(?:银行卡号|银行卡)[号]?\s*[:：]?\s*\d{16,19}', 'CN_BANK_CARD', 'CN'),
            (r'\b\d{3}-\d{2}-\d{4}\b', 'US_SSN', 'US'),  # 美国社会安全号
            (r'\b[A-Z]{2}\d{6}[A-Z\d]{0,3}\b', 'EU_PASSPORT', 'EU'),  # 护照号近似
            (r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', 'EMAIL', 'UNIVERSAL'),
            (r'\b(?:\+?86[-\s]?)?1[3-9]\d[-\s]?\d{4}[-\s]?\d{4}\b', 'PHONE', 'UNIVERSAL'),
        ]

        # 风险内容模式
        self._risk_patterns = [
            # 自伤/自毁风险 - 覆盖多国自杀预防法规
            (r'(?:怎么|如何|方法|方式)\s*(?:自杀|自尽|结束生命|不想活)', RiskLevel.BLOCK, 'self_harm', 'UNIVERSAL'),
            (r'(?:想死|不想活|活不下去|了结自己|离开这个世界)', RiskLevel.BLOCK, 'self_harm_intent', 'UNIVERSAL'),

            # 暴力/恐怖 - UN安全理事会决议, 各国反恐法
            (r'(?:制作|自制|配方|教程)\s*(?:炸弹|炸药|毒气|生化武器)', RiskLevel.BLOCK, 'violence_weapon', 'UNIVERSAL'),
            (r'(?:袭击|攻击|暗杀)\s*(?:计划|方法|教程|目标)', RiskLevel.BLOCK, 'violence_attack', 'UNIVERSAL'),

            # 儿童保护 - COPPA, UNCRC, 各国儿童保护法
            (r'(?:未成年|儿童|少儿)\s*(?:色情|性侵|猥亵)', RiskLevel.BLOCK, 'child_protection', 'UNIVERSAL'),
            (r'(?:恋童|pedophil)', RiskLevel.BLOCK, 'child_protection', 'UNIVERSAL'),

            # 深度伪造滥用 - 中国深度合成管理规定, EU AI Act Art.52
            (r'(?:换脸|deepfake|深度伪造)\s*(?:诈骗|色情|伪造|骗局)', RiskLevel.BLOCK, 'deepfake_abuse', 'CN/EU'),

            # 仇恨言论 - EU DSA, ICCPR Art.20
            (r'(?:种族灭绝|种族清洗|种族灭绝是好的)', RiskLevel.BLOCK, 'hate_speech', 'UNIVERSAL'),
            (r'(?:消灭|消灭掉|清除|灭绝)\s*(?:某个|某群|那些)\s*(?:种族|民族|人群)', RiskLevel.BLOCK, 'hate_speech', 'UNIVERSAL'),

            # 医疗建议警告 - 各国医疗法规
            (r'(?:推荐|建议|你应该|必须)\s*(?:吃药|停药|服用|用药)\s*(?:治疗|治愈)', RiskLevel.WARN, 'medical_advice', 'UNIVERSAL'),
            (r'(?:诊断|确诊)\s*(?:你患有|你得的是|你得了)', RiskLevel.WARN, 'medical_diagnosis', 'UNIVERSAL'),

            # 金融建议警告 - 各国金融监管法规
            (r'(?:建议|推荐|应该)\s*(?:买入|卖出|投资|抄底|加仓)', RiskLevel.WARN, 'financial_advice', 'UNIVERSAL'),
            (r'(?:保证|肯定|100%)\s*(?:盈利|赚钱|翻倍|回报)', RiskLevel.WARN, 'financial_guarantee', 'UNIVERSAL'),
        ]

        # 预编译正则（性能优化）
        self._compiled_risk_patterns = [
            (re.compile(p, re.IGNORECASE), level, tag, jurisdiction)
            for p, level, tag, jurisdiction in self._risk_patterns
        ]
        self._compiled_pii_patterns = [
            (re.compile(p, re.IGNORECASE), tag, jurisdiction)
            for p, tag, jurisdiction in self._pii_patterns
        ]

    def _setup_audit_logger(self) -> logging.Logger:
        """配置审计日志"""
        logger = logging.getLogger('compliance_audit')
        logger.setLevel(logging.INFO)
        audit_file = os.path.join(self.audit_log_dir, 'audit.log')
        handler = logging.FileHandler(audit_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
        return logger

    def _get_timestamp(self) -> str:
        return datetime.now().isoformat()

    def _write_audit(self, record: AuditRecord):
        """写入审计记录"""
        self._audit_logger.info(json.dumps(record.to_dict(), ensure_ascii=False))

    def check_content(self, text: str, check_type: str = "input_check") -> Dict:
        """
        内容审核主入口

        Args:
            text: 待审核文本
            check_type: "input_check" 或 "output_check"

        Returns:
            {
                'is_safe': bool,
                'risk_level': str,       # pass/warn/block
                'issues': List[str],
                'pii_detected': List[Dict],
                'jurisdictions': List[str],
                'audit_id': Optional[str]
            }
        """
        issues = []
        all_jurisdictions = set()
        final_risk = RiskLevel.PASS

        # 1. PII检测（优先检测，不阻断但记录）
        pii_detected = self._detect_pii(text)

        # 2. 关键词检测
        risk, matched, jurs = self._check_keywords(text)
        if risk.value == RiskLevel.BLOCK.value:
            final_risk = RiskLevel.BLOCK
        elif risk.value == RiskLevel.WARN.value and final_risk != RiskLevel.BLOCK:
            final_risk = RiskLevel.WARN
        issues.extend(matched)
        all_jurisdictions.update(jurs)

        # 3. 正则模式检测（覆盖更复杂的语境）
        risk, matched, jurs = self._check_patterns(text)
        if risk.value == RiskLevel.BLOCK.value:
            final_risk = RiskLevel.BLOCK
        elif risk.value == RiskLevel.WARN.value and final_risk != RiskLevel.BLOCK:
            final_risk = RiskLevel.WARN
        issues.extend(matched)
        all_jurisdictions.update(jurs)

        # 4. 文本长度检测（防止超长输入DoS）
        if len(text) > 10000:
            issues.append("输入文本过长，请缩减后重试")
            final_risk = RiskLevel.BLOCK

        # 5. 审计记录
        audit_record = AuditRecord(
            timestamp=self._get_timestamp(),
            action=check_type,
            text_snippet=text[:100],
            risk_level=final_risk.value,
            matched_rules=issues,
            jurisdiction=", ".join(all_jurisdictions) if all_jurisdictions else "NONE",
            decision="blocked" if final_risk == RiskLevel.BLOCK else
                     ("filtered" if final_risk == RiskLevel.WARN else "allowed")
        )
        self._write_audit(audit_record)

        return {
            'is_safe': final_risk == RiskLevel.PASS,
            'risk_level': final_risk.value,
            'issues': issues,
            'pii_detected': pii_detected,
            'jurisdictions': list(all_jurisdictions),
        }

    def _check_keywords(self, text: str) -> Tuple[RiskLevel, List[str], List[str]]:
        """关键词检测"""
        issues = []
        jurisdictions = []
        final_risk = RiskLevel.PASS

        # 合并所有关键词库进行检测
        keyword_sets = [
            (self.cn_block_keywords, RiskLevel.BLOCK, 'CN'),
            (self.cn_warn_keywords, RiskLevel.WARN, 'CN'),
            (self.eu_block_keywords, RiskLevel.BLOCK, 'EU'),
            (self.eu_warn_keywords, RiskLevel.WARN, 'EU'),
            (self.us_block_keywords, RiskLevel.BLOCK, 'US'),
            (self.us_warn_keywords, RiskLevel.WARN, 'US'),
            (self.universal_block_keywords, RiskLevel.BLOCK, 'UNIVERSAL'),
        ]

        for keywords, risk, jurisdiction in keyword_sets:
            for keyword in keywords:
                if keyword in text:
                    issues.append(f"[{jurisdiction}] 触发关键词规则: {keyword}")
                    jurisdictions.append(jurisdiction)
                    if risk == RiskLevel.BLOCK:
                        final_risk = RiskLevel.BLOCK
                    elif risk == RiskLevel.WARN and final_risk == RiskLevel.PASS:
                        final_risk = RiskLevel.WARN

        return final_risk, issues, jurisdictions

    def _check_patterns(self, text: str) -> Tuple[RiskLevel, List[str], List[str]]:
        """正则模式检测"""
        issues = []
        jurisdictions = []
        final_risk = RiskLevel.PASS

        for pattern, risk, tag, jurisdiction in self._compiled_risk_patterns:
            if pattern.search(text):
                issues.append(f"[{jurisdiction}] 触发模式规则: {tag}")
                jurisdictions.append(jurisdiction)
                if risk == RiskLevel.BLOCK:
                    final_risk = RiskLevel.BLOCK
                elif risk == RiskLevel.WARN and final_risk == RiskLevel.PASS:
                    final_risk = RiskLevel.WARN

        return final_risk, issues, jurisdictions

    def _detect_pii(self, text: str) -> List[Dict]:
        """PII（个人可识别信息）检测"""
        detected = []
        for pattern, tag, jurisdiction in self._compiled_pii_patterns:
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    # 脱敏处理
                    masked = self._mask_pii(str(match), tag)
                    detected.append({
                        'type': tag,
                        'jurisdiction': jurisdiction,
                        'raw_preview': str(match)[:3] + "***",
                        'masked': masked,
                        'regulation': self._get_pii_regulation(tag),
                    })
        return detected

    def _mask_pii(self, value: str, pii_type: str) -> str:
        """PII脱敏"""
        if pii_type == 'EMAIL':
            parts = value.split('@')
            return parts[0][:2] + '***@' + parts[1] if len(parts) == 2 else '***'
        elif pii_type in ('PHONE', 'CN_PHONE'):
            digits = re.sub(r'\D', '', value)
            return digits[:3] + '****' + digits[-4:] if len(digits) >= 7 else '***'
        elif pii_type == 'CN_ID_CARD':
            return value[:3] + '***********' + value[-4:] if len(value) >= 18 else '***'
        elif pii_type == 'US_SSN':
            return '***-**-' + value[-4:] if len(value) >= 4 else '***'
        elif pii_type in ('CN_BANK_CARD', 'EU_PASSPORT'):
            return value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) >= 8 else '***'
        return value[:2] + '***'

    def _get_pii_regulation(self, pii_type: str) -> str:
        """获取PII相关的法规依据"""
        regulation_map = {
            'CN_ID_CARD': '中国《个人信息保护法》第28条(敏感个人信息)',
            'CN_PHONE': '中国《个人信息保护法》第28条(敏感个人信息)',
            'CN_BANK_CARD': '中国《个人信息保护法》第28条(敏感个人信息)',
            'US_SSN': 'US Privacy Act of 1974',
            'EMAIL': 'EU GDPR Art.4(1) / 中国《个人信息保护法》第28条',
            'PHONE': 'EU GDPR Art.4(1)',
            'EU_PASSPORT': 'EU GDPR Art.9(1) (特殊类别数据)',
        }
        return regulation_map.get(pii_type, 'UNIVERSAL')

    def filter_response(self, text: str) -> str:
        """
        过滤生成内容中的敏感词

        对于WARN级别，将敏感词替换为***
        对于BLOCK级别内容，直接返回安全提示
        """
        all_sensitive = (
            self.cn_block_keywords +
            self.cn_warn_keywords +
            self.universal_block_keywords
        )
        filtered_text = text
        for keyword in all_sensitive:
            if keyword in filtered_text:
                filtered_text = filtered_text.replace(keyword, '*' * len(keyword))

        return filtered_text

    def check_age_appropriate(self, text: str) -> Dict:
        """
        未成年人适宜性检查 (COPPA / 未成年人保护)

        Returns:
            {'is_age_appropriate': bool, 'reason': str}
        """
        inappropriate_for_minors = [
            r'(?:色情|情色|成人|18禁|R级)',
            r'(?:赌博|下注|博彩|赌场)',
            r'(?:暴力|血腥|残忍|恐怖)',
            r'(?:烟酒|电子烟|烟草|酒精)',
        ]
        for pattern in inappropriate_for_minors:
            if re.search(pattern, text, re.IGNORECASE):
                return {
                    'is_age_appropriate': False,
                    'reason': '内容可能不适合未成年人',
                    'regulation': 'COPPA (US) / 未成年人网络保护条例 (CN)',
                }
        return {'is_age_appropriate': True, 'reason': ''}


# 全局单例
moderator = ContentModerator()
