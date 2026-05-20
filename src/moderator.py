import re

class ContentModerator:
    def __init__(self):
        self.sensitive_keywords = [
            '暴力', '恐怖', '极端', '反动', '分裂', '颠覆',
            '色情', '淫秽', '低俗', '赌博', '诈骗', '毒品',
            '攻击', '侮辱', '诽谤', '造谣', '煽动', '仇恨',
            '违法', '犯罪', '违禁', '非法', '管制'
        ]
        
        self.warning_patterns = [
            r'自杀|自伤|自残',
            r'暴力|伤害|杀人',
            r'色情|性|淫秽',
            r'诈骗|欺诈|虚假',
            r'违禁|违法|犯罪'
        ]
    
    def check_content(self, text):
        issues = []
        
        for keyword in self.sensitive_keywords:
            if keyword in text:
                issues.append(f"包含敏感关键词：{keyword}")
        
        for pattern in self.warning_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"包含违规内容模式：{pattern}")
        
        return {
            'is_safe': len(issues) == 0,
            'issues': issues
        }
    
    def filter_response(self, text):
        filtered_text = text
        for keyword in self.sensitive_keywords:
            filtered_text = filtered_text.replace(keyword, '***')
        return filtered_text


moderator = ContentModerator()
