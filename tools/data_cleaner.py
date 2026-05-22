#!/usr/bin/env python
"""
数据清洗管线 (Data Cleaning Pipeline)

功能:
  1. 精确/模糊去重 (Exact / Fuzzy Dedup)
  2. 质量过滤 (Quality Filter)
  3. 语法验证 (Syntax Validation)

用法:
  python tools/data_cleaner.py --input data/train.txt --output data/train_clean.txt
  python tools/data_cleaner.py --input data/ --output data_clean/ --lang python

设计原则:
  - 纯 Python 标准库 + 轻量依赖（无需深度学习框架）
  - 管线模式：每个过滤器可独立使用，也可串联
  - 报告模式：清洗过程输出详细统计
"""

import argparse
import hashlib
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional, Callable


# ============================================================
# 去重 (Deduplication)
# ============================================================

class Deduplicator:
    """
    代码样本去重器

    支持两种模式:
      - exact: 精确去重（基于 MD5 哈希）
      - fuzzy: 模糊去重（基于 MinHash / 规范化的近似检测）
    """

    def __init__(self, method: str = "exact"):
        self.method = method
        self._seen_hashes: Set[str] = set()
        self._removed_count: int = 0

    @property
    def removed(self) -> int:
        return self._removed_count

    def _hash_exact(self, text: str) -> str:
        """精确哈希（归一化后）"""
        # 去除首尾空白，统一换行符
        normalized = text.strip().replace("\r\n", "\n").replace("\r", "\n")
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _hash_fuzzy(self, text: str) -> str:
        """模糊哈希（去空格/注释后的近似匹配）"""
        # 去除所有空白符、注释
        stripped = re.sub(r"\s+", "", text)
        stripped = re.sub(r"#.*$", "", stripped, flags=re.MULTILINE)
        if len(stripped) < 10:
            # 太短无法做有意义的模糊匹配，回退到精确
            return self._hash_exact(text)
        return hashlib.md5(stripped.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """检查是否重复"""
        if self.method == "fuzzy":
            h = self._hash_fuzzy(text)
        else:
            h = self._hash_exact(text)

        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def deduplicate(self, samples: List[str]) -> Tuple[List[str], int]:
        """
        对样本列表去重

        Args:
            samples: 文本样本列表

        Returns:
            (去重后列表, 移除数量)
        """
        unique = []
        removed = 0
        for sample in samples:
            if self.is_duplicate(sample):
                removed += 1
            else:
                unique.append(sample)

        self._removed_count = removed
        return unique, removed

    def reset(self):
        """重置去重状态"""
        self._seen_hashes.clear()
        self._removed_count = 0


# ============================================================
# 质量过滤 (Quality Filter)
# ============================================================

class QualityFilter:
    """
    代码样本质量过滤器

    过滤条件（可配置）:
      - 最小长度（字符数）
      - 最大长度（字符数）
      - 最小行数
      - 最大行数
      - 非 ASCII 字符比例上限
      - 重复行比例上限
      - 最小字母/数字比例
      - 黑名单正则（如过滤包含特定垃圾模式的行）
    """

    def __init__(self,
                 min_chars: int = 10,
                 max_chars: int = 100000,
                 min_lines: int = 1,
                 max_lines: int = 5000,
                 max_non_ascii_ratio: float = 0.3,
                 max_repeat_line_ratio: float = 0.5,
                 min_alphanum_ratio: float = 0.1,
                 blacklist_patterns: Optional[List[str]] = None):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.max_non_ascii_ratio = max_non_ascii_ratio
        self.max_repeat_line_ratio = max_repeat_line_ratio
        self.min_alphanum_ratio = min_alphanum_ratio
        self.blacklist = blacklist_patterns or []
        self._stats = Counter()

    @property
    def stats(self) -> Counter:
        return self._stats

    def _check_length(self, text: str) -> Tuple[bool, str]:
        """检查长度"""
        n_chars = len(text)
        if n_chars < self.min_chars:
            return False, f"too_short({n_chars}<{self.min_chars})"
        if n_chars > self.max_chars:
            return False, f"too_long({n_chars}>{self.max_chars})"
        return True, ""

    def _check_lines(self, text: str) -> Tuple[bool, str]:
        """检查行数"""
        lines = text.split("\n")
        n_lines = len(lines)
        if n_lines < self.min_lines:
            return False, f"too_few_lines({n_lines}<{self.min_lines})"
        if n_lines > self.max_lines:
            return False, f"too_many_lines({n_lines}>{self.max_lines})"
        return True, ""

    def _check_non_ascii(self, text: str) -> Tuple[bool, str]:
        """检查非 ASCII 字符比例"""
        if len(text) == 0:
            return False, "empty"
        non_ascii = sum(1 for c in text if ord(c) > 127)
        ratio = non_ascii / len(text)
        if ratio > self.max_non_ascii_ratio:
            return False, f"too_many_non_ascii({ratio:.2f}>{self.max_non_ascii_ratio})"
        return True, ""

    def _check_repeat_lines(self, text: str) -> Tuple[bool, str]:
        """检查重复行比例"""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) <= 1:
            return True, ""

        line_counts = Counter(lines)
        most_common = line_counts.most_common(1)[0]
        repeat_ratio = most_common[1] / len(lines)
        if repeat_ratio > self.max_repeat_line_ratio:
            return False, f"too_many_repeat_lines({repeat_ratio:.2f}>{self.max_repeat_line_ratio})"
        return True, ""

    def _check_alphanum(self, text: str) -> Tuple[bool, str]:
        """检查字母数字比例"""
        if len(text) == 0:
            return False, "empty"
        alphanum = sum(1 for c in text if c.isalnum())
        ratio = alphanum / len(text)
        if ratio < self.min_alphanum_ratio:
            return False, f"too_few_alphanum({ratio:.2f}<{self.min_alphanum_ratio})"
        return True, ""

    def _check_blacklist(self, text: str) -> Tuple[bool, str]:
        """检查黑名单模式"""
        for pattern in self.blacklist:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"blacklisted({pattern})"
        return True, ""

    def is_valid(self, text: str) -> Tuple[bool, str]:
        """
        检查样本是否通过所有质量过滤

        Returns:
            (是否通过, 失败原因)
        """
        checks = [
            ("length", self._check_length),
            ("lines", self._check_lines),
            ("non_ascii", self._check_non_ascii),
            ("repeat_lines", self._check_repeat_lines),
            ("alphanum", self._check_alphanum),
            ("blacklist", self._check_blacklist),
        ]

        for check_name, check_fn in checks:
            ok, reason = check_fn(text)
            if not ok:
                self._stats[f"fail_{check_name}"] += 1
                return False, f"{check_name}:{reason}"

        self._stats["pass"] += 1
        return True, "ok"

    def filter(self, samples: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """
        过滤样本列表

        Returns:
            (通过样本, 统计信息)
        """
        self._stats.clear()
        passed = []

        for sample in samples:
            ok, reason = self.is_valid(sample)
            if ok:
                passed.append(sample)
            else:
                self._stats[f"removed_{reason.split(':')[0]}"] += 1

        self._stats["total"] = len(samples)
        self._stats["passed"] = len(passed)
        self._stats["removed"] = len(samples) - len(passed)

        return passed, dict(self._stats)


# ============================================================
# 语法验证 (Syntax Validation)
# ============================================================

class SyntaxValidator:
    """
    代码语法验证器

    支持:
      - Python: 使用 ast 模块验证
      - 通用: 基本的括号匹配检查
    """

    def __init__(self, language: str = "python"):
        self.language = language.lower()
        self._valid_count = 0
        self._invalid_count = 0
        self._error_details: List[Tuple[str, str]] = []

    @property
    def valid_rate(self) -> float:
        total = self._valid_count + self._invalid_count
        return self._valid_count / total if total > 0 else 0.0

    def validate_python(self, code: str) -> Tuple[bool, str]:
        """使用 Python AST 验证语法"""
        try:
            compile(code, "<string>", "exec")
            return True, "ok"
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (line {e.lineno})"
        except Exception as e:
            return False, f"CompileError: {str(e)[:80]}"

    def validate_generic(self, code: str) -> Tuple[bool, str]:
        """通用括号匹配检查（适用于任何语言）"""
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack = []

        in_string = False
        in_single_string = False
        string_char = None

        for i, ch in enumerate(code):
            # 简单的字符串检测
            if ch == '"' and not in_single_string:
                if not in_string:
                    in_string = True
                    string_char = '"'
                elif string_char == '"':
                    in_string = False
            elif ch == "'" and not in_string:
                if not in_single_string:
                    in_single_string = True
                    string_char = "'"
                elif string_char == "'":
                    in_single_string = False

            if in_string or in_single_string:
                continue

            if ch in pairs:
                stack.append((ch, i))
            elif ch in pairs.values():
                if not stack:
                    return False, f"Extra closing '{ch}' at position {i}"
                opening, pos = stack.pop()
                if pairs[opening] != ch:
                    return False, f"Mismatched '{opening}'...'{ch}' at position {i}"

        if stack:
            return False, f"Unclosed '{stack[-1][0]}' at position {stack[-1][1]}"
        return True, "ok"

    def validate(self, code: str) -> Tuple[bool, str]:
        """
        验证代码语法

        Returns:
            (是否有效, 错误信息)
        """
        if self.language == "python":
            ok, msg = self.validate_python(code)
        else:
            ok, msg = self.validate_generic(code)

        if ok:
            self._valid_count += 1
        else:
            self._invalid_count += 1
            self._error_details.append((code[:80], msg))

        return ok, msg

    def batch_validate(self, samples: List[str]) -> Dict:
        """
        批量验证

        Returns:
            统计报告
        """
        self._valid_count = 0
        self._invalid_count = 0
        self._error_details.clear()

        for sample in samples:
            self.validate(sample)

        return {
            "total": len(samples),
            "valid": self._valid_count,
            "invalid": self._invalid_count,
            "valid_rate": self.valid_rate,
            "errors": self._error_details[:10],  # 最多保留 10 个错误样例
        }


# ============================================================
# 清洗管线 (Cleaning Pipeline)
# ============================================================

class DataCleaner:
    """
    数据清洗管线

    将去重、质量过滤、语法验证串联成一个完整的清洗流程。

    用法:
        cleaner = DataCleaner()
        cleaner.add_step(Deduplicator(method="exact"))
        cleaner.add_step(QualityFilter(min_chars=10))
        cleaner.add_step(SyntaxValidator(language="python"))
        report = cleaner.clean(input_file, output_file)
    """

    def __init__(self):
        self.steps: List[Tuple[str, Callable]] = []

    def add_step(self, name: str, processor):
        """添加处理步骤"""
        self.steps.append((name, processor))

    def clean_file(self, input_path: str, output_path: str = None,
                   verbose: bool = True) -> Dict:
        """
        清洗文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（None = 不保存）
            verbose: 是否输出进度

        Returns:
            清洗报告
        """
        # 读取
        with open(input_path, "r", encoding="utf-8") as f:
            samples = f.readlines()

        # 处理空行
        samples = [s for s in samples if s.strip()]
        original_count = len(samples)

        report = {
            "input_file": input_path,
            "original_samples": original_count,
            "steps": {},
            "final_samples": 0,
            "removed_total": 0,
        }

        if verbose:
            print(f"[Cleaner] Input: {original_count} samples from {input_path}")

        # 逐步处理
        current = samples
        for step_name, processor in self.steps:
            step_start = len(current)

            if isinstance(processor, Deduplicator):
                processor.reset()
                current, removed = processor.deduplicate(current)
                report["steps"][step_name] = {
                    "type": "dedup",
                    "before": step_start,
                    "after": len(current),
                    "removed": removed,
                }
                if verbose:
                    print(f"  [{step_name}] {step_start} → {len(current)} "
                          f"(-{removed} dups)")

            elif isinstance(processor, QualityFilter):
                current, stats = processor.filter(current)
                report["steps"][step_name] = {
                    "type": "quality_filter",
                    "before": step_start,
                    "after": len(current),
                    "removed": step_start - len(current),
                    "details": stats,
                }
                if verbose:
                    print(f"  [{step_name}] {step_start} → {len(current)} "
                          f"(-{step_start - len(current)} low quality)")

            elif isinstance(processor, SyntaxValidator):
                validation = processor.batch_validate(current)
                valid_rate = validation["valid_rate"]
                # 只保留语法有效的样本
                valid_samples = []
                invalid_count = 0
                for sample in current:
                    ok, _ = processor.validate(sample)
                    if ok:
                        valid_samples.append(sample)
                    else:
                        invalid_count += 1
                current = valid_samples
                report["steps"][step_name] = {
                    "type": "syntax",
                    "before": step_start,
                    "after": len(current),
                    "removed": invalid_count,
                    "valid_rate": f"{valid_rate:.1%}",
                }
                if verbose:
                    print(f"  [{step_name}] {step_start} → {len(current)} "
                          f"(-{invalid_count} invalid syntax, {valid_rate:.1%} valid)")

        report["final_samples"] = len(current)
        report["removed_total"] = original_count - len(current)
        report["retention_rate"] = (
            len(current) / original_count if original_count > 0 else 0
        )

        # 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in current:
                    f.write(sample.rstrip() + "\n")
            if verbose:
                print(f"\n[Cleaner] Saved {len(current)} samples to {output_path}")

        return report


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CodeSprite 数据清洗管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/data_cleaner.py --input data/train.txt --output data/train_clean.txt
  python tools/data_cleaner.py --input data/train.txt --no-syntax  # 跳过语法检查
  python tools/data_cleaner.py --input data/raw/ --output data/clean/ --lang python
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="输入文件/目录")
    parser.add_argument("--output", "-o", help="输出文件/目录（默认: 输入_clean）")
    parser.add_argument("--lang", default="python", help="代码语言 (default: python)")
    parser.add_argument("--min-chars", type=int, default=10, help="最小字符数")
    parser.add_argument("--max-chars", type=int, default=100000, help="最大字符数")
    parser.add_argument("--fuzzy-dedup", action="store_true", help="使用模糊去重")
    parser.add_argument("--no-syntax", action="store_true", help="跳过语法验证")
    parser.add_argument("--report-only", action="store_true",
                       help="仅生成报告，不保存清洗结果")

    args = parser.parse_args()

    # 构建管线
    cleaner = DataCleaner()

    dedup_method = "fuzzy" if args.fuzzy_dedup else "exact"
    cleaner.add_step("dedup", Deduplicator(method=dedup_method))

    cleaner.add_step("quality", QualityFilter(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    ))

    if not args.no_syntax:
        cleaner.add_step("syntax", SyntaxValidator(language=args.lang))

    # 确定输出路径
    output_path = args.output
    if not output_path and not args.report_only:
        base = os.path.splitext(args.input)[0]
        output_path = f"{base}_clean.txt"

    # 执行
    report = cleaner.clean_file(
        args.input,
        output_path=None if args.report_only else output_path,
        verbose=True,
    )

    # 打印报告
    print("\n" + "=" * 50)
    print("清洗报告")
    print("=" * 50)
    print(f"  输入样本数:  {report['original_samples']:,}")
    print(f"  输出样本数:  {report['final_samples']:,}")
    print(f"  移除总数:    {report['removed_total']:,}")
    print(f"  保留率:      {report['retention_rate']:.1%}")
    print("-" * 50)
    for step_name, step_info in report["steps"].items():
        before = step_info["before"]
        after = step_info["after"]
        removed = step_info.get("removed", before - after)
        pct = removed / before * 100 if before > 0 else 0
        print(f"  {step_name}: {before:,} → {after:,} (-{removed:,}, -{pct:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
