"""
评测指标模块 (Evaluation Metrics)

扩展 Beyond Perplexity — 从单一 PPL 到分层评测:
  - 基础层: PPL (已有 evaluate.py)
  - 代码质量层: 编译通过率 / 语法正确率 / 静态分析通过率
  - 能力层: 代码补全准确率 / 编辑距离
  - 工程层: 推理延迟 / 显存占用 (需要运行时环境)

用法:
  from eval.metrics import compute_compile_rate, compute_syntax_validity
  rate = compute_compile_rate(samples, language="python")
"""

import ast
import os
import re
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ============================================================
# 语法正确率 (Syntax Validity)
# ============================================================

def check_syntax(code: str, language: str = "python") -> Tuple[bool, str]:
    """
    检查代码语法是否正确

    Args:
        code: 代码文本
        language: 语言 (python)

    Returns:
        (is_valid, error_message)
    """
    if language == "python":
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError(line {e.lineno}): {e.msg}"
        except Exception as e:
            return False, f"ParseError: {str(e)[:100]}"
    else:
        # 通用检查仅做括号匹配
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for i, ch in enumerate(code):
            if ch in pairs:
                stack.append((ch, i))
            elif ch in pairs.values():
                if not stack:
                    return False, f"Extra closing bracket at pos {i}"
                opening, _ = stack.pop()
                if pairs[opening] != ch:
                    return False, f"Mismatched bracket at pos {i}"
        if stack:
            return False, f"Unclosed '{stack[-1][0]}' at pos {stack[-1][1]}"
        return True, ""


def compute_syntax_validity(samples: List[str], language: str = "python") -> Dict:
    """
    计算语法正确率

    Returns:
        {
            "total": int,
            "valid": int,
            "invalid": int,
            "valid_rate": float,
            "errors": [(sample_preview, error_msg), ...]  最多 10 条
        }
    """
    valid = 0
    invalid = 0
    errors = []

    for sample in samples:
        ok, msg = check_syntax(sample, language)
        if ok:
            valid += 1
        else:
            invalid += 1
            if len(errors) < 10:
                errors.append((sample[:80].replace("\n", " "), msg))

    total = valid + invalid
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_rate": valid / total if total > 0 else 0.0,
        "errors": errors,
    }


# ============================================================
# 编译通过率 (Compile Rate)
# ============================================================

def _try_compile_python(code: str) -> Tuple[bool, str]:
    """尝试编译 Python 代码"""
    try:
        compile(code, "<test>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError(line {e.lineno}): {e.msg}"
    except Exception as e:
        return False, f"CompileError: {str(e)[:80]}"


def compute_compile_rate(samples: List[str], language: str = "python") -> Dict:
    """
    计算可编译率

    对于 Python: 使用 compile() 内置函数
    对于其他语言: 如果系统安装了编译器，尝试调用

    Returns:
        {
            "total": int,
            "compiled": int,
            "failed": int,
            "compile_rate": float,
            "errors": [...]
        }
    """
    compiled = 0
    failed = 0
    errors = []

    if language == "python":
        for sample in samples:
            ok, msg = _try_compile_python(sample)
            if ok:
                compiled += 1
            else:
                failed += 1
                if len(errors) < 10:
                    errors.append((sample[:80].replace("\n", " "), msg))
    else:
        # 通用语言：降级到语法检查
        result = compute_syntax_validity(samples, language)
        compiled = result["valid"]
        failed = result["invalid"]
        errors = result.get("errors", [])

    total = compiled + failed
    return {
        "total": total,
        "compiled": compiled,
        "failed": failed,
        "compile_rate": compiled / total if total > 0 else 0.0,
        "errors": errors,
    }


# ============================================================
# 编辑距离 (Levenshtein Distance)
# ============================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Levenshtein 编辑距离

    衡量生成代码与参考代码之间的差异。
    值越小 → 越接近。
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_edit_distance_metrics(generated: List[str], references: List[str]) -> Dict:
    """
    计算生成代码相对参考代码的编辑距离指标

    Args:
        generated: 生成的代码列表
        references: 参考代码列表（长度需一致）

    Returns:
        {
            "avg_distance": float,
            "avg_normalized_distance": float,  # 归一化（距离/参考长度）
            "exact_match_rate": float,
            "pairs": int,
        }
    """
    if len(generated) != len(references):
        raise ValueError(f"generated({len(generated)}) and references({len(references)}) must have same length")

    distances = []
    normalized = []
    exact_matches = 0

    for gen, ref in zip(generated, references):
        d = levenshtein_distance(gen, ref)
        distances.append(d)
        normalized.append(d / max(len(ref), 1))
        if d == 0:
            exact_matches += 1

    n = len(distances)
    return {
        "avg_distance": sum(distances) / n if n > 0 else 0,
        "avg_normalized_distance": sum(normalized) / n if n > 0 else 0,
        "exact_match_rate": exact_matches / n if n > 0 else 0,
        "pairs": n,
        "min_distance": min(distances) if distances else 0,
        "max_distance": max(distances) if distances else 0,
    }


# ============================================================
# CodeBLEU (简化版)
# ============================================================

def compute_ngram_match(reference: str, candidate: str, n: int = 4) -> float:
    """
    N-gram 匹配率（简化版 CodeBLEU 的 N-gram 分量）

    注意：完整 CodeBLEU 需要 AST 匹配和数据流匹配，
    这里仅提供 N-gram 精确匹配作为快速代理指标。

    Returns:
        N-gram 匹配率 [0, 1.0]
    """
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    ref_ngrams = get_ngrams(reference, n)
    cand_ngrams = get_ngrams(candidate, n)

    if not cand_ngrams:
        return 0.0

    matched = 0
    for ngram, count in cand_ngrams.items():
        matched += min(count, ref_ngrams.get(ngram, 0))

    total = sum(cand_ngrams.values())
    return matched / total if total > 0 else 0.0


def compute_bleu_proxy(references: List[str], candidates: List[str],
                       max_n: int = 4) -> Dict:
    """
    简化版 BLEU 得分

    仅计算 N-gram 精确匹配（不含 brevity penalty 和 AST 匹配），
    作为代码相似度的快速近似。

    Returns:
        {
            "bleu_approx": float,   # 0-1 之间的近似 BLEU
            "ngram_scores": {1: float, 2: float, 3: float, 4: float},
            "avg_length_ratio": float,
        }
    """
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have same length")

    ngram_scores = {}
    for n in range(1, max_n + 1):
        n_matches = []
        for ref, cand in zip(references, candidates):
            n_matches.append(compute_ngram_match(ref, cand, n))
        ngram_scores[n] = sum(n_matches) / len(n_matches) if n_matches else 0

    # 几何平均
    product = 1.0
    for s in ngram_scores.values():
        product *= max(s, 0.01)
    bleu_approx = product ** (1 / max_n)

    # 长度比
    ref_lens = [len(r.split()) for r in references]
    cand_lens = [len(c.split()) for c in candidates]
    avg_len_ratio = (sum(cand_lens) / len(cand_lens)) / (sum(ref_lens) / len(ref_lens)) if ref_lens else 0

    return {
        "bleu_approx": bleu_approx,
        "ngram_scores": ngram_scores,
        "avg_length_ratio": avg_len_ratio,
    }


# ============================================================
# 综合评测报告
# ============================================================

def generate_eval_report(metrics: Dict) -> str:
    """
    生成 Markdown 格式的评测报告

    Args:
        metrics: 各评测模块返回的 metrics dict 的聚合

    Returns:
        Markdown 字符串
    """
    lines = [
        f"# CodeSprite 评测报告",
        f"",
        f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## 一、基础指标",
        f"",
    ]

    if "perplexity" in metrics:
        ppl = metrics["perplexity"]
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| Perplexity | {ppl:.2f} |")
        lines.append(f"| Val Loss | {metrics.get('val_loss', 'N/A')} |")
        lines.append(f"")

    if "compile_rate" in metrics:
        cr = metrics["compile_rate"]
        lines.append(f"## 二、代码质量")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 可编译率 | {cr:.1%} |")
        if "syntax_validity" in metrics:
            sv = metrics["syntax_validity"]
            lines.append(f"| 语法正确率 | {sv:.1%} |")
        lines.append(f"")

    if "edit_distance" in metrics:
        ed = metrics["edit_distance"]
        lines.append(f"## 三、文本匹配")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 平均编辑距离 | {ed['avg_distance']:.1f} |")
        lines.append(f"| 归一化编辑距离 | {ed['avg_normalized_distance']:.3f} |")
        lines.append(f"| 精确匹配率 | {ed['exact_match_rate']:.1%} |")
        lines.append(f"")

    return "\n".join(lines)


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    "check_syntax",
    "compute_syntax_validity",
    "compute_compile_rate",
    "levenshtein_distance",
    "compute_edit_distance_metrics",
    "compute_ngram_match",
    "compute_bleu_proxy",
    "generate_eval_report",
]
