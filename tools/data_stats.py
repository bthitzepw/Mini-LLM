#!/usr/bin/env python
"""
数据统计报告 (Data Statistics Report)

对代码数据集进行统计分析，生成结构化的 Markdown 报告。

统计维度:
  - 样本总量、总字符数、总 token 数（如果提供 tokenizer）
  - 长度分布（分位数、直方图）
  - 语言分布（如果样本带语言标记）
  - 样本质量分布（空行率、注释率、非 ASCII 比例）

用法:
  python tools/data_stats.py --input data/train.txt
  python tools/data_stats.py --input data/ --output report.md
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional


def compute_basic_stats(samples: List[str]) -> Dict:
    """计算基础统计"""
    n = len(samples)
    if n == 0:
        return {"total_samples": 0, "total_chars": 0, "total_lines": 0}

    lengths = [len(s) for s in samples]
    lengths_sorted = sorted(lengths)

    # 行数统计
    line_counts = [len(s.split("\n")) for s in samples]

    return {
        "total_samples": n,
        "total_chars": sum(lengths),
        "avg_chars": sum(lengths) / n,
        "max_chars": max(lengths),
        "min_chars": min(lengths),
        "total_lines": sum(line_counts),
        "avg_lines": sum(line_counts) / n,
        # 分位数
        "p10": lengths_sorted[int(n * 0.10)] if n >= 10 else lengths_sorted[0],
        "p25": lengths_sorted[int(n * 0.25)] if n >= 4 else lengths_sorted[0],
        "p50": lengths_sorted[int(n * 0.50)],
        "p75": lengths_sorted[int(n * 0.75)],
        "p90": lengths_sorted[int(n * 0.90)] if n >= 10 else lengths_sorted[-1],
        "p95": lengths_sorted[int(n * 0.95)] if n >= 20 else lengths_sorted[-1],
    }


def compute_length_histogram(samples: List[str], bins: int = 10) -> List[Tuple[str, int]]:
    """生成长度直方图"""
    lengths = [len(s) for s in samples]
    if not lengths:
        return []

    min_len, max_len = min(lengths), max(lengths)
    if min_len == max_len:
        return [(f"{min_len}-{min_len}", len(samples))]

    bin_width = max((max_len - min_len) // bins, 1)
    histogram = Counter()

    for l in lengths:
        bucket = (l // bin_width) * bin_width
        histogram[f"{bucket}-{bucket + bin_width - 1}"] += 1

    return sorted(histogram.items(), key=lambda x: int(x[0].split("-")[0]))


def compute_quality_distribution(samples: List[str]) -> Dict:
    """分析样本质量分布"""
    empty_line_ratios = []
    comment_ratios = []
    non_ascii_ratios = []
    empty_samples = 0

    for s in samples:
        if not s.strip():
            empty_samples += 1
            continue

        lines = s.split("\n")
        total_lines = len(lines)

        # 空行比例
        empty_lines = sum(1 for line in lines if not line.strip())
        empty_line_ratios.append(empty_lines / total_lines if total_lines > 0 else 0)

        # 注释比例（Python style: #, 通用: //）
        comment_lines = sum(1 for line in lines
                          if line.strip().startswith("#") or line.strip().startswith("//"))
        comment_ratios.append(comment_lines / total_lines if total_lines > 0 else 0)

        # 非 ASCII 比例
        if len(s) > 0:
            non_ascii = sum(1 for c in s if ord(c) > 127)
            non_ascii_ratios.append(non_ascii / len(s))

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "empty_sample_count": empty_samples,
        "avg_empty_line_ratio": safe_avg(empty_line_ratios),
        "avg_comment_ratio": safe_avg(comment_ratios),
        "avg_non_ascii_ratio": safe_avg(non_ascii_ratios),
    }


def detect_language_breakdown(samples: List[str]) -> Dict[str, int]:
    """
    粗略检测语言分布

    基于文件扩展名和代码特征。
    """
    lang_counter = Counter()

    for s in samples:
        s_lower = s.lower()
        # Python 特征
        if "def " in s or "import " in s or "class " in s:
            if "from " in s or "print(" in s:
                lang_counter["python"] += 1
                continue
        # JavaScript/TypeScript 特征
        if "function " in s or "const " in s or "let " in s or "=>" in s:
            if "console.log" in s or "export " in s:
                lang_counter["javascript"] += 1
                continue
        # Java 特征
        if "public class" in s or "public static" in s or "System.out" in s:
            lang_counter["java"] += 1
            continue
        # C/C++ 特征
        if "#include" in s or "int main" in s:
            lang_counter["c/c++"] += 1
            continue
        # Go
        if "func " in s and "package " in s:
            lang_counter["go"] += 1
            continue
        # HTML
        if "<html" in s_lower or "<div" in s_lower:
            lang_counter["html"] += 1
            continue
        # 未识别
        lang_counter["unknown"] += 1

    return dict(lang_counter)


def generate_report(input_path: str, samples: List[str],
                   token_count: Optional[int] = None) -> str:
    """
    生成 Markdown 格式统计报告
    """
    basic = compute_basic_stats(samples)
    hist = compute_length_histogram(samples)
    quality = compute_quality_distribution(samples)
    langs = detect_language_breakdown(samples)

    lines = []
    lines.append(f"# 数据集统计报告")
    lines.append(f"")
    lines.append(f"**数据源**: `{input_path}`")
    lines.append(f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")

    # 基础统计
    lines.append(f"## 一、基础统计")
    lines.append(f"")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 样本总数 | {basic['total_samples']:,} |")
    lines.append(f"| 总字符数 | {basic['total_chars']:,} |")
    lines.append(f"| 总行数 | {basic['total_lines']:,} |")
    lines.append(f"| 平均字符数/样本 | {basic['avg_chars']:.1f} |")
    lines.append(f"| 平均行数/样本 | {basic['avg_lines']:.1f} |")
    lines.append(f"| 最小字符数 | {basic['min_chars']:,} |")
    lines.append(f"| 最大字符数 | {basic['max_chars']:,} |")
    if token_count:
        lines.append(f"| 预估 Token 数 | {token_count:,} |")
    lines.append(f"")

    # 长度分位数
    lines.append(f"## 二、长度分布")
    lines.append(f"")
    lines.append(f"| 分位数 | 字符数 |")
    lines.append(f"|--------|--------|")
    for p in ["p10", "p25", "p50", "p75", "p90", "p95"]:
        lines.append(f"| {p.upper()} | {basic[p]:,} |")
    lines.append(f"")

    # 直方图
    if hist:
        lines.append(f"### 长度直方图")
        lines.append(f"")
        max_count = max(c for _, c in hist)
        for bucket, count in hist:
            bar = "█" * min(int(count / max(max_count, 1) * 40), 40)
            lines.append(f"| `{bucket:>15}` | {bar} {count} |")
        lines.append(f"")

    # 质量分布
    lines.append(f"## 三、质量分布")
    lines.append(f"")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 空样本数 | {quality['empty_sample_count']} |")
    lines.append(f"| 平均空行比例 | {quality['avg_empty_line_ratio']:.1%} |")
    lines.append(f"| 平均注释比例 | {quality['avg_comment_ratio']:.1%} |")
    lines.append(f"| 平均非 ASCII 比例 | {quality['avg_non_ascii_ratio']:.1%} |")
    lines.append(f"")

    # 语言分布
    lines.append(f"## 四、语言分布")
    lines.append(f"")
    lines.append(f"| 语言 | 样本数 | 占比 |")
    lines.append(f"|------|--------|------|")
    total = sum(langs.values())
    for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"| {lang} | {count:,} | {pct:.1f}% |")
    lines.append(f"| **合计** | **{total:,}** | **100%** |")
    lines.append(f"")

    # 健康评估
    lines.append(f"## 五、健康评估")
    lines.append(f"")

    issues = []
    if basic['total_samples'] < 1000:
        issues.append(f"- ⚠️ 样本量偏小（{basic['total_samples']}），建议扩充到 10,000+ 以获得稳定训练")
    if quality['avg_empty_line_ratio'] > 0.3:
        issues.append(f"- ⚠️ 空行比例偏高（{quality['avg_empty_line_ratio']:.1%}），建议清理空行")
    if quality['avg_non_ascii_ratio'] > 0.1:
        issues.append(f"- ⚠️ 非 ASCII 字符比例偏高（{quality['avg_non_ascii_ratio']:.1%}），可能含非代码内容")
    if langs.get("unknown", 0) / max(total, 1) > 0.5:
        issues.append(f"- ⚠️ 大量样本语言未识别，可能数据格式不统一")

    if issues:
        for issue in issues:
            lines.append(issue)
    else:
        lines.append(f"- ✅ 数据集整体健康，无明显问题")
    lines.append(f"")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="CodeSprite 数据统计报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python tools/data_stats.py --input data/train.txt --output report.md",
    )
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", help="输出 Markdown 文件（默认: stdout）")
    parser.add_argument("--tokenizer", help="tokenizer 类型（暂未实现）")

    args = parser.parse_args()

    # 读取数据
    with open(args.input, "r", encoding="utf-8") as f:
        raw = f.read()

    samples = [s for s in raw.split("\n\n") if s.strip()]
    if len(samples) == 0:
        # Fallback: 按行分割
        samples = [s for s in raw.split("\n") if s.strip()]

    print(f"读取 {len(samples)} 个样本从 {args.input}")

    # 生成报告
    report = generate_report(args.input, samples)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存到 {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
