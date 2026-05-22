"""
评测模块 (Evaluation Module)

提供分层评测体系:
  - eval/metrics.py: 核心指标（编译率、语法正确率、编辑距离、BLEU代理）
  - eval/benchmark.py: 内置评测集和运行框架
"""

from eval.metrics import (
    check_syntax,
    compute_syntax_validity,
    compute_compile_rate,
    levenshtein_distance,
    compute_edit_distance_metrics,
    compute_ngram_match,
    compute_bleu_proxy,
    generate_eval_report,
)
from eval.benchmark import (
    COMPLETION_SAMPLES,
    CompletionResult,
    BenchmarkResult,
    CodeBenchmark,
)
