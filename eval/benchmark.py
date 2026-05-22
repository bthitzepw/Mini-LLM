"""
轻量级代码评测集 (Benchmark Framework)

内置小型评测集，适合在本地快速评估代码分析引擎的能力。
不依赖外部数据集下载。

评测任务:
  - 代码补全 (Code Completion)
  - 语法检查 (Syntax Check)
  - 编辑距离 (Edit Distance)

用法:
  from eval.benchmark import CodeBenchmark
  bm = CodeBenchmark()
  results = bm.run_all(model, tokenizer, backend)
  print(results.summary())
"""

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field


# ============================================================
# 内置评测样本
# ============================================================

# 代码补全评测样本: (prefix, expected_suffix, task_name)
COMPLETION_SAMPLES = [
    # Python 基础
    ("def add(a, b):\n    ", "return a + b", "python_add"),
    ("def factorial(n):\n    if n <= 1:\n        return 1\n    ", "return n * factorial(n - 1)", "python_factorial"),
    ("def is_even(x):\n    ", "return x % 2 == 0", "python_is_even"),
    ("def max_of_three(a, b, c):\n    ", "return max(a, b, c)", "python_max3"),
    ("def greet(name):\n    ", "return f\"Hello, {name}!\"", "python_greet"),

    # Python 中等
    ("def reverse_string(s):\n    ", "return s[::-1]", "python_reverse"),
    ("def count_vowels(s):\n    vowels = \"aeiou\"\n    ", "return sum(1 for c in s.lower() if c in vowels)", "python_vowels"),
    ("def filter_positive(nums):\n    ", "return [n for n in nums if n > 0]", "python_filter"),
    ("def square_list(nums):\n    ", "return [n ** 2 for n in nums]", "python_square"),
    ("def sum_list(nums):\n    ", "return sum(nums)", "python_sum"),

    # 通用编程
    ("def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    ", "result.extend(a[i:])\n    result.extend(b[j:])\n    return result", "merge_sorted"),
    ("def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    ", "return -1", "binary_search"),
]


@dataclass
class CompletionResult:
    """单个补全结果"""
    task_name: str
    prefix: str
    expected: str
    generated: str
    exact_match: bool
    edit_distance: int
    char_count: int


@dataclass
class BenchmarkResult:
    """评测结果汇总"""
    task_type: str
    total: int = 0
    exact_matches: int = 0
    avg_edit_distance: float = 0.0
    results: List[CompletionResult] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"{self.task_type} Benchmark: {self.total} tasks",
            f"  Exact Match: {self.exact_matches}/{self.total} ({self.exact_match_rate:.1%})",
            f"  Avg Edit Dist: {self.avg_edit_distance:.1f}",
        ]
        return "\n".join(lines)


class CodeBenchmark:
    """
    代码评测框架

    用法:
        bm = CodeBenchmark()
        # 逐个评测
        for prefix, expected, name in bm.completion_tasks():
            generated = engine.generate(prefix, max_new_tokens=20)
            bm.record_completion(name, prefix, expected, generated)
        result = bm.get_result()
        print(result.summary())
    """

    def __init__(self):
        self._completion_result = BenchmarkResult(task_type="Code Completion")
        self._syntax_result = BenchmarkResult(task_type="Syntax")

    def completion_tasks(self) -> List[Tuple[str, str, str]]:
        """返回所有补全任务: (prefix, expected, name)"""
        return list(COMPLETION_SAMPLES)

    def record_completion(self, task_name: str, prefix: str,
                          expected: str, generated: str):
        """记录一次补全结果"""
        from eval.metrics import levenshtein_distance

        gen_clean = generated[len(prefix):] if generated.startswith(prefix) else generated
        gen_clean = gen_clean.strip()
        expected_clean = expected.strip()

        d = levenshtein_distance(gen_clean, expected_clean)
        exact = gen_clean == expected_clean

        result = CompletionResult(
            task_name=task_name,
            prefix=prefix,
            expected=expected_clean,
            generated=gen_clean,
            exact_match=exact,
            edit_distance=d,
            char_count=len(gen_clean),
        )

        self._completion_result.results.append(result)
        self._completion_result.total = len(self._completion_result.results)
        if exact:
            self._completion_result.exact_matches += 1

        # 更新平均编辑距离
        dists = [r.edit_distance for r in self._completion_result.results]
        self._completion_result.avg_edit_distance = sum(dists) / len(dists) if dists else 0.0

    def run_completion(self, engine, tokenizer, max_new_tokens: int = 30,
                       temperature: float = 0.8) -> BenchmarkResult:
        """
        运行完整补全评测

        Args:
            engine: InferenceEngine 实例
            tokenizer: tokenizer 实例
            max_new_tokens: 每个任务的最大生成 token 数

        Returns:
            BenchmarkResult
        """
        self._completion_result = BenchmarkResult(task_type="Code Completion")

        for prefix, expected, name in COMPLETION_SAMPLES:
            try:
                if hasattr(engine, 'generate_with_kv_cache'):
                    generated = engine.generate_with_kv_cache(
                        prefix, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=50, top_p=0.9
                    )
                else:
                    generated = engine.generate(
                        prefix, max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
            except Exception as e:
                generated = f"ERROR: {e}"

            self.record_completion(name, prefix, expected, generated)

        return self._completion_result

    def get_result(self) -> BenchmarkResult:
        return self._completion_result

    def summary(self) -> str:
        lines = ["=" * 50, "CodeSprite Benchmark Report", "=" * 50]

        if self._completion_result.total > 0:
            lines.append(self._completion_result.summary())
            lines.append("")
            lines.append("Per-task results:")
            for r in self._completion_result.results:
                status = "OK" if r.exact_match else f"ED={r.edit_distance}"
                lines.append(f"  [{status:<8}] {r.task_name}: '{r.generated[:40]}...'")

        return "\n".join(lines)


# ============================================================
# 模块导出
# ============================================================

__all__ = [
    "COMPLETION_SAMPLES",
    "CompletionResult", "BenchmarkResult",
    "CodeBenchmark",
]
