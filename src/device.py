"""
统一设备管理模块
----------------
负责训练和推理的设备检测、显式日志、回退策略、资源隔离。

设计原则：
  - 显式检测 GPU 可用性，不靠异常捕获默默回退
  - 启动时打印设备断言/日志，杜绝"以为 GPU 训，实际 CPU"
  - 支持环境变量 CODESPRITE_ALLOW_CPU_FALLBACK 控制回退策略
  - CPU 推理时自动限制线程数，避免吃满全部核心

用法:
    from src.device import resolve_device, print_device_info

    device = resolve_device("auto")        # 自动选择最优设备
    device = resolve_device("cuda")        # 强制 GPU（不可用时按策略回退/报错）
    device = resolve_device("cpu")         # 强制 CPU
    print_device_info(device)              # 打印设备详情到日志
"""

import os
import sys
import logging

logger = logging.getLogger("CodeSprite")


def _detect_cuda() -> bool:
    """检测 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_mps() -> bool:
    """检测 Apple MPS 是否可用"""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def _get_cuda_info() -> dict:
    """获取 CUDA 设备详细信息"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_mem / (1024**3),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass
    return {}


def _limit_cpu_threads(cpu_threads: int = None):
    """
    CPU 资源隔离：限制线程数，避免推理吃掉全部核心。

    参数:
        cpu_threads: 显式指定线程数。None=留一半核心（至少2个）。
                     老电脑建议设为 2，牺牲速度换不卡顿。

    环境变量 OMP_NUM_THREADS 优先级高于此参数。
    """
    if os.environ.get("OMP_NUM_THREADS") is not None:
        # 用户已通过环境变量显式设置，尊重用户选择
        threads = int(os.environ["OMP_NUM_THREADS"])
    elif cpu_threads is not None:
        threads = cpu_threads
    else:
        cpu_count = os.cpu_count() or 4
        threads = max(2, cpu_count // 2)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def _allow_cpu_fallback() -> bool:
    """读取环境变量判断是否允许 CPU 回退
    环境变量: CODESPRITE_ALLOW_CPU_FALLBACK=true/false（默认 true）
    """
    val = os.environ.get("CODESPRITE_ALLOW_CPU_FALLBACK", "true").strip().lower()
    return val in ("true", "1", "yes", "on")


def resolve_device(target: str = "auto", *, allow_fallback: bool = None,
                   cpu_threads: int = None) -> str:
    """
    解析目标设备，返回 "cuda" / "cpu" / "mps" 字符串。

    参数:
        target: "auto" (自动最优) / "cuda" / "cpu" / "mps"
        allow_fallback: 是否允许在 GPU 不可用时回退到 CPU。
                        None 时读取环境变量 CODESPRITE_ALLOW_CPU_FALLBACK。
        cpu_threads: CPU 模式下的线程数。None=留一半核心。老电脑建议设为 2。

    返回值:
        str: "cuda" / "cpu" / "mps"

    异常:
        RuntimeError: 请求 GPU 但不可用且不允许回退时抛出
    """
    if allow_fallback is None:
        allow_fallback = _allow_cpu_fallback()

    target = target.strip().lower()

    # --- CPU 模式：直接返回 ---
    if target == "cpu":
        _limit_cpu_threads(cpu_threads)
        return "cpu"

    # --- MPS 模式（Apple Silicon）---
    if target == "mps":
        if _detect_mps():
            return "mps"
        if allow_fallback:
            logger.warning(
                "Apple MPS not available — falling back to CPU. "
                "Set CODESPRITE_ALLOW_CPU_FALLBACK=false to prevent this."
            )
            _limit_cpu_threads(cpu_threads)
            return "cpu"
        raise RuntimeError(
            "Apple MPS requested but not available. "
            "Install PyTorch with MPS support or use --device cpu. "
            "Or set CODESPRITE_ALLOW_CPU_FALLBACK=true to allow CPU fallback."
        )

    # --- CUDA / auto 模式 ---
    if target == "cuda" or target == "auto":
        if _detect_cuda():
            return "cuda"

        # GPU 不可用 — 决定是否回退
        if target == "cuda":
            reason = (
                "CUDA device explicitly requested but not available. "
                "Check: 1) NVIDIA driver installed? "
                "2) CUDA toolkit installed? "
                "3) PyTorch compiled with CUDA support? "
            )
            if allow_fallback:
                logger.warning(
                    reason + "Falling back to CPU. "
                    "Set CODESPRITE_ALLOW_CPU_FALLBACK=false to make this a hard error."
                )
                _limit_cpu_threads(cpu_threads)
                return "cpu"
            raise RuntimeError(
                reason + "Set --device cpu to train on CPU, "
                "or set CODESPRITE_ALLOW_CPU_FALLBACK=true to allow CPU fallback."
            )

        # "auto" 模式：GPU 不可用就默默走 CPU（这是合理的默认行为）
        logger.info("No GPU detected — using CPU.")
        _limit_cpu_threads(cpu_threads)
        return "cpu"

    # 未知 target
    raise ValueError(f"Unknown device target: '{target}'. Use 'auto', 'cuda', 'cpu', or 'mps'.")


def print_device_info(device_str: str):
    """
    打印设备可观测性信息（启动日志）。
    调用时机：训练/推理入口，在创建 backend 前。
    """
    separator = "=" * 56

    if device_str == "cuda":
        info = _get_cuda_info()
        lines = [
            separator,
            "  DEVICE: GPU (CUDA)",
            f"  GPU:    {info.get('name', 'unknown')}",
            f"  Memory: {info.get('memory_gb', 0):.1f} GB",
            f"  CUDA:   {info.get('cuda_version', 'unknown')}",
            f"  Count:  {info.get('count', 1)} device(s)",
            f"  CPU fallback: {'ALLOWED' if _allow_cpu_fallback() else 'DISABLED'} "
            f"(CODESPRITE_ALLOW_CPU_FALLBACK)",
            separator,
        ]
        print("\n".join(lines))
        logger.info(f"Training device: cuda (GPU: {info.get('name', 'unknown')})")

    elif device_str == "mps":
        lines = [
            separator,
            "  DEVICE: GPU (Apple MPS)",
            "  Backend: Metal Performance Shaders",
            f"  CPU fallback: {'ALLOWED' if _allow_cpu_fallback() else 'DISABLED'} "
            f"(CODESPRITE_ALLOW_CPU_FALLBACK)",
            separator,
        ]
        print("\n".join(lines))
        logger.info("Training device: mps (Apple MPS)")

    elif device_str == "cpu":
        threads = os.environ.get("OMP_NUM_THREADS", "auto")
        lines = [
            separator,
            "  DEVICE: CPU",
            f"  Threads: {threads} (OMP_NUM_THREADS)",
            f"  Cores:   {os.cpu_count() or 'unknown'}",
            separator,
        ]
        print("\n".join(lines))
        logger.info(f"Training device: cpu (threads={threads})")

    else:
        print(f"  DEVICE: {device_str}")


def warn_cpu_training():
    """CPU 训练风险警告"""
    msg = (
        "\n"
        "  +-------------------------------------------------+\n"
        "  |  WARNING: Training on CPU                        |\n"
        "  |  - Training will be SLOW (10-50x vs GPU)         |\n"
        "  |  - Large models may OOM or freeze the machine    |\n"
        "  |  - Consider using a smaller model or GPU         |\n"
        "  |  - Set CODESPRITE_ALLOW_CPU_FALLBACK=false       |\n"
        "  |    to abort when GPU is unavailable              |\n"
        "  +-------------------------------------------------+\n"
    )
    print(msg)
    logger.warning("Training on CPU — expect slow performance")
