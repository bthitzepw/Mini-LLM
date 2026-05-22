"""
结构化日志系统

双通道输出：
  - 控制台：INFO 级别，简洁格式（时间 + 级别 + 消息）
  - 文件：DEBUG 级别，详细格式（时间 + 模块 + 级别 + 文件:行号 + 消息）
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
) -> logging.Logger:
    """创建双通道日志记录器

    Args:
        name: 日志记录器名称（通常传 __name__）
        log_dir: 日志文件目录
        level: 控制台输出级别（DEBUG / INFO / WARNING / ERROR）

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 总级别设为最低，由处理器各自过滤

    # 避免重复添加处理器（多次调用 setup_logger 时）
    logger.handlers = []

    # ---- 控制台处理器 (INFO) ----
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # ---- 文件处理器 (DEBUG) ----
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
