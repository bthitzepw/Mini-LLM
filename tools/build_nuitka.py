#!/usr/bin/env python
"""
CodeSprite Nuitka 打包脚本
=========================

将 CodeSprite 项目打包为独立的 Windows 可执行文件 (.exe)。

前置条件:
    pip install nuitka

用法:
    python tools/build_nuitka.py              # 标准构建
    python tools/build_nuitka.py --clean      # 清理后重新构建
    python tools/build_nuitka.py --console    # 保留控制台窗口（调试用）

输出:
    dist/codesprite.dist/  — 完整分发包
    dist/codesprite.exe    — 主入口（如果有 --standalone）

为什么用 Nuitka 而不是 PyInstaller:
    - Nuitka 将 Python 编译为 C 再编译为机器码，稳定性优于 PyInstaller 的 zip 解包方案
    - PyTorch 生态的动态库依赖在 Nuitka 下问题更少
    - 打包产物体积相当，但 Nuitka 产物启动更快

已知限制:
    - 仅支持 64-bit Windows（PyTorch 官方不支持 32 位）
    - 首次打包需 10-30 分钟（含 C 编译），后续增量打包 2-5 分钟
    - 产物约 500MB-1GB（主要是 PyTorch 动态库）
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"


def check_nuitka():
    """检查 Nuitka 是否已安装"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nuitka", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[OK] Nuitka version: {result.stdout.strip()}")
            return True
    except Exception:
        pass

    print("[ERROR] Nuitka not found. Install with: pip install nuitka")
    return False


def check_64bit():
    """确认是 64 位环境"""
    is_64 = sys.maxsize > 2**32
    if not is_64:
        print("[ERROR] CodeSprite only supports 64-bit Windows for packaging.")
        print("        PyTorch does not support 32-bit platforms.")
        sys.exit(1)
    print("[OK] 64-bit Python detected")


def clean_dist():
    """清理旧的构建产物"""
    if DIST_DIR.exists():
        print(f"[CLEAN] Removing {DIST_DIR}")
        shutil.rmtree(DIST_DIR)
        # 等待文件系统同步（Windows 下删除大目录需要时间）
        import time
        time.sleep(1)


def build(args):
    """执行 Nuitka 构建"""

    # 入口文件
    main_file = PROJECT_ROOT / "web_app.py"

    cmd = [
        sys.executable, "-m", "nuitka",

        # 基本模式
        "--standalone",             # 独立分发包（包含所有依赖）
        "--enable-plugin=pylint-warnings",  # 更好的错误提示
        "--assume-yes-for-downloads",       # 自动下载需要的依赖

        # 输出配置
        f"--output-dir={DIST_DIR}",

        # Windows 特定
        "--windows-disable-console",
        "--windows-icon-from-ico=",  # 暂不设置图标

        # 性能优化
        "--jobs=4",                 # 4 线程并行编译 C 代码

        # 包含关键模块（Nuitka 可能漏掉的动态导入）
        "--include-package=ir",
        "--include-package=backends",
        "--include-package=training",
        "--include-package=inference",
        "--include-package=src",
        "--include-package=export",
        "--include-package=tools",

        # 包含数据文件
        f"--include-data-dir={PROJECT_ROOT / 'config'}={PROJECT_ROOT / 'config'}",
        f"--include-data-dir={PROJECT_ROOT / 'templates'}={PROJECT_ROOT / 'templates'}",

        # 入口文件
        str(main_file),
    ]

    if args.console:
        # 保留控制台（调试用）
        cmd.remove("--windows-disable-console")
        cmd.append("--console")

    # 需要检查 configure 状态吗？需要这个 python flag 吗？
    # 这里不需要，因为是为了使用独立环境的 python 路径。

    print(f"[BUILD] Starting Nuitka build...")
    print(f"[BUILD] Command: {' '.join(cmd)}")
    print(f"[BUILD] This may take 10-30 minutes on first run.\n")

    os.chdir(PROJECT_ROOT)
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print("\n[SUCCESS] Build completed!")
        exe_dir = DIST_DIR / "web_app.dist"
        if exe_dir.exists():
            print(f"[OUTPUT] {exe_dir}")
            exe_file = exe_dir / "web_app.exe"
            if exe_file.exists():
                size_mb = exe_file.stat().st_size / (1024 * 1024)
                print(f"[OUTPUT] {exe_file} ({size_mb:.1f} MB)")
        print("\n[TIP] Copy the entire 'dist/web_app.dist/' folder to the target machine.")
        print("      双鸡 web_app.exe 即可启动 Web 服务。")
    else:
        print(f"\n[FAILED] Nuitka build exited with code {result.returncode}")
        print("[TIPS] Common issues:")
        print("  1. Too many files open → close other apps and retry")
        print("  2. Antivirus blocking → add dist/ to exclusions")
        print("  3. Disk space → need ~3GB free for build")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="CodeSprite Nuitka 打包工具 (64-bit Windows only)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="删除旧的 dist/ 目录后重新构建"
    )
    parser.add_argument(
        "--console", action="store_true",
        help="保留控制台窗口（调试用）"
    )
    args = parser.parse_args()

    # 前置检查
    check_64bit()
    if not check_nuitka():
        sys.exit(1)

    # 清理
    if args.clean:
        clean_dist()

    # 构建
    build(args)


if __name__ == "__main__":
    main()
