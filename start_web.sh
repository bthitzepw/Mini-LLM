#!/usr/bin/env bash
# ============================================
#   CodeSprite Web 服务启动脚本 (Linux/macOS)
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  CodeSprite Web 服务启动脚本${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 检查 Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}[错误] 未找到 Python，请先安装 Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}[✓]${NC} Python: $PYTHON_VERSION"

# 检查依赖
echo -e "${YELLOW}[...]${NC} 检查依赖..."
$PYTHON -c "import flask" 2>/dev/null && echo -e "${GREEN}[✓]${NC} Flask" || {
    echo -e "${RED}[✗] 缺少 Flask，请运行: pip install -r requirements.txt${NC}"
    exit 1
}

# 检查 CUDA（可选）
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "GPU")
    echo -e "${GREEN}[✓]${NC} CUDA: $GPU_NAME"
else
    echo -e "${YELLOW}[!]${NC} 未检测到 CUDA，将使用 CPU 模式"
fi

echo ""
# 解析命令行参数
PORT=5000
HOST="0.0.0.0"
DEBUG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --host|-h)
            HOST="$2"
            shift 2
            ;;
        --debug|-d)
            DEBUG="--debug"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -p, --port PORT    指定端口 (默认: 5000)"
            echo "  -h, --host HOST    指定主机 (默认: 0.0.0.0)"
            echo "  -d, --debug        开启调试模式"
            echo "  --help             显示帮助"
            exit 0
            ;;
        *)
            echo -e "${RED}[错误] 未知参数: $1${NC}"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}[启动]${NC} 正在启动 Web 服务..."
echo -e "  地址: ${CYAN}http://localhost:${PORT}${NC}"
echo -e "  按 Ctrl+C 停止服务"
echo ""

# 启动服务
$PYTHON web_app.py
