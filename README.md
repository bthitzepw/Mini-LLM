# CodeSprite — 框架无关 IR 架构的微型代码语言模型

> **架构升级：CodeSprite 采用框架无关的 IR（中间表示）架构**
>
> 模型定义层（`ir/`）不依赖任何计算框架（PyTorch/NumPy），计算由可插拔的 `backends/` 后端提供。
> 同一份模型代码可以：用 PyTorch 训练，用 NumPy 纯 CPU 推理，导出为 GGUF/ONNX 格式。

约 3800 万参数，Transformer Decoder 架构，专注于代码编程领域，支持中英文双语。
集成 RoPE 旋转位置编码、SwiGLU 门控前馈网络、GQA 分组查询注意力、KV-Cache 推理加速等现代技术。

---

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [准备训练数据](#准备训练数据)
- [训练模型](#训练模型)
- [使用模型](#使用模型)
- [Web 服务](#web-服务)
- [训练配置说明](#训练配置说明)
- [API 接口文档](#api-接口文档)
- [自动学习系统](#自动学习系统)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [License](#license)

---

## 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| Python | 3.10+ | 3.12 |
| PyTorch | 2.0+ | 2.8+ (CUDA 12.x) |
| GPU 显存 | - | >= 2 GB（训练时约需 0.5 GB） |
| 操作系统 | Windows / Linux / macOS | - |
| CUDA | - | 11.8+（GPU 加速） |

> **注意**：没有 GPU 也可以运行，程序会自动回退到 CPU 模式，但速度会慢很多。

### 安装 PyTorch

先安装与你系统匹配的 PyTorch 版本，参考官网：https://pytorch.org/get-started/locally/

```bash
# 示例：Windows + CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 示例：CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 安装项目依赖

```bash
git clone https://github.com/bthitzepw/codesprite.git
cd codesprite
pip install -r requirements.txt
```

`requirements.txt` 包含以下依赖：

```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.14.0
flask>=2.0.0
```

---

## 快速开始

### 最快上手（3 步）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（见下一节），然后训练
python train.py

# 3. 训练完成后，交互式生成
python generate.py
```

### 仅推理（使用预训练权重）

如果你有训练好的 `best_model.pt` 文件：

```bash
# 将权重文件放入 checkpoints 目录
mkdir -p checkpoints
cp your_best_model.pt checkpoints/best_model.pt

# 直接开始对话
python generate.py
```

---

## 准备训练数据

训练数据放在 `data/raw/` 目录下，需要三个文件：

```
data/raw/
├── train.txt    # 训练集（必需）
├── val.txt      # 验证集（必需）
└── test.txt     # 测试集（评估时使用）
```

### 数据格式

每行一条训练样本，支持中英文混合文本。本项目聚焦代码编程领域，建议使用以下类型的数据：

```
def fibonacci(n):
    """计算斐波那契数列第n项"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 快速排序算法
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

### 数据建议

- **训练集**：建议 1000 条以上，越多越好
- **验证集**：建议 50-100 条，用于监控过拟合
- **测试集**：建议 30-50 条，用于最终评估
- **质量**：优先数据质量而非数量，确保代码正确、格式规范
- **多样性**：覆盖多种编程语言（Python / JavaScript / Java / C++ / Go / Rust / SQL 等）

### 自定义 BPE 分词器

本项目使用内置的 `SimpleTokenizer`（字符级 BPE，词表大小 4268），支持 ASCII 字符、CJK 汉字和常见代码符号。无需额外预训练分词器，开箱即用。

如需调整词表大小，修改 `config/config.yaml` 中的 `vocab_size`。

---

## 训练模型

### 基础训练

```bash
python train.py
```

使用 `config/config.yaml` 中的默认配置开始训练。

### 命令行参数

```bash
# 启用 EMA（推荐用于追求推理稳定性）
python train.py --use-ema

# 禁用混合精度（调试时使用，或显存不足时）
python train.py --no-amp

# 启用梯度检查点（用计算换显存）
python train.py --use-checkpointing

# 禁用 RoPE 或 SwiGLU（用于消融实验）
python train.py --no-rope
python train.py --no-swiglu

# 自定义超参数
python train.py --lr 0.001 --epochs 20 --batch-size 32

# 设置标签平滑值
python train.py --label-smoothing 0.1
```

### 参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `standard` | 训练模式：`standard` / `auto` / `find-lr` |
| `--lr` | 0.0003 | 学习率（覆盖配置文件） |
| `--epochs` | 10 | 训练轮数 |
| `--batch-size` | 16 | 批量大小 |
| `--no-amp` | False | 禁用混合精度训练 |
| `--no-rope` | False | 禁用 RoPE 位置编码 |
| `--no-swiglu` | False | 禁用 SwiGLU 前馈网络 |
| `--use-ema` | False | 启用指数移动平均 |
| `--use-checkpointing` | False | 启用梯度检查点 |
| `--label-smoothing` | 0.05 | 标签平滑系数 |

### 训练输出

训练过程中会输出每个 epoch 的训练损失和验证指标：

```
Epoch 6/10 Summary:
  Train Loss: 4.2698
  Val Loss: 3.2807
  Perplexity: 26.60
  Learning Rate: 6.00e-05
  Epoch Time: 15.5s
  Total Time: 94.1s
  >> New best model! Val Loss: 3.2807
```

- **Train Loss**：训练集上的平均损失，越低越好
- **Val Loss**：验证集上的损失，用于判断是否过拟合
- **Perplexity**：困惑度（PPL = e^loss），越低表示模型越确信
  - < 10：优秀
  - < 30：良好
  - < 100：一般
  - > 500：需要更多训练
- **Early Stopping**：验证损失连续 5 个 epoch 没有改善时自动停止训练

### 检查点保存

训练完成后，模型权重保存在 `checkpoints/` 目录：

```
checkpoints/
├── best_model.pt              # 验证损失最低的模型（推荐使用这个）
├── checkpoint_epoch_8.pt      # 第 8 轮检查点
├── checkpoint_epoch_9.pt      # 第 9 轮检查点
└── checkpoint_epoch_10.pt     # 第 10 轮检查点
```

> **注意**：`checkpoints/` 目录和 `.pt` 文件已在 `.gitignore` 中排除，不会上传到 GitHub。如需分享模型权重，建议使用 [Hugging Face Hub](https://huggingface.co/)、Google Drive 等平台。

### 从已有检查点继续训练

将 `best_model.pt` 放入 `checkpoints/` 目录后，直接运行 `python train.py` 即可自动加载并继续训练。

### 学习率查找器

自动寻找最优学习率：

```bash
python train.py --find-lr
```

程序会绘制 loss-lr 曲线并建议最佳学习率。

---

## 使用模型

### 交互式生成

```bash
# PyTorch 后端（默认）
python generate.py

# NumPy 后端（纯 CPU，无需 PyTorch）
python generate.py --backend numpy

# 指定 GPU 设备
python generate.py --device cuda
```

进入交互模式后：

```
You: def fibonacci(n):
Model: def fibonacci(n):
    """计算斐波那契数列第n项"""
    if n <= 1:
        return n
    ...

You: :temp 0.5
Temperature set to 0.5

You: quit
Goodbye!
```

#### 交互命令

| 命令 | 说明 |
|------|------|
| `<文本>` | 输入提示，生成续写 |
| `:temp <n>` | 设置温度（0.1-2.0，越低越确定性） |
| `:topk <n>` | 设置 Top-K（1-200） |
| `:topp <n>` | 设置 Top-P（0.0-1.0，nucleus 采样） |
| `:len <n>` | 设置最大生成长度（10-500） |
| `:info` | 显示模型信息 |
| `quit` / `exit` / `q` | 退出 |

### 模型评估

```bash
python evaluate.py
```

输出包含：
- **Perplexity（困惑度）**：核心指标
- **Token-level Loss**：平均损失
- **生成质量抽检**：对 8 个预设代码提示进行生成测试
- **质量评级**：根据 PPL 自动评级

---

## Web 服务

### 启动

```bash
python web_app.py
```

浏览器打开 http://localhost:5000

### 功能

- 网页对话界面（带反馈按钮）
- 内容审核与安全过滤
- AI 生成内容标识
- 用户协议与隐私政策页面
- 学习状态控制面板

### 一键启动（Windows）

双击 `start_web.bat` 即可启动 Web 服务。

---

## 训练配置说明

所有配置集中在 `config/config.yaml`：

```yaml
model:
  vocab_size: 4268           # 词表大小
  hidden_size: 512           # 隐藏层维度
  num_layers: 8              # Transformer 层数
  num_heads: 8               # 注意力头数
  intermediate_size: 2048    # FFN 中间层维度
  dropout: 0.1               # Dropout 比率
  max_seq_length: 512        # 最大序列长度
  tie_weights: true          # Embedding 与 LM Head 权重共享
  use_rope: true             # RoPE 旋转位置编码
  use_swiglu: true           # SwiGLU 前馈网络

training:
  batch_size: 16             # 批量大小
  learning_rate: 0.0003      # 学习率
  num_epochs: 10             # 训练轮数
  warmup_steps: 500          # 预热步数
  gradient_accumulation_steps: 4  # 梯度累积步数（等效 batch = 16×4 = 64）
  max_grad_norm: 1.0         # 梯度裁剪
  weight_decay: 0.01         # 权重衰减
  use_amp: true              # 混合精度（FP16 加速 + 节省显存）
  label_smoothing: 0.05      # 标签平滑（防止过拟合）
  use_ema: false             # EMA（指数移动平均）
  early_stopping_patience: 5 # 早停耐心值
  save_total_limit: 3        # 最多保留 N 个检查点

system:
  device: "cuda"             # "cuda" 或 "cpu"
  seed: 42                   # 随机种子
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
```

---

## API 接口文档

Web 服务启动后（`python web_app.py`），可通过 API 调用模型。

### 文本生成

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "def hello():", "max_new_tokens": 100, "temperature": 0.8}'
```

响应：
```json
{
  "success": true,
  "text": "def hello():\n    print('Hello, World!')\n...",
  "session_id": "xxx",
  "interaction_id": "xxx"
}
```

### 模型信息

```bash
curl http://localhost:5000/api/info
```

### 健康检查

```bash
curl http://localhost:5000/api/health
```

### 提交反馈

```bash
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"interaction_id": "xxx", "feedback": 1}'
```

`feedback` 值：`1`（赞）、`-1`（踩）、`0`（无反馈）

### API 端点一览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/generate` | POST | 文本生成 |
| `/api/info` | GET | 模型信息 |
| `/api/health` | GET | 健康检查 |
| `/api/feedback` | POST | 提交反馈（赞/踩） |
| `/api/learning-status` | GET | 自动学习状态 |
| `/api/learning/start` | POST | 手动触发增量学习 |
| `/api/learning/config` | GET/POST | 学习配置管理 |
| `/api/user-rights` | GET | 用户权利概览 |
| `/api/data-export` | GET | 数据导出 |
| `/api/data-delete` | POST | 数据删除 |

---

## 自动学习系统

### 启用自动学习

编辑 `config/config.yaml`：

```yaml
auto_learning:
  enabled: true              # 启用自动学习
  min_feedback_samples: 20   # 积累 20 条反馈后触发学习
  incremental_epochs: 3      # 增量训练 3 轮
  incremental_lr: 0.00005    # 使用更小的学习率
  augmentation_enabled: true # 启用数据增强
```

### 工作流程

1. 用户通过 Web 界面与模型对话
2. 对生成结果点赞/点踩，系统自动记录反馈
3. 当正面反馈样本达到阈值时，自动触发增量训练
4. 也可以通过 API 手动触发：`POST /api/learning/start`

---

## 常见问题

### Q: 如何切换推理后端？

```bash
# PyTorch 后端（GPU 推理）
python generate.py --backend pytorch --device cuda

# NumPy 后端（CPU 推理，无需安装 PyTorch）
python generate.py --backend numpy
```

### Q: 如何导出模型？

```python
# GGUF 导出（用于 llama.cpp）
from export.gguf import export_gguf
export_gguf(model, "codesprite.gguf")

# ONNX 导出
from export.onnx import export_onnx
export_onnx(model, "codesprite.onnx")
```

### Q: 如何转换旧版权重？

```bash
python tools/convert_checkpoint.py --old checkpoints/best_model.pt --new checkpoints/best_model_v2.pt
```

### Q: 训练时 Val Loss 显示 `inf` 怎么办？

前几个 epoch 出现 `inf` 通常是因为 warmup 阶段学习率极小，模型输出数值不稳定。训练几个 epoch 后学习率上升，Val Loss 会自动恢复正常。如果持续为 `inf`，尝试：
- 增大学习率：`--lr 0.001`
- 减小 warmup 步数：修改 `config.yaml` 中 `warmup_steps`
- 禁用混合精度：`--no-amp`

### Q: 没有 GPU 能训练吗？

可以，程序会自动回退到 CPU。但速度较慢，建议：
- 减小 `batch_size`
- 减小 `num_epochs`
- 使用 Google Colab（免费 GPU）等平台

### Q: 如何使用自己的训练数据？

将数据按行写入 `data/raw/train.txt`、`data/raw/val.txt`、`data/raw/test.txt`，每行一条样本，然后直接运行 `python train.py`。

### Q: 模型权重文件太大，无法上传 GitHub？

`checkpoints/` 目录已在 `.gitignore` 中排除。如需分享权重，推荐：
- [Hugging Face Hub](https://huggingface.co/)（推荐，免费）
- Google Drive
- 百度网盘

### Q: 如何调整模型大小？

修改 `config/config.yaml`：

```yaml
model:
  hidden_size: 256           # 减小隐藏层（参数更少）
  num_layers: 4              # 减少层数
  num_heads: 4               # 减少注意力头
  # 或增大以获得更强模型
  hidden_size: 768
  num_layers: 12
  num_heads: 12
```

> 注意：修改模型架构后，旧的检查点将不兼容，需要重新训练。

### Q: Windows 下 PowerShell 报错 npm 相关问题？

本项目不依赖 npm/Node.js，纯 Python 项目，不受影响。

---

## 项目结构

```
codesprite/
├── ir/                      # 模型结构定义（零框架依赖）
│   ├── config.py            # ModelConfig 数据类
│   ├── layers.py            # 抽象层定义（Linear/Attention/FFN等）
│   ├── transformer.py       # 完整 Transformer 模型结构
│   └── graph.py             # 计算图（预留）
├── ops/                     # 算子抽象（数学接口）
│   ├── attention.py         # 注意力算子
│   ├── activation.py        # 激活函数（SiLU/GELU/Softmax）
│   └── norm.py              # 归一化（LayerNorm/RMSNorm）
├── backends/                # 计算后端实现
│   ├── base.py              # Backend 抽象接口
│   ├── pytorch.py           # PyTorch 后端（训练用）
│   └── numpy.py             # NumPy 后端（纯CPU推理）
├── training/                # 训练模块
│   ├── trainer.py           # 后端无关训练器
│   └── optimizer.py         # 优化器工具
├── inference/               # 推理接口
│   └── engine.py            # 推理引擎（自动选后端）
├── export/                  # 跨平台导出
│   ├── gguf.py              # GGUF 格式导出（llama.cpp兼容）
│   └── onnx.py              # ONNX 格式导出
├── tools/                   # 工具脚本
│   └── convert_checkpoint.py  # 旧权重转换工具
├── config/
│   └── config.yaml          # 模型和训练配置
├── data/
│   └── raw/                 # 训练数据
├── src/                     # 保留组件
│   ├── tokenizer.py         # 分词器 + 数据集（框架无关）
│   ├── moderator.py         # 内容审核
│   └── compliance.py        # 安全合规
├── templates/               # Web 前端
├── train.py                 # 训练入口
├── generate.py              # 交互式生成
├── evaluate.py              # 模型评估
├── web_app.py               # Flask Web 服务
├── requirements.txt         # Python 依赖
├── LICENSE                  # MIT
└── README.md
```

---

## 架构设计

```
               ┌──────────────────────┐
               │    ir/ (模型定义层)    │
               │  零框架依赖，纯结构描述  │
               └──────────┬───────────┘
                          │ delegate
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ PyTorch后端  │ │  NumPy后端   │ │  (MLX后端)   │
   │ 训练+GPU推理  │ │ CPU纯推理    │ │  Apple芯片   │
   └─────────────┘ └─────────────┘ └─────────────┘
          ↓               ↓
   ┌─────────────┐ ┌─────────────┐
   │ GGUF导出     │ │ ONNX导出     │
   │ llama.cpp   │ │ ONNX Runtime │
   └─────────────┘ └─────────────┘
```

**核心思想**：模型定义与计算框架完全解耦。`ir/` 中的代码不 `import torch`，不 `import numpy`，只描述"模型有哪些层、长什么样"。具体怎么算由 `backends/` 决定。

### 模型架构细节

```
Transformer Decoder (LLaMA 风格)

┌──────────────────────────────────┐
│         Token Embedding           │
│         Dropout                   │
├──────────────────────────────────┤
│   TransformerBlock × 8           │
│   ├── RMSNorm (Pre-Norm)         │
│   ├── Self-Attention (GQA)       │
│   │   ├── RoPE Position Encoding │
│   │   ├── Rotating KV-Cache      │
│   │   └── Causal Masking         │
│   ├── Residual                   │
│   ├── RMSNorm (Pre-Norm)         │
│   ├── SwiGLU FeedForward         │
│   └── Residual                   │
├──────────────────────────────────┤
│   Final RMSNorm                  │
│   LM Head (Tied Weights)         │
└──────────────────────────────────┘

Parameters: ~37.9M | Hidden: 512 | Heads: 8 | Layers: 8
Vocab: 4268 | Context: 512 tokens | Activation: SwiGLU
Position: RoPE | Norm: RMSNorm | Attention: GQA
```

---

## License

MIT License
