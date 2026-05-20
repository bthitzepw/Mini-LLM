# Mini LLM - 微型代码语言模型

从零构建的微型语言模型，专注于代码编程，支持中英文双语，集成深度学习增强和自动学习系统。

## 特性

### 深度学习增强
| 特性 | 说明 |
|------|------|
| **RoPE** | 旋转位置编码，支持序列长度外推 |
| **SwiGLU** | 门控前馈网络，更强表达能力 |
| **KV-Cache** | 键值缓存加速自回归推理 |
| **混合精度 (AMP)** | FP16/FP32混合计算，加速训练+节省显存 |
| **标签平滑** | 防止过拟合，提升泛化能力 |
| **EMA** | 指数移动平均，提升推理稳定性 |
| **梯度检查点** | 用计算换显存，支持更大batch |
| **早停机制** | 防止过拟合 |
| **Top-P 采样** | Nucleus采样，更自然的文本生成 |
| **学习率查找器** | 自动寻找最优学习率 |

### 自动学习系统
| 特性 | 说明 |
|------|------|
| **用户反馈** | 对话点赞/点踩收集 |
| **交互记录** | SQLite持久化存储 |
| **数据增强** | 代码变换生成新训练样本 |
| **增量训练** | 基于反馈数据自动微调 |
| **进度追踪** | 训练轮次、指标变化追踪 |
| **Web 控制面板** | 在线查看学习状态 |

### 代码安全 & 合规
- 多国内容审核（CN/EU/US）
- PII 检测与脱敏
- 代码安全过滤（恶意代码/漏洞利用/攻击脚本）
- 安全教育豁免机制
- GDPR/PIPL/CCPA 数据权利接口
- AI 生成内容标识

## 模型架构

```
Transformer Decoder (类 GPT)

┌──────────────────────────────────┐
│         Token Embedding           │
│         Dropout                   │
├──────────────────────────────────┤
│   TransformerBlock × 8            │
│   ├── LayerNorm (Pre-Norm)        │
│   ├── Multi-Head Self-Attention   │
│   │   ├── RoPE Position Encoding  │
│   │   ├── KV-Cache Support        │
│   │   └── Causal Masking          │
│   ├── Residual Connection         │
│   ├── LayerNorm (Pre-Norm)        │
│   ├── SwiGLU Feed-Forward Network │
│   └── Residual Connection         │
├──────────────────────────────────┤
│   Final LayerNorm                 │
│   LM Head (Weight Tied)           │
└──────────────────────────────────┘

参数量: ~50M | 隐藏层: 512 | 头数: 8 | 层数: 8
词表: 4268 (ASCII + CJK汉字 + 代码符号)
上下文: 512 tokens
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 标准训练

```bash
# 基础训练
python train.py

# 深度学习增强训练（推荐）
python train.py --use-ema

# 禁用混合精度
python train.py --no-amp

# 自定义参数
python train.py --lr 0.001 --epochs 20 --batch-size 32
```

### 自动学习训练

```bash
# 从用户反馈数据中学习
python train.py --mode auto

# 查找最优学习率
python train.py --find-lr
```

### 交互式生成

```bash
python generate.py
```

### 模型评估

```bash
python evaluate.py
```

### Web 服务

```bash
python web_app.py
# 打开 http://localhost:5000
```

## 训练配置

```yaml
# 深度学习特性
model:
  use_rope: true           # 旋转位置编码
  use_swiglu: true         # SwiGLU前馈网络

training:
  use_amp: true            # 混合精度训练
  label_smoothing: 0.05    # 标签平滑
  use_ema: false           # 指数移动平均
  early_stopping_patience: 5

# 自动学习配置
auto_learning:
  enabled: false           # 是否启用自动学习
  min_feedback_samples: 20 # 最低反馈样本数
  incremental_epochs: 3    # 增量训练轮次
  incremental_lr: 0.00005  # 增量训练学习率
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/generate` | POST | 文本生成 |
| `/api/info` | GET | 模型信息 |
| `/api/health` | GET | 健康检查 |
| `/api/feedback` | POST | 提交反馈 |
| `/api/learning-status` | GET | 学习状态 |
| `/api/learning/start` | POST | 手动触发学习 |
| `/api/learning/config` | GET/POST | 学习配置 |
| `/api/user-rights` | GET | 用户权利 |
| `/api/data-export` | GET | 数据导出 |
| `/api/data-delete` | POST | 数据删除 |

## 项目结构

```
├── config/
│   └── config.yaml          # 模型和训练配置
├── data/
│   ├── auto_learning.db     # 自动学习数据库
│   └── raw/                 # 训练数据
│       ├── train.txt        # 训练集
│       ├── val.txt          # 验证集
│       └── test.txt         # 测试集
├── src/
│   ├── model.py             # Transformer模型（RoPE/KV-Cache/SwiGLU）
│   ├── tokenizer.py         # 代码+中文分词器
│   ├── trainer.py           # 训练器（AMP/EMA/标签平滑）
│   ├── auto_learner.py      # 自动学习系统
│   ├── moderator.py         # 内容审核
│   └── compliance.py        # 合规基础设施
├── templates/
│   ├── index.html           # Web界面（含反馈按钮和学习面板）
│   ├── agreement.html       # 用户协议
│   └── privacy.html         # 隐私政策
├── train.py                 # 训练入口
├── generate.py              # 交互式生成
├── evaluate.py              # 模型评估
├── web_app.py               # Flask Web服务
├── requirements.txt
└── README.md
```

## License

MIT License
