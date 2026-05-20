# Mini LLM - 从零训练微型代码语言模型

一个专注于**代码编程**的微型语言模型，基于 Transformer Decoder 架构，支持多语言代码生成、代码补全和代码解释。

## 项目概述

本项目实现了一个面向代码生成领域的微型语言模型，训练数据涵盖 Python、JavaScript、Java、C++、Go、Rust、TypeScript、SQL、Shell 等主流编程语言。项目包含完整的：

- Transformer Decoder 模型实现（类 GPT）
- 字符级分词器（优化代码符号识别）
- 多语言代码训练语料
- 完整的训练 / 评估 / 生成流程
- Flask 网页交互界面（代码高亮展示）
- 内容审核与代码安全过滤机制
- 多司法管辖区合规体系（CN / EU / US）
- 用户协议和隐私政策

## 模型规格

| 参数 | 值 |
|------|-----|
| **参数量** | ~50M |
| **架构** | Transformer Decoder（类 GPT） |
| **层数** | 8 |
| **隐藏维度** | 512 |
| **注意力头数** | 8 |
| **FFN 中间维度** | 2048 |
| **上下文长度** | 512 tokens |
| **训练语料** | Python / JS / Java / C++ / Go / Rust / TS / SQL / Shell / HTML / CSS |

## 核心能力

- **代码生成**：根据自然语言描述生成代码片段
- **代码补全**：续写未完成的代码
- **代码解释**：用自然语言解释代码逻辑
- **多语言支持**：覆盖 10+ 种主流编程语言

## 项目结构

```
mini-llm/
├── config/
│   └── config.yaml          # 超参数配置
├── src/
│   ├── __init__.py
│   ├── model.py              # Transformer 模型实现
│   ├── tokenizer.py          # 字符级分词器（代码优化）
│   ├── trainer.py            # 训练器（梯度累积 / 学习率调度）
│   ├── moderator.py          # 内容审核 + 代码安全过滤
│   └── compliance.py         # 合规基础设施（审计日志 / 速率限制 / GDPR）
├── data/
│   └── raw/                   # 训练数据
│       ├── train.txt          # 训练集（多语言代码语料）
│       ├── val.txt            # 验证集
│       └── test.txt           # 测试集
├── templates/                # 网页模板
│   ├── index.html            # 主界面（代码高亮展示）
│   ├── agreement.html        # 用户协议
│   └── privacy.html          # 隐私政策
├── checkpoints/              # 模型检查点（训练后生成）
├── logs/                     # 训练日志 + 审计日志
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── generate.py               # 交互式代码生成
├── web_app.py                # Flask 网页服务
├── start_web.bat             # Windows 一键启动
├── LICENSE                   # MIT 许可证
└── requirements.txt          # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**GPU 训练**（推荐）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 准备训练数据

在 `data/raw/` 目录下放置代码训练语料，项目已内置多语言代码数据：

```
data/raw/
├── train.txt   # 多语言代码片段 + 注释 + 解释（Python / JS / Java / C++ / Go / Rust / TS / SQL / Shell / HTML / CSS）
├── val.txt     # 验证集
└── test.txt    # 测试集
```

自定义训练数据格式：每行一个代码片段或代码相关的文本描述。

### 3. 开始训练

```bash
python train.py
```

训练完成后，权重保存在 `checkpoints/` 目录。

### 4. 交互式代码生成

```bash
python generate.py
```

### 5. 网页界面（推荐）

```bash
# Windows
start_web.bat
# 或
python web_app.py
```

浏览器打开 **http://localhost:5000**，可以：
- 输入自然语言描述，生成代码
- 调整生成参数（温度 / Top-K / 生成长度）
- 代码高亮展示输出结果
- 实时查看模型信息

## 模型架构

```
MiniLLM (Code-Focused)
├── TokenEmbedding (vocab_size -> hidden_size)
├── PositionEmbedding (max_seq_length -> hidden_size)
├── Dropout
├── TransformerBlock × 8
│   ├── Multi-Head Self-Attention
│   │   ├── Q / K / V Projections
│   │   ├── Scaled Dot-Product Attention
│   │   └── Output Projection
│   ├── Residual Connection + LayerNorm
│   ├── Feed-Forward Network
│   │   ├── Linear (hidden_size -> intermediate_size)
│   │   ├── GELU Activation
│   │   └── Linear (intermediate_size -> hidden_size)
│   └── Residual Connection + LayerNorm
├── Final LayerNorm
└── LM Head (hidden_size -> vocab_size, tied weights)
```

## 合规与安全

### 代码安全审核

模型内置代码安全过滤机制，自动拦截以下类型请求：

| 安全类别 | 示例 | 法规依据 |
|---------|------|---------|
| 恶意代码 | 病毒 / 木马 / 蠕虫 | 各国网络安全法 |
| 漏洞利用 | SQL注入 / XSS / RCE | CFAA (US) / 网安法 (CN) |
| 攻击脚本 | DDoS / 暴力破解 | CFAA / Computer Misuse Act (UK) |
| 恶意软件制作 | 勒索软件 / 键盘记录 | 各国反计算机犯罪法 |
| 数据窃取 | 爬取敏感信息 / 数据泄露 | GDPR / PIPL |

### 多司法管辖区合规

| 司法管辖区 | 核心法规 |
|-----------|---------|
| **中国** | 生成式AI管理办法、网络安全法、个人信息保护法(PIPL)、数据安全法 |
| **欧盟** | EU AI Act、GDPR、Digital Services Act、NIS2 |
| **美国** | AI Executive Order、CFAA、COPPA、CCPA/CPRA |
| **国际公约** | Berne Convention（知识产权） |

## 配置说明

在 `config/config.yaml` 中调整模型和训练参数：

### 模型配置
- `vocab_size`: 词表大小（字符级分词器为 261）
- `hidden_size`: 隐藏层维度（512）
- `num_layers`: Transformer层数（8）
- `num_heads`: 注意力头数（8）
- `max_seq_length`: 最大序列长度（512）
- `tie_weights`: 权重共享（true，减少参数量）

### 训练配置
- `batch_size`: 批大小（16）
- `learning_rate`: 学习率（0.0003）
- `num_epochs`: 训练轮数（10）
- `gradient_accumulation_steps`: 梯度累积步数（4）

## 扩展建议

### 1. 增强代码能力
- 使用代码专用数据集（如 CodeParrot、The Stack）
- 实现 BPE/WordPiece 分词器以支持代码 token
- 添加代码语法检查器作为后处理

### 2. 扩展模型规模
- 增加 `num_layers` 到 12-24 层
- 增加 `hidden_size` 到 768-1024
- 使用混合精度训练（FP16/BF16）

### 3. 高级功能
- 实现 Beam Search 解码
- 添加 Retrieval-Augmented Generation (RAG)
- 支持代码语法高亮自动标注
- 添加 Fine-tuning 接口

## 硬件需求

- **最小**: 4GB GPU 显存或仅 CPU
- **推荐**: 8GB+ GPU 显存
- **训练时间**: 数小时（取决于数据量和硬件）

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 参考资料

- "Attention Is All You Need" - Vaswani et al.
- "Codex: A Generative Pre-trained Model for Code" - Chen et al.
- "Code Llama: Open Foundation Models for Code" - Roziere et al.
- "The Stack: 3 TB of Permissively Licensed Source Code" - Li et al.
