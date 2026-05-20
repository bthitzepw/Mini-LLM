# Mini LLM - 从零训练微型语言模型

这是一个从头开始实现和训练微型语言模型的完整项目，非常适合学习大型语言模型（LLM）的核心原理。

## 项目概述

本项目实现了一个基于Transformer架构的微型语言模型，包含完整的：
- Transformer模型实现
- 自定义分词器
- 数据处理流程
- 训练循环
- 评估和生成脚本
- 网页访问界面
- 内容审核机制
- 用户协议和隐私政策

## 模型规格

- **参数量**: ~50M参数
- **架构**: Transformer Decoder
- **层数**: 8层
- **隐藏维度**: 512
- **注意力头数**: 8
- **上下文长度**: 512 tokens

这个规模可以在个人电脑上训练，适合学习和实验。

## 项目结构

```
mini-llm/
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── __init__.py
│   ├── model.py              # Transformer模型实现
│   ├── tokenizer.py          # 分词器实现
│   ├── trainer.py            # 训练器
│   └── moderator.py          # 内容审核模块
├── data/
│   └── raw/                   # 训练数据目录
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── templates/                # 网页模板
│   ├── index.html            # 主界面
│   ├── agreement.html        # 用户协议
│   └── privacy.html          # 隐私政策
├── checkpoints/              # 模型检查点
├── logs/                     # 训练日志
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── generate.py               # 交互式生成脚本
├── web_app.py                # 网页服务器
├── start_web.bat             # 网页启动脚本（Windows）
└── requirements.txt          # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

在 `data/raw/` 目录下放置你的文本数据，每个文件包含一行一条文本：

```bash
mkdir -p data/raw
# 添加你的训练数据
echo "这是一个示例文本。" > data/raw/train.txt
echo "这是验证数据。" > data/raw/val.txt
echo "这是测试数据。" > data/raw/test.txt
```

### 3. 开始训练

```bash
python train.py
```

### 4. 交互式生成

训练完成后，使用生成脚本进行对话：

```bash
python generate.py
```

### 5. 模型评估

```bash
python evaluate.py
```

### 6. 网页访问（推荐）

启动网页服务器，在浏览器中与模型交互：

**Windows用户**：
```bash
# 双击运行
start_web.bat

# 或在命令行中运行
python web_app.py
```

**其他系统**：
```bash
python web_app.py
```

然后在浏览器中打开：**http://localhost:5000**

网页界面提供：
- 美观的聊天界面
- 可调整的生成参数（生成长度、温度、Top-K）
- 实时显示模型信息（参数量、层数、设备）
- 用户协议和隐私政策链接
- 合规提示横幅
- AI生成内容标识

## 合规功能

本项目已按照《生成式人工智能服务管理暂行办法》要求实现以下功能：

### 1. 内容审核
- 输入内容审核：检测敏感关键词和违规模式
- 输出生成内容审核：过滤敏感内容
- 可扩展的审核规则，支持自定义敏感词库

### 2. 用户协议和隐私政策
- 完整的用户服务协议页面
- 详细的隐私政策说明
- 数据安全和使用规范说明

### 3. 透明标识
- AI生成内容明确标识
- 免责声明说明内容仅供参考
- 使用场景和限制说明

### 4. 安全机制
- 本地部署，数据不出域
- 无个人身份信息收集
- 会话数据本地处理

## 配置说明

所有超参数都在 `config/config.yaml` 中定义：

### 模型配置
- `vocab_size`: 词表大小
- `hidden_size`: 隐藏层维度
- `num_layers`: Transformer层数
- `num_heads`: 注意力头数
- `intermediate_size`: FFN中间层维度
- `max_seq_length`: 最大序列长度
- `dropout`: Dropout比例

### 训练配置
- `batch_size`: 批大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `warmup_steps`: 预热步数
- `max_steps`: 最大训练步数
- `gradient_accumulation_steps`: 梯度累积步数
- `max_grad_norm`: 梯度裁剪阈值

## 学习要点

通过这个项目，你将学习到：

1. **Transformer架构**: 理解Self-Attention、Multi-Head Attention的原理
2. **语言模型**: 了解GPT类模型的训练方式（Next Token Prediction）
3. **训练技巧**: 梯度累积、梯度裁剪、学习率调度
4. **数据处理**: 分词、数据批处理、padding
5. **模型优化**: 权重共享、LayerNorm、Dropout
6. **分布式训练基础**: （如需扩展）

## 模型架构

```
MiniLLM
├── TokenEmbedding (vocab_size -> hidden_size)
├── PositionEmbedding (max_seq_length -> hidden_size)
├── Dropout
├── TransformerBlock × 8
│   ├── Attention
│   │   ├── Q/K/V Projections
│   │   └── Output Projection
│   ├── Residual Connection + LayerNorm
│   ├── FeedForward
│   │   ├── Linear (hidden_size -> intermediate_size)
│   │   ├── GELU Activation
│   │   └── Linear (intermediate_size -> hidden_size)
│   └── Residual Connection + LayerNorm
├── LayerNorm
└── LM Head (hidden_size -> vocab_size)
```

## 扩展建议

### 1. 增加模型规模
修改 `config.yaml` 中的参数来训练更大的模型：
- 增加 `num_layers` 到 12
- 增加 `hidden_size` 到 768
- 增加 `num_heads` 到 12

### 2. 使用更大数据集
使用公开数据集，如：
- Wikipedia
- BookCorpus
- Common Crawl

### 3. 实现更多功能
- 支持BPE/WordPiece分词
- 实现Beam Search
- 添加早停机制
- 实现混合精度训练
- 添加分布式训练支持

## 硬件需求

- **最小**: 4GB GPU显存（或仅CPU）
- **推荐**: 8GB+ GPU显存
- **训练时间**: 约几小时到几天（取决于数据集大小）

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 参考资料

- "Attention Is All You Need" - Vaswani et al.
- "GPT-3: Language Models are Few-Shot Learners" - Brown et al.
- "LLaMA: Open and Efficient Foundation Language Models" - Touvron et al.
