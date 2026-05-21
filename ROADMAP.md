# CodeSprite 演进路线图

> 最后更新：2026-05-21

---

## 一、IR 架构与后端扩展

### 1.1 SSA 与 Pass 优化框架

当前 `ir/` 层以结构描述为主，下一步引入编译器中成熟的 SSA（静态单赋值）设计：

- [ ] **引入 SSA 核心抽象**：统一 `Operation`（操作）、`Region`（区域）、`Value`（值）基础结构
- [ ] **图优化 Pass 管线**：
  - [ ] 算子融合（Operator Fusion）— 合并相邻 Attention + Linear 等
  - [ ] 常量折叠（Constant Folding）
  - [ ] 死代码消除（Dead Code Elimination）
  - [ ] 公共子表达式消除（CSE）
- [ ] **Pass 注册与调度机制**：支持自定义 Pass 插入和顺序编排

### 1.2 扩展更多计算后端

- [ ] **MLX 后端**（Apple Silicon 原生加速，Metal GPU）
- [ ] **JAX 后端**（函数式、自动微分、TPU 支持）
- [ ] **TinyGrad 后端**（极简深度学习框架，适合教学）
- [ ] **TVM / MicroTVM**（嵌入式端侧部署）
- [ ] **LiteRT**（Google 轻量级推理运行时，前身 TFLite）

### 1.3 IR 序列化与可视化

- [ ] **文本化 IR 快照导出**：将 `ir/` 图结构渲染为可读文本格式
- [ ] **Python 代码生成**：从 IR 还原回可执行的 Python 代码
- [ ] **Graphviz / Mermaid 可视化**：输出模型计算图，方便调试
- [ ] **Pass 前后对比 difff**：可视化 Pass 优化前后的图变化

---

## 二、代码生成能力与模型优化

### 2.1 FIM（Fill-in-the-Middle）代码补全

- [ ] **FIM 数据格式**：引入 `<PRE>` `<MIDDLE>` `<SUF>` 哨兵标记
- [ ] **训练数据预处理**：从代码语料中随机切分 prefix-middle-suffix 三元组
- [ ] **FIM 推理模式**：支持 `codesprite fim --prefix "..." --suffix "..."`
- [ ] **PSM 与 SPM 双模式**：Prefix-Suffix-Middle 和 Suffix-Prefix-Middle 两种切分策略

### 2.2 长序列处理

- [ ] **上下文扩展至 1024/2048**：调整 `max_seq_length`，评估显存/内存开销
- [ ] **滑动窗口注意力**：推理层支持滑动窗口 + 环形缓冲区管理长输入
- [ ] **NTK-aware RoPE 动态缩放**：不重训前提下临时扩展上下文容量
- [ ] **YaRN / ReRoPE 实证评估**：对比几种动态缩放方案的 PPL

### 2.3 量化与边缘部署

- [ ] **INT8 量化**：基于 GGUF 的 Q8_0 量化 + 质量对比
- [ ] **INT4 量化**：GGUF Q4_K_M 量化 + ONNX Runtime INT4
- [ ] **量化后 PPL 基准**：WikiText-2 / CodeSearchNet 上的困惑度对比表
- [ ] **量化后推理速度对比**：CPU / GPU / Apple Silicon 上的 tokens/s

### 2.4 Tokenizer 专业化

- [ ] **代码专用 BPE Tokenizer**：基于 The Stack / CodeSearchNet 代码语料训练
- [ ] **代码符号压缩比优化**：重点优化括号、缩进、关键字、运算符
- [ ] **多语言 Tokenizer 对比**：Python / JavaScript / Go / Rust 的压缩效率
- [ ] **特殊 Token 设计**：`<FIM_PREFIX>` / `<FIM_MIDDLE>` / `<FIM_SUFFIX>` / `<REPO_NAME>` 等

---

## 三、工程化、Agent 化与生态

### 3.1 IDE / CLI 插件

- [ ] **CLI 工具**：`codesprite` 命令行，子命令 `complete` / `chat` / `fim` / `serve`
- [ ] **VS Code 扩展**：侧边栏面板，实时代码补全（FIM 模式）
- [ ] **JetBrains 插件**：IntelliJ / PyCharm 支持
- [ ] **LSP 集成**：通过 Language Server Protocol 与编辑器深度整合

### 3.2 静态分析与自愈层

- [ ] **外挂静态分析器**：mypy（Python）/ eslint（JavaScript）/ ruff
- [ ] **生成→校验→重试管道**：代码生成后自动跑语法/类型检查，失败触发约束重试
- [ ] **约束解码（Constrained Decoding）**：根据上下文限制 token 候选集，防止幻觉
- [ ] **幻觉检测与自愈**：检测生成的 API 是否真实存在，不存在则回退修正

### 3.3 RAG 与仓库级理解

- [ ] **轻量 Vector Store**：FAISS / Chroma / LanceDB 本地嵌入索引
- [ ] **Per-File Isolation 检索**：生成前检索当前文件依赖和局部上下文
- [ ] **仓库级代码理解**：项目级符号索引 + 依赖图，弥补小模型上下文短板
- [ ] **BM25 + 语义混合检索**：关键词匹配 + 向量相似度融合排序

### 3.4 CI/CD 与工程规范

- [ ] **GitHub Actions 流水线**：
  - [ ] 单元测试自动运行（pytest）
  - [ ] 模型权重完整性检查（checksum 验证）
  - [ ] GGUF / ONNX 导出格式自动验证
  - [ ] 代码风格检查（ruff / black）
- [ ] **CHANGELOG.md**：按 Semantic Versioning 记录版本变更
- [ ] **Semantic Versioning**：`MAJOR.MINOR.PATCH` 版本管理规范
- [ ] **Release 自动化**：打 tag → 自动构建 → 发布到 GitHub Releases

---

## 优先级建议

| 优先级 | 模块 | 理由 |
|--------|------|------|
| P0 | FIM 代码补全 | 直接提升开发体验，IDE 插件前置依赖 |
| P0 | INT4/INT8 量化 | 降低部署门槛，消费级设备可用 |
| P1 | SSA + Pass 优化 | IR 架构核心差异化能力 |
| P1 | Tokenizer 专业化 | 代码场景感知提升明显 |
| P1 | CLI 工具 | 融入开发者工作流的最短路径 |
| P2 | MLX / JAX 后端 | Apple Silicon 生态价值高 |
| P2 | RAG + 仓库理解 | 弥补小模型短板 |
| P2 | CI/CD 自动化 | 工程化成熟度 |
| P3 | VS Code 扩展 | 依赖 FIM + CLI 先完成 |
| P3 | TVM / 端侧部署 | 特定场景需求 |

---

> 欢迎通过 [GitHub Issues](https://github.com/bthitzepw/CodeSprite/issues) 参与讨论和贡献。
