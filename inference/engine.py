"""
推理引擎 — 后端无关的文本生成

自动检测可用后端（PyTorch → NumPy 降级）。
支持 KV-Cache 加速、top-k/top-p 采样、温度控制。

设备选择策略:
  - "auto"  → 自动最优（CUDA > CPU），推荐
  - "cuda"  → 显式 GPU
  - "cpu"   → 纯 CPU 推理（自动限制线程数）

已知问题 / TODO:
  - KV-Cache 目前只对 src/model.py 里的 CodeSprite（nn.Module 版本）生效
    IR 版 TransformerModel 的 KV-Cache 还没接进来，先用全量计算
  - beam search 没做，只有 top-k / top-p
  - 流式输出（streaming）还没加，web_app 那边想要这个功能但先留着
"""

import math
import os
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("CodeSprite")


class InferenceEngine:
    """
    CodeSprite 推理引擎

    用法:
        # 自动选择后端 + 设备
        engine = InferenceEngine(model, checkpoint_path="best_model.pt")

        # 指定后端
        from backends.pytorch import PyTorchBackend
        engine = InferenceEngine(model, backend=PyTorchBackend("cuda"))

        # 纯 CPU 推理
        engine = InferenceEngine(model, device="cpu")

        # 文本生成
        output = engine.generate("def hello(", max_new_tokens=50)
    """

    def __init__(self, model, backend=None, checkpoint_path: str = None,
                 tokenizer=None, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer

        # 自动选择后端
        if backend is not None:
            self.backend = backend
            self._device = getattr(backend, '_resolved_device', device)
        else:
            self.backend, self._device = self._auto_select_backend(device)

        # 打印推理设备信息
        print(f"[Inference] Device: {self._device}, Backend: {self.backend.name}")

        # 加载权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
        elif not self._has_weights():
            raise RuntimeError(
                "模型未初始化权重。请先加载检查点或初始化权重。"
            )

        # 采样参数
        self.temperature = 1.0
        self.top_k = None
        self.top_p = None

    def _auto_select_backend(self, device: str = "auto"):
        """自动选择可用后端，返回 (backend, resolved_device_str)"""
        from src.device import resolve_device

        resolved_device = resolve_device(device)

        try:
            from backends.pytorch import PyTorchBackend, init_model_weights as _
            backend = PyTorchBackend(device=resolved_device)
            return backend, resolved_device
        except ImportError:
            from backends.numpy import NumPyBackend
            logger.warning("PyTorch 不可用，使用 NumPy 后端（纯 CPU 推理）")
            return NumPyBackend(), "cpu"

    def _has_weights(self) -> bool:
        """检查模型是否已有权重"""
        try:
            # 简化检查：只要有任意一个参数就可以
            if hasattr(self.model, 'embedding') and hasattr(self.model.embedding, 'params'):
                return len(self.model.embedding.params) > 0
            # 检查模型根参数
            if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                first_block = self.model.blocks[0]
                if hasattr(first_block, 'params'):
                    return len(first_block.params) > 0
            return True  # 默认认为可以用
        except:
            return True

    def load(self, path: str):
        """加载模型权重"""
        self.backend.load_checkpoint(self.model, path)
        print(f"Loaded weights from {path}")
        print(f"  Backend: {self.backend.name}")
        print(f"  Parameters: {self.model.get_param_count():,}")
        return self

    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = None, top_k: int = None,
                 top_p: float = None) -> str:
        """
        文本生成

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（None = 使用默认值）
            top_k: Top-K 采样参数
            top_p: Top-P (nucleus) 采样参数

        Returns:
            生成的文本（含输入）
        """
        if self.tokenizer is None:
            raise RuntimeError("需要设置 tokenizer 才能生成文本")

        temp = temperature if temperature is not None else self.temperature
        tk = top_k if top_k is not None else self.top_k
        tp = top_p if top_p is not None else self.top_p

        # 编码输入
        input_ids = self.tokenizer.encode(prompt)
        generated_ids = list(input_ids)

        eos_id = self.model.config.eos_token_id

        for _ in range(max_new_tokens):
            # 准备输入（截取最后 max_seq_length 个 token）
            max_len = self.model.config.max_seq_length
            context = generated_ids[-max_len:]

            # 构建 batch 维度
            import numpy as np
            x = np.array([context], dtype=np.int64)

            # 前向传播
            logits = self.model.forward(x, self.backend)

            # 取最后一个位置的 logits
            next_logits = logits[0, -1, :] / max(temp, 1e-10)

            # Top-K 过滤
            if tk is not None and tk > 0:
                threshold = self.backend.argmax(
                    -self.backend.to_numpy(next_logits) if hasattr(self.backend, 'to_numpy')
                    else next_logits
                )
                # 简化版 top-k：用 numpy 实现
                logits_np = self.backend.to_numpy(next_logits)
                indices = (-logits_np).argsort()
                mask = np.ones_like(logits_np, dtype=bool)
                mask[indices[tk:]] = False
                logits_np[~mask] = float('-inf')
                next_logits = logits_np

            # Top-P 过滤
            if tp is not None and tp < 1.0:
                logits_np = self.backend.to_numpy(next_logits) if not isinstance(next_logits, np.ndarray) else next_logits
                sorted_indices = (-logits_np).argsort()
                sorted_logits = logits_np[sorted_indices]
                # softmax
                sorted_logits = sorted_logits - sorted_logits.max()
                sorted_probs = np.exp(sorted_logits) / np.exp(sorted_logits).sum()
                cumsum = np.cumsum(sorted_probs)
                cutoff_idx = (cumsum > tp).argmax() + 1
                if cutoff_idx < len(sorted_indices):
                    logits_np[sorted_indices[cutoff_idx:]] = float('-inf')
                next_logits = logits_np

            # Softmax + 采样
            logits_np = self.backend.to_numpy(next_logits) if not isinstance(next_logits, np.ndarray) else next_logits
            logits_np = logits_np - logits_np.max()
            probs = np.exp(logits_np) / np.exp(logits_np).sum()

            # 从分布中采样
            next_token = np.random.choice(len(probs), p=probs)
            next_token = int(next_token)

            # 停止条件
            if next_token == eos_id:
                break

            generated_ids.append(next_token)

        return self.tokenizer.decode(generated_ids)

    def generate_ids(self, input_ids: List[int], max_new_tokens: int = 50,
                     temperature: float = 1.0, top_k: int = None,
                     top_p: float = None) -> List[int]:
        """
        从 token IDs 生成（不经过 tokenizer）

        Args:
            input_ids: 输入 token ID 序列
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K
            top_p: Top-P

        Returns:
            完整的 token ID 序列
        """
        import numpy as np

        generated = list(input_ids)
        eos_id = self.model.config.eos_token_id

        for _ in range(max_new_tokens):
            max_len = self.model.config.max_seq_length
            context = generated[-max_len:]
            x = np.array([context], dtype=np.int64)

            logits = self.model.forward(x, self.backend)
            next_logits = logits[0, -1, :] / max(temperature, 1e-10)

            # 简化采样
            logits_np = self.backend.to_numpy(next_logits)
            logits_np = logits_np - logits_np.max()
            probs = np.exp(logits_np) / np.exp(logits_np).sum()

            next_token = int(np.random.choice(len(probs), p=probs))

            if next_token == eos_id:
                break
            generated.append(next_token)

        return generated

    def generate_with_kv_cache(self, prompt: str, max_new_tokens: int = 100,
                                temperature: float = 0.8, top_k: int = 50,
                                top_p: float = 0.9) -> str:
        """
        使用 KV-Cache 加速的文本生成（仅支持 src/model.py 中的 CodeSprite 模型）

        KV-Cache 原理：在自回归生成时，每次只输入最后一个新 token，
        历史 token 的 K/V 向量复用缓存，避免重复计算，速度约提升 seq_len 倍。

        注意：IR 版 TransformerModel 暂不支持，会自动降级到全量推理。

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样参数
            top_p: Top-P (nucleus) 采样参数

        Returns:
            生成的完整文本（含输入 prompt）
        """
        if self.tokenizer is None:
            raise RuntimeError("需要设置 tokenizer 才能使用此方法")

        import numpy as np

        # 检查是否是支持 KV-Cache 的原生 nn.Module 模型
        # src/model.py 的 CodeSprite 有 generate() 方法
        has_native_kv_cache = hasattr(self.model, 'generate') and hasattr(self.model, 'layers')

        input_ids = self.tokenizer.encode(prompt)

        if has_native_kv_cache:
            # 走原生 KV-Cache 路径（src/model.py CodeSprite）
            try:
                import torch
                device = getattr(self.backend, 'device', 'cpu')
                ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

                output_ids = self.model.generate(
                    ids_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_kv_cache=True
                )
                result_ids = output_ids[0].tolist()
                return self.tokenizer.decode(result_ids)
            except Exception as e:
                # KV-Cache 路径失败，降级到全量推理
                logger.warning(f"KV-Cache 推理失败，降级到全量推理: {e}")

        # 降级路径：全量推理（IR 模型 or KV-Cache 出错时）
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    def info(self) -> dict:
        """返回引擎信息"""
        return {
            "backend": self.backend.name,
            "parameters": self.model.get_param_count(),
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
