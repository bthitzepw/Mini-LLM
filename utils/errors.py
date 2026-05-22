"""
自定义异常体系

所有 CodeSprite 异常继承自 CodeSpriteError，
便于上层统一捕获和分类处理。
"""


class CodeSpriteError(Exception):
    """基础异常 — 所有 CodeSprite 异常的父类"""
    pass


class ConfigError(CodeSpriteError):
    """配置错误 — YAML 格式错误、字段缺失、值非法等"""
    pass


class BackendError(CodeSpriteError):
    """后端错误 — 算子未实现、设备不兼容、计算失败等"""
    pass


class DeviceError(CodeSpriteError):
    """设备错误 — CUDA 不可用、显存不足、设备选择失败等"""
    pass


class DataError(CodeSpriteError):
    """数据错误 — 文件不存在、格式不合法、分词失败等"""
    pass
