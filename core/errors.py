"""
core/errors.py · 错误码体系 E100~E500
作者：晨星
"""


class RecForgeError(Exception):
    """所有 RecForge 错误的基类。"""

    code = "E000"

    def __init__(self, message: str = "", *, code: str | None = None):
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(RecForgeError):
    """E100 · 配置/参数错误。"""

    code = "E100"


class DataError(RecForgeError):
    """E200 · 数据生成/载入错误。"""

    code = "E200"


class ModelError(RecForgeError):
    """E300 · 模型训练/推断错误。"""

    code = "E300"


class EvalError(RecForgeError):
    """E400 · 评测/指标错误。"""

    code = "E400"


class PipelineError(RecForgeError):
    """E500 · 流水线编排错误。"""

    code = "E500"


__all__ = [
    "ConfigError",
    "DataError",
    "EvalError",
    "ModelError",
    "PipelineError",
    "RecForgeError",
]
