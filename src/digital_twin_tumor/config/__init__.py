"""Configuration sub-package.

Provides environment detection, settings loading, and typed configuration
access.

Quick usage::

    from digital_twin_tumor.config import get_config, detect_platform

    cfg = get_config()
    print(cfg["compute"]["precision"])
"""

from __future__ import annotations

from digital_twin_tumor.config.environment import (
    detect_gpu,
    detect_platform,
    get_precision,
)
from digital_twin_tumor.config.settings import get_config, get_typed_config

__all__ = [
    "detect_gpu",
    "detect_platform",
    "get_config",
    "get_precision",
    "get_typed_config",
]
