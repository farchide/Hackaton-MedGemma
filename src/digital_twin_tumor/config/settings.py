"""Settings module -- single entry point for application configuration.

:func:`get_config` returns the merged configuration dictionary.  It loads
``config/default.yaml``, overlays the environment-specific file when running
on Kaggle or Colab, and finally applies any ``DTT_`` prefixed environment
variable overrides.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from digital_twin_tumor.config.environment import detect_platform
from digital_twin_tumor.domain.models import AppConfig

# Project root is two levels up from ``src/digital_twin_tumor/config/``.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Overlay map: platform name -> config filename
_OVERLAY_MAP: dict[str, str] = {
    "kaggle": "kaggle.yaml",
    "colab": "colab.yaml",
}


@functools.lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the fully merged configuration dictionary.

    The result is cached so that repeated calls within the same process are
    essentially free.

    Resolution order:

    1. ``config/default.yaml``
    2. ``config/<platform>.yaml`` if the file exists
    3. Environment variables with ``DTT_`` prefix

    Returns
    -------
    dict[str, Any]
        The merged configuration tree.
    """
    config_dir = _PROJECT_ROOT / "config"
    default_path = config_dir / "default.yaml"

    platform = detect_platform()
    overlay_filename = _OVERLAY_MAP.get(platform)
    overlay_path = config_dir / overlay_filename if overlay_filename else None

    app_config = AppConfig.load(
        default_path=default_path,
        overlay_path=overlay_path,
        env_prefix="DTT_",
    )
    return app_config.data


def get_typed_config() -> AppConfig:
    """Return the :class:`AppConfig` wrapper for typed access.

    This is a convenience function when callers prefer ``config.get("key")``
    style access over raw dictionary lookups.

    Returns
    -------
    AppConfig
        Frozen configuration object.
    """
    config_dir = _PROJECT_ROOT / "config"
    default_path = config_dir / "default.yaml"

    platform = detect_platform()
    overlay_filename = _OVERLAY_MAP.get(platform)
    overlay_path = config_dir / overlay_filename if overlay_filename else None

    return AppConfig.load(
        default_path=default_path,
        overlay_path=overlay_path,
        env_prefix="DTT_",
    )
