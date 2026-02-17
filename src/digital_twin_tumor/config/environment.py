"""Environment detection utilities.

Provides helpers to detect the runtime platform (Kaggle, Colab, local), the
available GPU hardware, and the recommended numerical precision.
"""

from __future__ import annotations

import os
import shutil
from typing import Any


def detect_platform() -> str:
    """Detect the runtime platform.

    Returns
    -------
    str
        One of ``"kaggle"``, ``"colab"``, or ``"local"``.
    """
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def detect_gpu() -> dict[str, Any]:
    """Detect GPU availability and basic properties.

    Returns a dict with keys:

    * ``available`` -- whether a CUDA GPU is present.
    * ``name`` -- GPU model name or ``"none"``.
    * ``vram_gb`` -- estimated VRAM in GiB, or ``0.0``.

    Returns
    -------
    dict[str, Any]
        GPU information dictionary.
    """
    result: dict[str, Any] = {
        "available": False,
        "name": "none",
        "vram_gb": 0.0,
    }

    # Fast check: if nvidia-smi is not on PATH, skip the import attempt.
    if shutil.which("nvidia-smi") is None:
        return result

    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            result["available"] = True
            result["name"] = props.name
            result["vram_gb"] = round(props.total_mem / (1024**3), 1)
    except ImportError:
        pass

    return result


def get_precision(vram_gb: float) -> str:
    """Choose the best numerical precision given available VRAM.

    The heuristic favours higher precision when possible, falling back to
    quantised formats for memory-constrained environments.

    Parameters
    ----------
    vram_gb:
        Amount of GPU VRAM available in GiB.

    Returns
    -------
    str
        One of ``"bfloat16"``, ``"float16"``, ``"int4"``, or ``"cpu"``.
    """
    if vram_gb <= 0.0:
        return "cpu"
    if vram_gb >= 24.0:
        return "bfloat16"
    if vram_gb >= 12.0:
        return "float16"
    return "int4"
