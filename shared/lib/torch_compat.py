"""Runtime workarounds for native torch backends."""

from __future__ import annotations

import platform
import sys

from .logging import get_logger

logger = get_logger(__name__)


def apply_torch_workarounds() -> bool:
    """Disable oneDNN (mkldnn) on Linux/aarch64. Returns True if it was changed.

    torch 2.1 segfaults inside its oneDNN kernels on Linux arm64, which is what
    Docker runs on Apple Silicon. A native crash cannot be caught, so it would
    take the whole API process down on the first embedding or CNN request.
    Other platforms are left untouched.
    """
    if not (
        sys.platform.startswith("linux")
        and platform.machine().lower() in ("aarch64", "arm64")
    ):
        return False
    try:
        import torch
    except ImportError:
        return False
    torch.backends.mkldnn.enabled = False
    logger.info("Disabled torch mkldnn on linux/aarch64 to avoid a native crash")
    return True
