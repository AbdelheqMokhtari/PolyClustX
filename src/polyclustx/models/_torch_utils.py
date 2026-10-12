"""Shared device selection without changing process-wide PyTorch settings."""

from __future__ import annotations

import secrets
import threading
from contextlib import contextmanager

import torch

_INITIALIZATION_LOCK = threading.RLock()


@contextmanager
def initialization_seed(seed):
    """Initialize CPU modules reproducibly without resetting caller RNG states."""
    with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed if seed is not None else secrets.randbits(63))
        yield


def resolve_device(value):
    """Support CPU/CUDA explicitly; auto uses CUDA only when available."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda (LMGEC requires float64)")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available; use device='cpu' or install a CUDA-enabled PyTorch build")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is out of range")
    return device
