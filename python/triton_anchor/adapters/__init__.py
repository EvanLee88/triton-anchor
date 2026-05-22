"""Adapter package — TTIR → Linalg/TritonGPU conversion adapters."""

from .base import (
    ITritonToLinalgAdapter as ITritonToLinalgAdapter,
    ILinalgOptAdapter as ILinalgOptAdapter,
    ILinalgPybindAdapter as ILinalgPybindAdapter,
    AdapterConversionError as AdapterConversionError,
)
from .registry import AdapterRegistry as AdapterRegistry, get_adapter as get_adapter

# 注册默认自带的 in-process adapter
try:
    from .triton_linalg_adapter import TritonLinalgAdapter

    AdapterRegistry.register(TritonLinalgAdapter())
except ImportError:
    pass
