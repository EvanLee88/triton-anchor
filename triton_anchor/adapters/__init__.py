"""Adapter package — TTIR → Linalg/TritonGPU conversion adapters."""
from .base import (
    ITritonToLinalgAdapter,
    ILinalgOptAdapter,
    ILinalgPybindAdapter,
    AdapterConversionError,
)
from .registry import AdapterRegistry, get_adapter
