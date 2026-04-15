"""
triton-anchor: Unified Triton Compilation Frontend
===================================================

A plugin-based compilation frontend that supports three compute paradigms:
  - AME Matrix (RISC-V matrix extensions, e.g., SpacemiT X60)
  - Tensor Processor (dedicated tensor units, e.g., Sophgo BM1684X)
  - gpGPU (SIMT execution model, e.g., USC GPU, NVIDIA)

Architecture:
  Layer 0  — DSL Extensions      (triton.dsl_extensions entry_points)
  Layer 1  — TTIR Pipeline       (core invariant: 7 mandatory passes)
  Layer 2  — Linalg Adapters     (triton-shared / triton-linalg / hybrid)
  Layer 2.5 — AnchorIR Spec      (core invariant: dual-track dialect whitelist)
  Layer 3  — Backend Plugins     (triton.backends entry_points)
"""

__version__ = "0.1.3"

from .hw_capability import HWCapability, ComputeParadigm
from .anchor_ir import AnchorIRTrack, AnchorIRValidator
from .pipeline import build_ttir_pipeline
