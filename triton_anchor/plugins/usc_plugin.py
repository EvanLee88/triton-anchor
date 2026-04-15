"""
USC GPU Backend Plugin — Stub
================================

Interface reservation for USC GPU backend (gpGPU paradigm).
Preserves design extensibility for future integration.

Key difference from Linalg-path backends:
  - USC uses ``lowering_path="triton_gpu"`` (TritonGPU IR, not Linalg)
  - Does NOT go through Linalg Adapter (Layer 2)
  - Shares only Layer 1 (TTIR Pipeline) with other backends

Compilation pipeline (when implemented):
  TTIR → TritonGPU IR → GPUToLLVMUSC → SPIR-V → binary

Dependencies (not yet integrated):
  - fantasy-triton: ftvm dialect + GPUToLLVMUSC pass
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from .base import BackendPlugin
from ..hw_capability import (
    HWCapability,
    ComputeParadigm,
    GPGPUCapability,
)


class USCPlugin(BackendPlugin):
    """Backend plugin for USC GPU.

    Status: **STUB** — interface reservation with design extensibility.

    USC GPU uses the TritonGPU lowering path, NOT the Linalg path.
    This means:
      - It shares the TTIR Pipeline (Layer 1) with other backends
      - It does NOT use Linalg Adapters (Layer 2)
      - It has its own TritonGPU → LLVM → binary pipeline
    """

    @property
    def name(self) -> str:
        return "usc"

    @property
    def hw_capability(self) -> HWCapability:
        from ..anchor_ir import AnchorIRTrack
        return HWCapability(
            name="usc-gpu",
            arch_family="gpu",
            compute_paradigm=ComputeParadigm.GPGPU,
            anchor_ir_track=AnchorIRTrack.TRITON_GPU,  # Key: NOT linalg
            ptr_model="gpu",
            gpgpu_cap=GPGPUCapability(
                num_warps=4,
                warp_size=32,
                shared_mem_size=49152,
                num_stages=2,
                num_ctas=1,
                cluster_dims=(1, 1, 1),
                supported_dtypes={"fp32", "fp16", "bf16", "int8"},
            ),
        )

    def validate_environment(self) -> Tuple[bool, str]:
        return False, (
            "USC GPU backend not yet integrated. "
            "Requires fantasy-triton toolchain."
        )

    def lower_anchor_ir_to_target(self, anchor_ir: Any, metadata: dict) -> bytes:
        raise NotImplementedError(
            "USC GPU backend uses TritonGPU path, not Linalg/AnchorIR. "
            "Integration pending fantasy-triton. "
            "Use TritonGPU → GPUToLLVMUSC → SPIR-V pipeline directly."
        )

    def create_launcher(self, binary: bytes, signature: dict, metadata: dict) -> Callable:
        raise NotImplementedError("USC GPU launcher not yet integrated.")

    def get_op_coverage(self) -> Dict[str, str]:
        return {
            "tt.dot": "optimal",        # WMMA Tensor Core
            "tt.load": "optimal",       # ld.global / cp.async
            "tt.store": "optimal",
            "tt.reduce": "optimal",     # shfl.sync warp shuffle
            "tt.scan": "optimal",
            "tt.atomic_rmw": "optimal",
            "tt.atomic_cas": "optimal",
            "tt.broadcast": "optimal",
            "tt.splat": "optimal",
        }
