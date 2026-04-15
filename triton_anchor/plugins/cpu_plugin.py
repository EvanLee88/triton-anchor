"""
CPU Reference Backend Plugin — Stub
=====================================

CPU backend serving as the golden reference for cross-backend
numerical consistency testing.

Compilation pipeline (when implemented):
  AnchorIR → LLVM IR (via MLIR) → opt → Host binary (.so)

This backend is intentionally *slow* but *correct* — it's used
to validate that other backends produce numerically equivalent results.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .base import BackendPlugin
from ..hw_capability import (
    HWCapability,
    ComputeParadigm,
    GPGPUCapability,
)


class CPUPlugin(BackendPlugin):
    """CPU reference backend for correctness validation.

    Status: **STUB** — planned as the golden reference for cross-backend
    numerical consistency testing::

        @pytest.mark.parametrize("backend", ["cpu", "sophgo", "spacemit"])
        def test_matmul_consistency(backend):
            cpu_result = run_kernel(matmul_kernel, backend="cpu")
            target_result = run_kernel(matmul_kernel, backend=backend)
            assert_allclose(cpu_result, target_result, rtol=1e-5)
    """

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def hw_capability(self) -> HWCapability:
        from ..anchor_ir import AnchorIRTrack
        return HWCapability(
            name="cpu-reference",
            arch_family="gpu",       # Use GPU paradigm for simplest compatibility
            compute_paradigm=ComputeParadigm.GPGPU,
            anchor_ir_track=AnchorIRTrack.LINALG,
            ptr_model="axis_info",
            gpgpu_cap=GPGPUCapability(
                num_warps=1,
                warp_size=1,
                shared_mem_size=2**20,
                num_stages=1,
            ),
        )

    def validate_environment(self) -> Tuple[bool, str]:
        return False, "CPU reference backend not yet implemented"

    def lower_anchor_ir_to_target(self, anchor_ir: Any, metadata: dict) -> bytes:
        raise NotImplementedError(
            "CPU reference backend not yet implemented. "
            "Will use MLIR → LLVM IR → host compilation."
        )

    def create_launcher(self, binary: bytes, signature: dict, metadata: dict) -> Callable:
        raise NotImplementedError("CPU launcher not yet implemented.")

    def get_op_coverage(self) -> Dict[str, str]:
        return {
            "tt.dot": "fallback",
            "tt.load": "optimal",
            "tt.store": "optimal",
            "tt.reduce": "optimal",
            "tt.scan": "fallback",
            "tt.atomic_rmw": "emulated",
            "tt.atomic_cas": "emulated",
            "tt.broadcast": "optimal",
            "tt.splat": "optimal",
        }
