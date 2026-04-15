"""
SpacemiT Backend Plugin — Stub
================================

Interface reservation for SpacemiT RISC-V AME (Advanced Matrix Extension)
backend.  Full implementation requires spine-triton toolchain.

Compilation pipeline (when implemented):
  AnchorIR → spine-opt → LLVM IR → opt → llc → g++ → .so

Dependencies (not yet integrated):
  - spine-triton: spine-opt compiler tool
  - RISC-V AME toolchain (llc with AME ISA support)
"""

from __future__ import annotations

import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BackendPlugin
from ..hw_capability import (
    HWCapability,
    ComputeParadigm,
    MatrixCapability,
)


class SpacemiTPlugin(BackendPlugin):
    """Backend plugin for SpacemiT RISC-V AME (X60).

    Status: **STUB** — provides interface and HWCapability declaration.
    Full implementation requires spine-triton integration.
    """

    @property
    def name(self) -> str:
        return "spacemit"

    @property
    def hw_capability(self) -> HWCapability:
        from ..anchor_ir import AnchorIRTrack
        return HWCapability(
            name="spacemit-x60",
            arch_family="riscv",
            compute_paradigm=ComputeParadigm.AME_MATRIX,
            anchor_ir_track=AnchorIRTrack.LINALG,
            ptr_model="structured",
            preferred_adapter="triton-shared",
            matrix_cap=MatrixCapability(
                num_matrix_registers=8,
                tile_shape=(8, 8),
                supported_dtypes={"fp32", "fp16", "int8"},
                has_accumulator_tiles=True,
                vector_length=256,
                supports_pointwise=True,
            ),
            num_cores=4,
        )

    def validate_environment(self) -> Tuple[bool, str]:
        spine_opt = shutil.which("spine-opt")
        if not spine_opt:
            return False, (
                "spine-opt not found on PATH. "
                "Install spine-triton: https://github.com/spacemit-com/spine-triton"
            )
        return True, f"spine-opt found at {spine_opt}"

    def get_adapter_preference(self) -> Optional[str]:
        return "triton-shared"

    def get_dsl_extensions(self) -> List[str]:
        return ["smt"]

    def lower_anchor_ir_to_target(self, anchor_ir: Any, metadata: dict) -> bytes:
        raise NotImplementedError(
            "SpacemiT backend not yet integrated. "
            "Requires spine-triton toolchain (spine-opt → LLVM → .so).\n"
            "See: https://github.com/spacemit-com/spine-triton"
        )

    def create_launcher(self, binary: bytes, signature: dict, metadata: dict) -> Callable:
        raise NotImplementedError(
            "SpacemiT launcher not yet integrated."
        )

    def get_op_coverage(self) -> Dict[str, str]:
        return {
            "tt.dot": "optimal",       # AME matrix multiply
            "tt.load": "optimal",      # tile load
            "tt.store": "optimal",     # tile store
            "tt.reduce": "optimal",    # RVV vector reduction
            "tt.scan": "fallback",     # sequential fallback
            "tt.atomic_rmw": "unsupported",
            "tt.broadcast": "optimal",
            "tt.splat": "optimal",
            "tt.expand_dims": "optimal",
            "tt.trans": "optimal",
        }
