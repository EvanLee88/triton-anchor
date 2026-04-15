"""
Sophgo Backend Plugin
======================

Wraps triton_race's existing RaceBackend as a BackendPlugin.

Compilation pipeline:
  AnchorIR → LinalgToPPL → ppl-compile → cmake build → .so

Dependencies:
  - triton_race must be installed (provides libtriton.so with race passes)
  - PPL_PROJECT_ROOT environment variable
  - PPLCOMPILE_PATH environment variable
  - TRITON_DUMP_DIR environment variable
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BackendPlugin
from ..hw_capability import (
    HWCapability,
    ComputeParadigm,
    TensorCapability,
)

logger = logging.getLogger(__name__)


class SophgoPlugin(BackendPlugin):
    """Backend plugin for Sophgo TPU (BM1684X / BM1690).

    This plugin wraps triton_race's existing compilation pipeline:
      1. LinalgToPPL pass (convert Linalg IR to PPL dialect)
      2. ppl-compile tool (compile PPL IR to object code)
      3. cmake build (link into .so with DMA + runtime support)

    Environment requirements:
      - ``PPL_PROJECT_ROOT``: Sophgo PPL SDK root
      - ``PPLCOMPILE_PATH``: Path to ppl-compile tool
      - ``TRITON_DUMP_DIR``: Working directory for intermediate IR files
    """

    @property
    def name(self) -> str:
        return "sophgo"

    @property
    def hw_capability(self) -> HWCapability:
        from ..anchor_ir import AnchorIRTrack
        return HWCapability(
            name="sophgo-bm1684x",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            anchor_ir_track=AnchorIRTrack.LINALG,
            ptr_model="axis_info",
            preferred_adapter="triton-linalg",
            tensor_cap=TensorCapability(
                num_cores=8,
                local_mem_size=16 * 1024 * 1024,   # 16MB per core
                global_mem_size=12 * 1024**3,       # 12GB HBM
                dma_channels=2,
                supported_dtypes={"fp32", "fp16", "int8", "int32"},
                max_tensor_dims=4,
            ),
            num_cores=8,
        )

    def validate_environment(self) -> Tuple[bool, str]:
        """Check Sophgo PPL toolchain availability."""
        checks = []

        # Check triton_race
        try:
            from triton._C.libtriton import passes
            if hasattr(passes, 'race'):
                checks.append(("triton_race", True, ""))
            else:
                checks.append(("triton_race", False, "passes.race not available"))
        except ImportError:
            checks.append(("triton_race", False, "triton._C.libtriton not importable"))

        # Check PPL toolchain
        ppl_root = os.environ.get("PPL_PROJECT_ROOT", "")
        checks.append(("PPL_PROJECT_ROOT", bool(ppl_root), ppl_root or "not set"))

        pplcompile = os.environ.get("PPLCOMPILE_PATH", "")
        checks.append(("PPLCOMPILE_PATH", bool(pplcompile), pplcompile or "not set"))

        dump_dir = os.environ.get("TRITON_DUMP_DIR", "")
        checks.append(("TRITON_DUMP_DIR", bool(dump_dir), dump_dir or "not set"))

        failed = [(name, msg) for name, ok, msg in checks if not ok]
        if failed:
            details = "; ".join(f"{n}: {m}" for n, m in failed)
            return False, f"Missing: {details}"

        return True, "Sophgo PPL toolchain ready"

    def get_adapter_preference(self) -> Optional[str]:
        return "triton-linalg"

    def lower_anchor_ir_to_target(self, anchor_ir: Any, metadata: dict) -> bytes:
        """Convert AnchorIR to .so via PPL pipeline.

        Delegates to triton_race's existing ``_make_pplir()`` + ``_pplir_to_so()``.
        """
        try:
            from triton._C.libtriton import ir, passes
        except ImportError:
            raise RuntimeError("triton_race not installed")

        # Stage 1: Linalg → PPL IR
        pm = ir.pass_manager(anchor_ir.context)
        pm.enable_debug()
        pm.enable_verifier(False)
        passes.race.linalg_to_ppl.add_linalg_to_ppl(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

        try:
            pm.run(anchor_ir)
        except Exception as e:
            strict = os.getenv("TRITON_RACE_STRICT_PPLIR", "0") == "1"
            if strict:
                raise
            logger.warning(f"PPLIR lowering recoverable failure: {e}")

        # Stage 2: PPL IR → .so (delegates to triton_race utility)
        # Import the existing function from triton_race backend
        try:
            # triton_race registers as 'sophgo' backend, not 'race'
            import importlib
            race_compiler = importlib.import_module(
                "triton.backends.sophgo.compiler"
            )
            return race_compiler._pplir_to_so(anchor_ir, metadata)
        except (ImportError, ModuleNotFoundError):
            # Fallback: try direct import from third_party source layout
            try:
                from triton.third_party.sophgo.backend.compiler import _pplir_to_so
                return _pplir_to_so(anchor_ir, metadata)
            except ImportError:
                raise RuntimeError(
                    "Cannot find _pplir_to_so. Ensure triton_race is installed "
                    "and the 'sophgo' backend is available."
                )

    def create_launcher(self, binary: bytes, signature: dict, metadata: dict) -> Callable:
        """Create a Sophgo TPU kernel launcher.

        Delegates to triton_race's SOPHGOLauncher.
        """
        try:
            # triton_race registers as 'sophgo' backend
            from triton.backends.sophgo.driver import SOPHGOLauncher

            # Create a minimal source-like object for the launcher
            class _StubSrc:
                def __init__(self):
                    self.constants = {}
                    self.signature = signature
                    class _fn:
                        arg_names = list(signature.keys())
                        constexprs = set()
                    self.fn = _fn()

            # Create a minimal metadata-like object
            from collections import namedtuple
            MetaTuple = namedtuple('Meta', ['name', 'so_path'])
            meta = MetaTuple(
                name=metadata.get("name", "kernel"),
                so_path=metadata.get("so_path", ""),
            )

            return SOPHGOLauncher(_StubSrc(), meta)
        except ImportError:
            raise RuntimeError(
                "SOPHGOLauncher not available. Install triton_race."
            )

    # ── Hooks ────────────────────────────────────────────────────────

    def on_ttir_ready(self, ttir_module: Any, metadata: dict) -> None:
        """Inject Sophgo-specific attributes into TTIR."""
        from ..pipeline import inject_hw_attributes
        inject_hw_attributes(ttir_module, self.hw_capability, metadata)

    def load_dialects(self, context: Any) -> None:
        """Load triton-linalg + PPL dialects."""
        try:
            from triton._C.libtriton import passes
            passes.race.load_dialects(context)
        except (ImportError, AttributeError):
            logger.warning("Could not load race dialects")

    def get_op_coverage(self) -> Dict[str, str]:
        return {
            "tt.dot": "optimal",
            "tt.load": "optimal",
            "tt.store": "optimal",
            "tt.reduce": "optimal",
            "tt.scan": "fallback",
            "tt.atomic_rmw": "optimal",
            "tt.atomic_cas": "optimal",
            "tt.broadcast": "optimal",
            "tt.splat": "optimal",
            "tt.expand_dims": "optimal",
            "tt.trans": "optimal",
            "tt.cat": "fallback",
            "tt.histogram": "unsupported",
            "tt.sort": "unsupported",
        }
