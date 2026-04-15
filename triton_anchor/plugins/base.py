"""
BackendPlugin — Layer 3 Plugin Interface
==========================================

Backend plugins implement the final stages of compilation:
  AnchorIR → hardware-specific IR → binary (.so / .cubin / .hsaco)

Lifecycle (5 phases):
  1. Discovery:     ``entry_points("triton.backends")`` or explicit registration
  2. Configuration: ``validate_environment()`` checks toolchain availability
  3. Frontend Hook: ``on_ttir_ready()`` / ``on_anchor_ir_ready()``
  4. Compilation:   ``lower_anchor_ir_to_target()``
  5. Launch:        ``create_launcher()``

Stability guarantee:
  - Abstract methods are never removed
  - New hook methods always have default implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..hw_capability import HWCapability


class BackendPlugin(ABC):
    """Abstract base class for backend compiler plugins.

    Hardware vendors implement this interface and distribute it as
    an independent pip package with ``entry_points`` registration::

        # pyproject.toml
        [project.entry-points."triton.backends"]
        sophgo = "triton_backend_sophgo:SophgoPlugin"

    Minimal implementation requires only 4 methods:
      - ``name``
      - ``hw_capability``
      - ``lower_anchor_ir_to_target()``
      - ``create_launcher()``
    """

    # ═══════════════════════════════════════════════════════════════════
    # Discovery
    # ═══════════════════════════════════════════════════════════════════

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend name (e.g., 'sophgo', 'spacemit', 'usc')."""
        ...

    @property
    @abstractmethod
    def hw_capability(self) -> HWCapability:
        """Declarative hardware capability descriptor."""
        ...

    # ═══════════════════════════════════════════════════════════════════
    # Configuration
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    def validate_environment(self) -> Tuple[bool, str]:
        """Check that the hardware toolchain is available.

        Returns:
            A tuple ``(is_valid, message)``.  If ``is_valid`` is False,
            ``message`` should explain what's missing.
        """
        ...

    def get_adapter_preference(self) -> Optional[str]:
        """Override adapter selection for this backend.

        Returns:
            Adapter name (e.g., 'triton-linalg') or None for automatic.
        """
        return None

    def get_dsl_extensions(self) -> List[str]:
        """List of DSL extension namespaces this backend supports.

        Returns:
            Namespace list, e.g., ['smt'] for SpacemiT.
        """
        return []

    def get_allowed_dialects(self) -> List[str]:
        """List of MLIR dialect namespaces this backend injects into AnchorIR.

        These dialects are added to the AnchorIR whitelist during post-hook
        validation (Phase 2).  Use this to declare extension dialects that
        the backend's ``on_anchor_ir_ready()`` hook may inject.

        Returns:
            Dialect namespace list, e.g., ['ftvm'] for USC/FangHua GPU,
            ['xsmt', 'xsmt_async'] for SpacemiT.  Default: empty.
        """
        return []

    # ═══════════════════════════════════════════════════════════════════
    # Compilation
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    def lower_anchor_ir_to_target(self, anchor_ir: Any, metadata: dict) -> bytes:
        """Convert AnchorIR to target binary.

        This is the main compilation method.  The implementation should:
        1. Apply backend-specific IR transformations
        2. Invoke hardware vendor toolchain
        3. Return the compiled binary

        Args:
            anchor_ir: AnchorIR module (MLIR Module or text string).
            metadata: Compilation metadata (mutated in-place with output paths).

        Returns:
            Compiled binary as bytes (.so, .cubin, etc.).
        """
        ...

    @abstractmethod
    def create_launcher(self, binary: bytes, signature: dict, metadata: dict) -> Callable:
        """Create a callable kernel launcher.

        Args:
            binary: Compiled binary from ``lower_anchor_ir_to_target()``.
            signature: Kernel argument signature.
            metadata: Compilation metadata.

        Returns:
            A Python callable that launches the kernel on hardware.
        """
        ...

    # ═══════════════════════════════════════════════════════════════════
    # Hooks — Optional extension points
    # ═══════════════════════════════════════════════════════════════════

    def on_ttir_ready(self, ttir_module: Any, metadata: dict) -> None:
        """Hook ②: Called after TTIR optimization, before Adapter conversion.

        Use this to inject hardware attributes or apply backend-specific
        TTIR transformations.

        Default: no-op.
        """
        pass

    def on_anchor_ir_ready(self, anchor_ir: Any, metadata: dict) -> Any:
        """Hook ④: Called after AnchorIR generation, before backend compilation.

        Use this to apply backend-specific optimizations on AnchorIR.

        Args:
            anchor_ir: The AnchorIR module.
            metadata: Compilation metadata.

        Returns:
            The (possibly modified) AnchorIR module.

        Default: pass-through (return anchor_ir unchanged).
        """
        return anchor_ir

    def get_op_coverage(self) -> Dict[str, str]:
        """Declare which Triton ops this backend supports.

        Returns:
            A dict mapping Triton op names to support status:
            ``{"tt.dot": "optimal", "tt.atomic_rmw": "unsupported", ...}``

            Valid statuses: "optimal", "fallback", "emulated", "unsupported"
        """
        return {}

    # ═══════════════════════════════════════════════════════════════════
    # Compatibility — bridge to existing BaseBackend interface
    # ═══════════════════════════════════════════════════════════════════

    def to_base_backend_args(self) -> dict:
        """Generate arguments compatible with existing BaseBackend.add_stages().

        This enables gradual migration from the old BaseBackend interface
        to the new BackendPlugin interface.
        """
        hw = self.hw_capability
        return {
            "target": hw.to_gpu_target(),
            "paradigm": hw.compute_paradigm.value,
            "lowering_path": hw.lowering_path,
        }

    def load_dialects(self, context: Any) -> None:
        """Load backend-specific MLIR dialects into the given context.

        Default: no-op.  Override if the backend defines custom MLIR dialects.
        """
        pass

    def get_codegen_implementation(self) -> dict:
        """Return code generation callbacks for compatibility with BaseBackend.

        Default: minimal dot size of (1, 1, 1).
        """
        return {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
