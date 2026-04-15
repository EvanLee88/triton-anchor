"""
Unified Compiler — compile() with Plugin Hooks
=================================================

This is the unified compilation entry point that orchestrates all
four layers of the plugin architecture:

  Layer 0 (DSL Extensions)  → auto-discovered at import time
  Layer 1 (TTIR Pipeline)   → ``build_ttir_pipeline()``
  Layer 2 (Linalg Adapter)  → ``adapter.convert()``
  Layer 2.5 (AnchorIR)      → ``validator.validate()``
  Layer 3 (Backend Plugin)  → ``plugin.lower_anchor_ir_to_target()``

Hook injection points:
  ① DSL Extension injection  — before AST → TTIR
  ② on_ttir_ready            — after TTIR optimization
  ③ Adapter conversion       — TTIR → AnchorIR
  ④ on_anchor_ir_ready       — after AnchorIR generation
  ⑤ lower_anchor_ir_to_target — AnchorIR → binary
  ⑥ create_launcher          — binary → Python callable
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hw_capability import HWCapability
    from .plugins.base import BackendPlugin

logger = logging.getLogger(__name__)


def unified_compile(
    ttir_module: Any,
    plugin: BackendPlugin,
    metadata: Optional[dict] = None,
    validate_anchor_ir: bool = True,
) -> dict:
    """Unified compilation pipeline with all hook injection points.

    This function orchestrates the full compilation from an optimized TTIR
    module to a kernel binary, using the plugin-based architecture.

    Args:
        ttir_module: MLIR module after AST→TTIR conversion.
        plugin: The target backend plugin.
        metadata: Compilation metadata dict (created if None).
        validate_anchor_ir: Whether to validate AnchorIR compliance.

    Returns:
        Updated metadata dict containing:
          - All compilation artifacts
          - ``"binary"``: compiled kernel binary (bytes)
          - ``"so_path"``: path to .so file (if applicable)

    Raises:
        AnchorIRError: If AnchorIR validation fails.
        AdapterConversionError: If TTIR→Linalg conversion fails.
        RuntimeError: If backend compilation fails.

    Example::

        from triton_anchor import unified_compile
        from triton_anchor.plugins.sophgo_plugin import SophgoPlugin

        plugin = SophgoPlugin()
        metadata = {"name": "my_kernel"}
        result = unified_compile(ttir_module, plugin, metadata)
        binary = result["binary"]
    """
    if metadata is None:
        metadata = {}

    hw = plugin.hw_capability
    logger.info(f"Unified compile: target={plugin.name}, paradigm={hw.compute_paradigm.value}")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: TTIR Optimization (Layer 1 — core invariant)
    # ═══════════════════════════════════════════════════════════════════
    from .pipeline import make_ttir, inject_hw_attributes

    logger.debug("Stage 1: TTIR optimization (7 mandatory passes)")
    make_ttir(ttir_module, metadata, hw=hw)

    # Hook ②: on_ttir_ready
    logger.debug("Hook ②: on_ttir_ready")
    plugin.on_ttir_ready(ttir_module, metadata)
    inject_hw_attributes(ttir_module, hw, metadata)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: TTIR → Hardware-Aware IR (Layer 2)
    # ═══════════════════════════════════════════════════════════════════
    if hw.lowering_path == "linalg":
        # Linalg path: use Adapter
        from .adapters.registry import get_adapter
        from .anchor_ir import AnchorIRValidator, AnchorIRTrack

        adapter = get_adapter(hw)
        logger.debug(f"Stage 2: Adapter conversion (adapter={adapter.name()})")

        anchor_ir = adapter.convert(ttir_module, metadata)

        # ═══════════════════════════════════════════════════════════════
        # Stage 2.5: Two-Phase AnchorIR Validation (v0.1.3 contract)
        # ═══════════════════════════════════════════════════════════════
        if validate_anchor_ir:
            track = AnchorIRTrack(hw.lowering_path)
            validator = AnchorIRValidator(track=track)
            ir_text = str(anchor_ir) if not isinstance(anchor_ir, str) else anchor_ir

            # Phase 1 (pre-hook): base whitelist + forbidden only
            logger.debug("Validation Phase 1 (pre-hook): base whitelist check")
            pre_violations = validator.validate_pre_hook(ir_text)
            if pre_violations:
                logger.warning(
                    f"AnchorIR pre-hook validation: {len(pre_violations)} violation(s) "
                    f"(kernel={metadata.get('name', '?')})"
                )
                for v in pre_violations:
                    logger.warning(f"  {v}")

        # Hook ④: on_anchor_ir_ready (backend injects extension ops)
        logger.debug("Hook ④: on_anchor_ir_ready")
        anchor_ir = plugin.on_anchor_ir_ready(anchor_ir, metadata)

        if validate_anchor_ir:
            ir_text = str(anchor_ir) if not isinstance(anchor_ir, str) else anchor_ir

            # Phase 2 (post-hook): base + extension whitelist
            ext_allowed = set(plugin.get_allowed_dialects())
            logger.debug(f"Validation Phase 2 (post-hook): base + ext={ext_allowed}")
            post_violations = validator.validate_post_hook(ir_text, ext_allowed=ext_allowed)
            if post_violations:
                logger.warning(
                    f"AnchorIR post-hook validation: {len(post_violations)} violation(s) "
                    f"(kernel={metadata.get('name', '?')})"
                )
                for v in post_violations:
                    logger.warning(f"  {v}")

        # ═══════════════════════════════════════════════════════════════
        # Stage 3: Backend Compilation (Layer 3)
        # ═══════════════════════════════════════════════════════════════
        logger.debug(f"Stage 3: Backend compilation (plugin={plugin.name})")
        binary = plugin.lower_anchor_ir_to_target(anchor_ir, metadata)

    elif hw.lowering_path == "triton_gpu":
        # TritonGPU path: skip Adapter, go directly to backend
        logger.debug("Stage 2-3: TritonGPU path (skipping Adapter)")
        binary = plugin.lower_anchor_ir_to_target(ttir_module, metadata)

    else:
        raise ValueError(f"Unknown lowering_path: {hw.lowering_path}")

    metadata["binary"] = binary
    logger.info(f"Compilation complete: {len(binary)} bytes")

    return metadata


def compile_with_discovery(
    ttir_module: Any,
    target_backend: str,
    metadata: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Compile using auto-discovered backend plugin.

    This is the highest-level API — discovers the target plugin by name
    and delegates to ``unified_compile()``.

    Args:
        ttir_module: MLIR module after AST→TTIR conversion.
        target_backend: Backend name (e.g., 'sophgo', 'spacemit').
        metadata: Compilation metadata dict.
        **kwargs: Passed to ``unified_compile()``.

    Raises:
        RuntimeError: If no plugin found for target_backend.
    """
    from .plugins.registry import PluginRegistry

    plugin = PluginRegistry.find_for_target(target_backend)
    if plugin is None:
        available = PluginRegistry.list_plugins()
        raise RuntimeError(
            f"No backend plugin found for '{target_backend}'. "
            f"Available backends: {[p['name'] for p in available]}\n"
            f"Install a backend package: pip install triton-backend-{target_backend}"
        )

    return unified_compile(ttir_module, plugin, metadata, **kwargs)
