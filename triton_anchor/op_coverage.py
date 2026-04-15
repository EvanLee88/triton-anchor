"""
OpCoverageMatrix — Cross-Backend Op Support Matrix
=====================================================

Provides compile-time detection of op support gaps across backends.
Instead of runtime crashes, users get clear compile-time reports.

Usage::

    matrix = OpCoverageMatrix()
    matrix.register_plugin(SophgoPlugin())
    matrix.register_plugin(SpacemiTPlugin())

    # Check single op
    status = matrix.check_op("tt.dot", "sophgo")
    assert status.quality == "optimal"

    # Generate coverage report
    report = matrix.generate_report()
    print(report)

    # Find unsupported ops
    gaps = matrix.find_gaps("sophgo")
    for op, status in gaps.items():
        print(f"  {op}: {status}")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .plugins.base import BackendPlugin


class OpQuality(Enum):
    """Quality level of op support on a backend."""
    OPTIMAL = "optimal"          # Native hardware support, best performance
    FALLBACK = "fallback"        # Functional but sub-optimal implementation
    EMULATED = "emulated"        # Software emulation, potentially slow
    UNSUPPORTED = "unsupported"  # Not available — compile-time error


@dataclass
class OpStatus:
    """Status of a single op on a single backend."""
    quality: OpQuality
    backend: str
    op_name: str
    notes: str = ""

    @property
    def is_supported(self) -> bool:
        return self.quality != OpQuality.UNSUPPORTED


class OpCoverageMatrix:
    """Cross-backend operator coverage analysis.

    Aggregates op coverage declarations from all registered backend plugins
    and provides query / reporting facilities.
    """

    # Standard Triton ops to track
    STANDARD_OPS = [
        "tt.dot", "tt.load", "tt.store",
        "tt.reduce", "tt.scan",
        "tt.atomic_rmw", "tt.atomic_cas",
        "tt.broadcast", "tt.splat", "tt.expand_dims",
        "tt.trans", "tt.cat",
        "tt.histogram", "tt.sort",
    ]

    def __init__(self):
        self._coverage: Dict[str, Dict[str, str]] = {}  # backend → {op → quality}

    def register_plugin(self, plugin: BackendPlugin) -> None:
        """Register a backend plugin's op coverage."""
        self._coverage[plugin.name] = plugin.get_op_coverage()

    def check_op(self, op_name: str, backend: str) -> OpStatus:
        """Check the support status of an op on a backend."""
        if backend not in self._coverage:
            return OpStatus(OpQuality.UNSUPPORTED, backend, op_name,
                          notes="Backend not registered")

        quality_str = self._coverage[backend].get(op_name, "unsupported")
        try:
            quality = OpQuality(quality_str)
        except ValueError:
            quality = OpQuality.UNSUPPORTED

        return OpStatus(quality, backend, op_name)

    def find_gaps(self, backend: str) -> Dict[str, OpStatus]:
        """Find ops that are unsupported or sub-optimal on a backend."""
        gaps = {}
        for op in self.STANDARD_OPS:
            status = self.check_op(op, backend)
            if status.quality in (OpQuality.UNSUPPORTED, OpQuality.EMULATED):
                gaps[op] = status
        return gaps

    def generate_report(self) -> str:
        """Generate a human-readable coverage report.

        Returns:
            Formatted table comparing op support across backends.
        """
        if not self._coverage:
            return "No backends registered."

        backends = sorted(self._coverage.keys())
        header = f"{'Op':<20} | " + " | ".join(f"{b:<12}" for b in backends)
        separator = "-" * len(header)

        lines = [header, separator]
        for op in self.STANDARD_OPS:
            cells = []
            for backend in backends:
                status = self.check_op(op, backend)
                symbol = {
                    OpQuality.OPTIMAL: "✅ optimal",
                    OpQuality.FALLBACK: "⚠️  fallback",
                    OpQuality.EMULATED: "🔶 emulated",
                    OpQuality.UNSUPPORTED: "❌ unsupported",
                }[status.quality]
                cells.append(f"{symbol:<12}")
            lines.append(f"{op:<20} | " + " | ".join(cells))

        return "\n".join(lines)

    def validate_kernel_ops(
        self, ops: List[str], backend: str
    ) -> List[OpStatus]:
        """Validate that all ops in a kernel are supported on the backend.

        Args:
            ops: List of Triton op names used in the kernel.
            backend: Target backend name.

        Returns:
            List of unsupported OpStatus (empty if all supported).
        """
        unsupported = []
        for op in ops:
            status = self.check_op(op, backend)
            if not status.is_supported:
                unsupported.append(status)
        return unsupported
