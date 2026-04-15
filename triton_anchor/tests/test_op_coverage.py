"""Tests for OpCoverageMatrix."""

import pytest
from triton_anchor.op_coverage import OpCoverageMatrix, OpQuality
from triton_anchor.plugins.sophgo_plugin import SophgoPlugin
from triton_anchor.plugins.spacemit_plugin import SpacemiTPlugin


class TestOpCoverageMatrix:
    def setup_method(self):
        self.matrix = OpCoverageMatrix()
        self.matrix.register_plugin(SophgoPlugin())
        self.matrix.register_plugin(SpacemiTPlugin())

    def test_check_op_optimal(self):
        status = self.matrix.check_op("tt.dot", "sophgo")
        assert status.quality == OpQuality.OPTIMAL
        assert status.is_supported

    def test_check_op_unsupported(self):
        status = self.matrix.check_op("tt.histogram", "sophgo")
        assert status.quality == OpQuality.UNSUPPORTED
        assert not status.is_supported

    def test_check_op_unknown_backend(self):
        status = self.matrix.check_op("tt.dot", "nonexistent")
        assert status.quality == OpQuality.UNSUPPORTED

    def test_find_gaps(self):
        gaps = self.matrix.find_gaps("spacemit")
        assert "tt.atomic_rmw" in gaps
        assert gaps["tt.atomic_rmw"].quality == OpQuality.UNSUPPORTED

    def test_generate_report(self):
        report = self.matrix.generate_report()
        assert "sophgo" in report
        assert "spacemit" in report
        assert "tt.dot" in report

    def test_validate_kernel_ops(self):
        # All supported
        unsupported = self.matrix.validate_kernel_ops(
            ["tt.dot", "tt.load", "tt.reduce"], "sophgo"
        )
        assert len(unsupported) == 0

        # Some unsupported
        unsupported = self.matrix.validate_kernel_ops(
            ["tt.dot", "tt.histogram"], "sophgo"
        )
        assert len(unsupported) == 1
        assert unsupported[0].op_name == "tt.histogram"

    def test_empty_matrix(self):
        empty = OpCoverageMatrix()
        report = empty.generate_report()
        assert "No backends" in report
