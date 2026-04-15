#!/usr/bin/env python3
"""
Standalone test runner for triton-anchor.
Run from the project root:
    cd triton-anchor && python3 run_tests.py
"""
import sys
import os

# Ensure package is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_hw_capability():
    from triton_anchor.hw_capability import (
        HWCapability, ComputeParadigm, TensorCapability,
        MatrixCapability, GPGPUCapability,
    )

    # Sophgo TPU
    hw = HWCapability(
        name="sophgo-bm1684x", arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        lowering_path="linalg", ptr_model="axis_info",
        tensor_cap=TensorCapability(num_cores=8),
    )
    assert hw.name == "sophgo-bm1684x"
    assert hw.compute_paradigm == ComputeParadigm.TENSOR_PROCESSOR
    target = hw.to_gpu_target()
    assert isinstance(target, dict) or hasattr(target, "backend")
    print(f"  [PASS] Sophgo HWCapability: {hw.name}, target={target}")

    # SpacemiT AME
    hw2 = HWCapability(
        name="spacemit-x60", arch_family="riscv",
        compute_paradigm=ComputeParadigm.AME_MATRIX,
        lowering_path="linalg", ptr_model="structured",
        matrix_cap=MatrixCapability(num_matrix_registers=8),
    )
    print(f"  [PASS] SpacemiT HWCapability: {hw2.name}")

    # USC GPU
    hw3 = HWCapability(
        name="usc-gpu", arch_family="gpu",
        compute_paradigm=ComputeParadigm.GPGPU,
        lowering_path="triton_gpu", ptr_model="gpu",
        gpgpu_cap=GPGPUCapability(num_warps=4),
    )
    assert hw3.lowering_path == "triton_gpu"
    print(f"  [PASS] USC HWCapability: lowering={hw3.lowering_path}")

    # Validation: missing cap
    try:
        HWCapability(
            name="bad", arch_family="riscv",
            compute_paradigm=ComputeParadigm.AME_MATRIX,
            lowering_path="linalg", ptr_model="structured",
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [PASS] Validation: missing matrix_cap detected")

    # Validation: wrong lowering path
    try:
        HWCapability(
            name="bad", arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            lowering_path="triton_gpu", ptr_model="axis_info",
            tensor_cap=TensorCapability(),
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [PASS] Validation: wrong lowering_path detected")


def test_anchor_ir():
    from triton_anchor.anchor_ir import AnchorIRValidator, AnchorIRError

    v = AnchorIRValidator()

    # Valid IR
    valid = """
    func.func @kernel(%arg0: memref<128xf32>) {
      %c0 = arith.constant 0 : index
      %val = memref.load %arg0[%c0] : memref<128xf32>
      return
    }
    """
    assert v.is_valid(valid)
    print("  [PASS] Valid Linalg IR accepted")

    # Invalid: tt.* ops
    invalid = """
    func.func @kernel(%arg0: !tt.ptr<f32>) {
      %0 = tt.load %arg0 : f32
      return
    }
    """
    violations = v.validate(invalid)
    assert len(violations) > 0
    assert any(viol.dialect == "tt" for viol in violations)
    print(f"  [PASS] Invalid IR: {len(violations)} violations (tt.* detected)")

    # Custom extensions
    v2 = AnchorIRValidator(extra_allowed={"xsmt"})
    ext_ir = """
    func.func @kernel() {
      %0 = xsmt.alloc : memref<128xf32>
      return
    }
    """
    assert v2.is_valid(ext_ir)
    print("  [PASS] Custom extension dialect accepted")


def test_registries():
    from triton_anchor.adapters.base import ITritonToLinalgAdapter
    from triton_anchor.adapters.registry import AdapterRegistry, AdapterNotFoundError
    from triton_anchor.plugins.registry import PluginRegistry
    from triton_anchor.plugins.base import BackendPlugin
    from triton_anchor.hw_capability import (
        HWCapability, ComputeParadigm, TensorCapability,
    )

    # Mock adapter
    class MockAdapter(ITritonToLinalgAdapter):
        def name(self): return "mock"
        def convert(self, m, md, ctx=None): return m

    AdapterRegistry.reset()
    AdapterRegistry.register(MockAdapter())
    assert AdapterRegistry.get("mock") is not None
    print(f"  [PASS] AdapterRegistry: {AdapterRegistry.list_adapters()}")

    # Adapter selection
    hw = HWCapability(
        name="test", arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        lowering_path="linalg", ptr_model="axis_info",
        preferred_adapter="mock",
        tensor_cap=TensorCapability(),
    )
    a = AdapterRegistry.get_adapter(hw)
    assert a.name() == "mock"
    print("  [PASS] Adapter selection by preference")

    # Plugin registry
    PluginRegistry.reset()
    from triton_anchor.plugins.sophgo_plugin import SophgoPlugin
    from triton_anchor.plugins.spacemit_plugin import SpacemiTPlugin
    PluginRegistry.register(SophgoPlugin())
    PluginRegistry.register(SpacemiTPlugin())
    plugins = PluginRegistry.list_plugins()
    assert len(plugins) == 2
    print(f"  [PASS] PluginRegistry: {len(plugins)} plugins")
    for p in plugins:
        print(f"    {p['name']}: paradigm={p['paradigm']}, ok={p['environment_ok']}")


def test_op_coverage():
    from triton_anchor.op_coverage import OpCoverageMatrix, OpQuality
    from triton_anchor.plugins.sophgo_plugin import SophgoPlugin
    from triton_anchor.plugins.spacemit_plugin import SpacemiTPlugin

    matrix = OpCoverageMatrix()
    matrix.register_plugin(SophgoPlugin())
    matrix.register_plugin(SpacemiTPlugin())

    # Check specific op
    s = matrix.check_op("tt.dot", "sophgo")
    assert s.quality == OpQuality.OPTIMAL
    print("  [PASS] tt.dot on sophgo: optimal")

    s2 = matrix.check_op("tt.atomic_rmw", "spacemit")
    assert s2.quality == OpQuality.UNSUPPORTED
    print("  [PASS] tt.atomic_rmw on spacemit: unsupported")

    # Gap analysis
    gaps = matrix.find_gaps("spacemit")
    assert "tt.atomic_rmw" in gaps
    print(f"  [PASS] SpacemiT gaps: {list(gaps.keys())}")

    # Report
    report = matrix.generate_report()
    assert "sophgo" in report
    print(f"  [PASS] Report: {len(report)} chars")


def test_dsl_extensions():
    from triton_anchor.extensions.base import DSLExtensionPlugin, BuiltinSpec
    from triton_anchor.extensions.registry import DSLExtensionRegistry

    DSLExtensionRegistry.reset()
    DSLExtensionRegistry.discover()
    exts = DSLExtensionRegistry.list_extensions()
    print(f"  [PASS] DSLExtensionRegistry: {len(exts)} extensions")

    # Test BuiltinSpec creation
    spec = BuiltinSpec(name="dot", arg_types=["tensor", "tensor"], ret_type="tensor")
    assert spec.name == "dot"
    print(f"  [PASS] BuiltinSpec: {spec.name}")


def test_usc_triton_gpu_path():
    from triton_anchor.plugins.usc_plugin import USCPlugin

    usc = USCPlugin()
    hw = usc.hw_capability
    assert hw.lowering_path == "triton_gpu"
    assert hw.compute_paradigm.value == "gpgpu"
    print(f"  [PASS] USC: lowering={hw.lowering_path}, skips Adapter layer")


if __name__ == "__main__":
    tests = [
        ("HWCapability & ComputeParadigm", test_hw_capability),
        ("AnchorIR Validator", test_anchor_ir),
        ("Adapter & Plugin Registries", test_registries),
        ("OpCoverageMatrix", test_op_coverage),
        ("DSL Extension System", test_dsl_extensions),
        ("USC TritonGPU Path", test_usc_triton_gpu_path),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
