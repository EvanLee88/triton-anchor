"""Tests for Adapter and Plugin registries."""

import pytest
from triton_anchor.adapters.base import ITritonToLinalgAdapter
from triton_anchor.adapters.registry import AdapterRegistry, AdapterNotFoundError
from triton_anchor.plugins.base import BackendPlugin
from triton_anchor.plugins.registry import PluginRegistry
from triton_anchor.hw_capability import (
    HWCapability,
    ComputeParadigm,
    TensorCapability,
)


class MockAdapter(ITritonToLinalgAdapter):
    def name(self):
        return "mock-adapter"

    def convert(self, ttir_module, metadata, context=None):
        return ttir_module  # pass-through


class MockPlugin(BackendPlugin):
    @property
    def name(self):
        return "mock"

    @property
    def hw_capability(self):
        return HWCapability(
            name="mock-device",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            lowering_path="linalg",
            ptr_model="axis_info",
            tensor_cap=TensorCapability(),
        )

    def validate_environment(self):
        return True, "mock environment OK"

    def lower_anchor_ir_to_target(self, anchor_ir, metadata):
        return b"mock_binary"

    def create_launcher(self, binary, signature, metadata):
        return lambda: None


class TestAdapterRegistry:
    def setup_method(self):
        AdapterRegistry.reset()

    def test_register_and_get(self):
        adapter = MockAdapter()
        AdapterRegistry.register(adapter)
        assert AdapterRegistry.get("mock-adapter") is adapter

    def test_get_nonexistent(self):
        assert AdapterRegistry.get("nonexistent") is None

    def test_list_adapters(self):
        AdapterRegistry.register(MockAdapter())
        adapters = AdapterRegistry.list_adapters()
        assert "mock-adapter" in adapters

    def test_get_adapter_with_preference(self):
        AdapterRegistry.register(MockAdapter())
        hw = HWCapability(
            name="test",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            lowering_path="linalg",
            ptr_model="axis_info",
            preferred_adapter="mock-adapter",
            tensor_cap=TensorCapability(),
        )
        adapter = AdapterRegistry.get_adapter(hw)
        assert adapter.name() == "mock-adapter"

    def test_get_adapter_preference_not_found(self):
        hw = HWCapability(
            name="test",
            arch_family="tpu",
            compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
            lowering_path="linalg",
            ptr_model="axis_info",
            preferred_adapter="nonexistent",
            tensor_cap=TensorCapability(),
        )
        with pytest.raises(AdapterNotFoundError):
            AdapterRegistry.get_adapter(hw)


class TestPluginRegistry:
    def setup_method(self):
        PluginRegistry.reset()

    def test_register_and_get(self):
        plugin = MockPlugin()
        PluginRegistry.register(plugin)
        assert PluginRegistry.get("mock") is plugin

    def test_find_for_target(self):
        PluginRegistry.register(MockPlugin())
        found = PluginRegistry.find_for_target("mock")
        assert found is not None
        assert found.name == "mock"

    def test_list_plugins(self):
        PluginRegistry.register(MockPlugin())
        plugins = PluginRegistry.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "mock"
        assert plugins[0]["paradigm"] == "tensor"

    def test_environment_info(self):
        plugin = MockPlugin()
        PluginRegistry.register(plugin)
        plugins = PluginRegistry.list_plugins()
        assert plugins[0]["environment_ok"] is True
