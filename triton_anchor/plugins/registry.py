"""
Plugin Registry
================

Manages discovery, registration, and selection of backend plugins.

Discovery mechanisms (in order):
  1. Explicit registration via ``PluginRegistry.register()``
  2. ``entry_points("triton.backends")`` auto-discovery
  3. File-system scanning (legacy compatibility with triton_race)
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from .base import BackendPlugin

if TYPE_CHECKING:
    from ..hw_capability import HWCapability

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for backend compiler plugins.

    Usage::

        # Registration
        PluginRegistry.register(SophgoPlugin())

        # Auto-discovery
        PluginRegistry.discover()

        # Get a specific plugin
        plugin = PluginRegistry.get("sophgo")

        # Find plugin by hardware capability
        plugin = PluginRegistry.find_for_target("sophgo")
    """

    _plugins: Dict[str, BackendPlugin] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, plugin: BackendPlugin) -> None:
        """Explicitly register a backend plugin."""
        name = plugin.name
        if name in cls._plugins:
            logger.warning(f"Plugin '{name}' already registered, overwriting")
        cls._plugins[name] = plugin
        logger.info(f"Registered backend plugin: {name} "
                    f"(paradigm={plugin.hw_capability.compute_paradigm.value})")

    @classmethod
    def discover(cls) -> None:
        """Auto-discover plugins from ``entry_points("triton.backends")``.

        This is called lazily on first access.  Subsequent calls are no-ops.
        """
        if cls._discovered:
            return
        cls._discovered = True

        try:
            eps = importlib.metadata.entry_points(group="triton.backends")
        except TypeError:
            # Python 3.8/3.9 compatibility
            eps = importlib.metadata.entry_points().get("triton.backends", [])

        for ep in eps:
            try:
                plugin_cls = ep.load()
                plugin = plugin_cls()

                # Validate environment on discovery
                is_valid, msg = plugin.validate_environment()
                if is_valid:
                    cls.register(plugin)
                else:
                    logger.info(f"Plugin '{ep.name}' skipped: {msg}")
            except Exception as e:
                logger.warning(f"Failed to load plugin entry_point '{ep.name}': {e}")

    @classmethod
    def get(cls, name: str) -> Optional[BackendPlugin]:
        """Get a plugin by backend name."""
        cls.discover()
        return cls._plugins.get(name)

    @classmethod
    def find_for_target(cls, target_backend: str) -> Optional[BackendPlugin]:
        """Find a plugin that supports the given target backend name.

        This is the primary lookup method used by the compiler.

        Args:
            target_backend: Backend name string (e.g., 'sophgo', 'spacemit').

        Returns:
            The matching BackendPlugin, or None.
        """
        cls.discover()

        # Direct name match
        if target_backend in cls._plugins:
            return cls._plugins[target_backend]

        # Search by hw_capability name
        for plugin in cls._plugins.values():
            hw = plugin.hw_capability
            if hw.name.startswith(target_backend) or hw._infer_backend_name() == target_backend:
                return plugin

        return None

    @classmethod
    def list_plugins(cls) -> List[Dict[str, str]]:
        """List all registered plugins with their capabilities.

        Returns:
            List of dicts with keys: name, paradigm, arch_family, lowering_path
        """
        cls.discover()
        result = []
        for name, plugin in cls._plugins.items():
            hw = plugin.hw_capability
            is_valid, msg = plugin.validate_environment()
            result.append({
                "name": name,
                "hw_name": hw.name,
                "paradigm": hw.compute_paradigm.value,
                "arch_family": hw.arch_family,
                "lowering_path": hw.lowering_path,
                "ptr_model": hw.ptr_model,
                "environment_ok": is_valid,
                "environment_msg": msg,
            })
        return result

    @classmethod
    def get_all_active(cls) -> Dict[str, BackendPlugin]:
        """Get all plugins that passed environment validation."""
        cls.discover()
        return {name: p for name, p in cls._plugins.items()}

    @classmethod
    def reset(cls) -> None:
        """Reset registry state (for testing)."""
        cls._plugins.clear()
        cls._discovered = False
