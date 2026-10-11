"""Configuration-independent construction for registered algorithm plugins.

Training YAML names an algorithm class (for example ``PGOPDPlugin``), not a
module-level factory function.  This module owns the small, explicit bridge
from that class name to a worker-local plugin instance.  The bridge is kept
at module scope so it remains pickle-safe when the Actor, Teacher, and Learner
are launched with Python's ``spawn`` start method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import os

from utils.algorithm_runtime import AlgorithmRuntimeConfig, resolve_algorithm_runtime_config

from .algorithm_base import AlgorithmPlugin


_PLUGIN_CLASSES: dict[str, type[AlgorithmPlugin]] = {}
_TEST_PLUGIN_MODULES_ENV = "FRUGALOPL_TEST_PLUGIN_MODULES"


def _load_opt_in_test_plugins() -> None:
    """Import explicitly named test-only plugin modules when requested.

    Production configurations name only classes that this package registers
    itself.  GPU smoke tests occasionally need a purpose-built plugin without
    promoting it to that production registry, so their launcher may opt in
    through ``FRUGALOPL_TEST_PLUGIN_MODULES``.  Restrict the import surface to
    the repository's ``tests`` package; this is a test harness, not a general
    dynamic plugin loader.
    """

    raw_modules = os.environ.get(_TEST_PLUGIN_MODULES_ENV, "")
    for module_name in (item.strip() for item in raw_modules.split(",")):
        if not module_name:
            continue
        if not module_name.startswith("tests."):
            raise ValueError(
                f"{_TEST_PLUGIN_MODULES_ENV} may import only tests.* modules, got {module_name!r}."
            )
        importlib.import_module(module_name)


def register_plugin_class(plugin_class: type[AlgorithmPlugin]) -> type[AlgorithmPlugin]:
    """Register one concrete plugin class under its unqualified class name.

    Plugin modules call this decorator at import time.  Registration is
    deliberately explicit: configuration cannot import an arbitrary module or
    execute an arbitrary callable merely by changing a YAML value.
    """

    if not isinstance(plugin_class, type) or not issubclass(plugin_class, AlgorithmPlugin):
        raise TypeError("Only AlgorithmPlugin subclasses may be registered.")
    inherited_builder = getattr(
        AlgorithmPlugin.from_runtime_config,
        "__func__",
        AlgorithmPlugin.from_runtime_config,
    )
    plugin_builder = getattr(
        plugin_class.from_runtime_config,
        "__func__",
        plugin_class.from_runtime_config,
    )
    if plugin_builder is inherited_builder:
        raise TypeError(
            f"Registered plugin class {plugin_class.__name__!r} must implement "
            "from_runtime_config(tokenizer, runtime)."
        )
    name = plugin_class.__name__
    if not name or name.startswith("_"):
        raise ValueError("A registered plugin class must have a public class name.")
    existing = _PLUGIN_CLASSES.get(name)
    if existing is not None and existing is not plugin_class:
        raise ValueError(f"Algorithm plugin class name {name!r} is already registered.")
    _PLUGIN_CLASSES[name] = plugin_class
    return plugin_class


def plugin_class_from_name(plugin_name: str) -> type[AlgorithmPlugin]:
    """Return the explicitly registered plugin class named by configuration."""

    if not isinstance(plugin_name, str) or not plugin_name.strip():
        raise ValueError("algorithm.plugin must be a non-empty registered plugin class name.")
    normalized = plugin_name.strip()
    _load_opt_in_test_plugins()
    plugin_class = _PLUGIN_CLASSES.get(normalized)
    if plugin_class is None:
        choices = ", ".join(sorted(_PLUGIN_CLASSES)) or "(none registered)"
        raise ValueError(
            f"algorithm.plugin {normalized!r} is not registered. Available plugin classes: {choices}."
        )
    return plugin_class


def _configured_plugin_name(config: Mapping[str, object]) -> str:
    algorithm = config.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError("config must contain an algorithm mapping.")
    plugin_name = algorithm.get("plugin")
    if not isinstance(plugin_name, str) or not plugin_name.strip():
        raise ValueError("algorithm.plugin must be a non-empty registered plugin class name.")
    return plugin_name.strip()


@dataclass(frozen=True)
class ConfiguredPluginFactory:
    """A serializable callable that creates one pre-bound plugin per Worker."""

    plugin_name: str
    runtime: AlgorithmRuntimeConfig

    def __post_init__(self) -> None:
        normalized = self.plugin_name.strip() if isinstance(self.plugin_name, str) else self.plugin_name
        if not isinstance(normalized, str):
            raise ValueError("algorithm.plugin must be a non-empty registered plugin class name.")
        plugin_class_from_name(normalized)
        if not isinstance(self.runtime, AlgorithmRuntimeConfig):
            raise TypeError("runtime must be an AlgorithmRuntimeConfig.")
        object.__setattr__(self, "plugin_name", normalized)

    def __call__(self, tokenizer: object) -> AlgorithmPlugin:
        plugin_class = plugin_class_from_name(self.plugin_name)
        plugin = plugin_class.from_runtime_config(tokenizer, self.runtime)
        if not isinstance(plugin, AlgorithmPlugin):  # Defensive contract for third-party registrations.
            raise TypeError(f"{self.plugin_name}.from_runtime_config() must return an AlgorithmPlugin.")
        return plugin


def plugin_factory_from_name(
    plugin_name: str,
    runtime: AlgorithmRuntimeConfig,
) -> ConfiguredPluginFactory:
    """Create a spawn-safe factory bound to one resolved algorithm runtime."""

    return ConfiguredPluginFactory(plugin_name, runtime)


def plugin_factory_from_config(config: Mapping[str, object]) -> ConfiguredPluginFactory:
    """Resolve and bind one complete training configuration before Workers start."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    plugin_name = _configured_plugin_name(config)
    plugin_class = plugin_class_from_name(plugin_name)
    if plugin_class.requires_frozen_reference:
        run = config.get("run", {})
        policy = run.get("policy", {}) if isinstance(run, Mapping) else {}
        lora = policy.get("lora", {}) if isinstance(policy, Mapping) else {}
        if not isinstance(lora, Mapping) or lora.get("enabled") is not True:
            raise ValueError(f"{plugin_name} requires run.policy.lora.enabled=true to keep its reference Teacher fixed.")
    return plugin_factory_from_name(
        plugin_name,
        resolve_algorithm_runtime_config(config),
    )


__all__ = [
    "ConfiguredPluginFactory",
    "plugin_class_from_name",
    "plugin_factory_from_config",
    "plugin_factory_from_name",
    "register_plugin_class",
]
