"""Algorithm plugins for FrugalOPL."""

from .algorithm_base import AlgorithmPlugin, Metrics, Signals
from .plugin_registry import (
    ConfiguredPluginFactory,
    plugin_class_from_name,
    plugin_factory_from_config,
    plugin_factory_from_name,
    register_plugin_class,
)
from .actor_test_plugin import ActorTestPlugin
from .grpo_plugin import GRPOPlugin
from .pg_opd_plugin import PGOPDPlugin
from .rlvr_plugin import RLVRPlugin

__all__ = [
    "ActorTestPlugin",
    "AlgorithmPlugin",
    "ConfiguredPluginFactory",
    "GRPOPlugin",
    "Metrics",
    "PGOPDPlugin",
    "RLVRPlugin",
    "Signals",
    "plugin_class_from_name",
    "plugin_factory_from_config",
    "plugin_factory_from_name",
    "register_plugin_class",
]
