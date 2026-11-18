"""Algorithm plugins for FrugalOPL."""

from .algorithm_base import AlgorithmPlugin, Metrics, ObjectiveResult, Signals
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
from .pg_opsd_plugin import PGOPSDPlugin
from .kl_opsd_plugin import KLOPSDPlugin
from .ocsd_plugin import OCSDPlugin
from .iso_gpso_plugin import IsoGPSOPlugin
from .self_shot_opsd_plugin import SelfShotOPSDPlugin
from .tail_sup_plugin import TailSupPlugin

__all__ = [
    "ActorTestPlugin",
    "AlgorithmPlugin",
    "ConfiguredPluginFactory",
    "GRPOPlugin",
    "IsoGPSOPlugin",
    "KLOPSDPlugin",
    "Metrics",
    "ObjectiveResult",
    "OCSDPlugin",
    "PGOPDPlugin",
    "PGOPSDPlugin",
    "Signals",
    "SelfShotOPSDPlugin",
    "TailSupPlugin",
    "plugin_class_from_name",
    "plugin_factory_from_config",
    "plugin_factory_from_name",
    "register_plugin_class",
]
