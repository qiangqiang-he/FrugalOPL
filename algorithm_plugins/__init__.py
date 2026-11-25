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
from .ratio_opsd_plugin import RatioOPSDPlugin
from .tcdo_plugin import TCDOPlugin
from .tcpo_v1_plugin import TCPOV1Plugin
from .tca_plugin import TCAPlugin
from .fkl_plugin import FKLPlugin
from .fkl_true_answer_plugin import FKLTrueAnswerPlugin
from .clip_opsd_plugin import ClipOPSDPlugin
from .tra_plugin import TRAPlugin
from .tra_privilege_plugins import TRASelfShotPlugin, TRARandomShotPlugin, TRATrueAnswerPlugin
from .tail_sup_plugin import TailSupPlugin
from .rado_plugin import RADOPlugin

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
    "RatioOPSDPlugin",
    "TCDOPlugin",
    "TCPOV1Plugin",
    "TCAPlugin",
    "FKLPlugin",
    "FKLTrueAnswerPlugin",
    "ClipOPSDPlugin",
    "TRAPlugin",
    "TRASelfShotPlugin",
    "TRARandomShotPlugin",
    "TRATrueAnswerPlugin",
    "TailSupPlugin",
    "RADOPlugin",
    "plugin_class_from_name",
    "plugin_factory_from_config",
    "plugin_factory_from_name",
    "register_plugin_class",
]
