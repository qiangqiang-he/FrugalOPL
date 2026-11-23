"""TCPO_v1: the public name for the validated three-branch TCDO objective."""
from collections.abc import Mapping
from .plugin_registry import register_plugin_class
from .tcdo_plugin import TCDOPlugin


@register_plugin_class
class TCPOV1Plugin(TCDOPlugin):
    """Keep the objective unchanged and accept checkpoints under its old name."""

    def training_state_dict(self):
        return {**super().training_state_dict(), "algorithm": "TCPO_v1"}

    def load_training_state_dict(self, state, *, step):
        if isinstance(state, Mapping) and state.get("algorithm") == "tcdo":
            state = {**state, "algorithm": "TCPO_v1"}
        super().load_training_state_dict(state, step=step)
