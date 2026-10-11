"""Validation and lookup helpers for Controller-managed signal pipelines."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final


_WORKER_CLASSES: Final[frozenset[str]] = frozenset({"Controller", "Actor", "Teacher", "Learner"})
@dataclass(frozen=True)
class PipelineNode:
    """One configured Worker node in signal-flow order."""

    node_id: str
    worker_class: str
    plugin_hook: str | None = None
    objective_fn: str | None = None
    cpu_offload: bool = False


@dataclass(frozen=True)
class TrainingPipeline:
    """A validated linear Controller -> Actor -> Teachers -> Learner flow."""

    nodes: tuple[PipelineNode, ...]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    @property
    def teacher_nodes(self) -> tuple[PipelineNode, ...]:
        return self.nodes_for_class("Teacher")

    @property
    def has_teacher(self) -> bool:
        return bool(self.teacher_nodes)

    def node(self, node_id: str) -> PipelineNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ValueError(f"Pipeline does not contain node {node_id!r}.")

    def nodes_for_class(self, worker_class: str) -> tuple[PipelineNode, ...]:
        """Return configured nodes for one Worker class in pipeline order."""

        return tuple(node for node in self.nodes if node.worker_class == worker_class)

    def node_for_class(self, worker_class: str) -> PipelineNode:
        """Return the one configured node for a singleton Worker class."""

        nodes = self.nodes_for_class(worker_class)
        if len(nodes) != 1:
            raise ValueError(
                f"Pipeline must contain exactly one {worker_class} node; found {len(nodes)}."
            )
        return nodes[0]

    def successor_of(self, node_id: str) -> str:
        for index, node in enumerate(self.nodes):
            if node.node_id == node_id:
                if index + 1 == len(self.nodes):
                    raise ValueError(f"Pipeline node {node_id!r} has no downstream node.")
                return self.nodes[index + 1].node_id
        raise ValueError(f"Pipeline does not contain node {node_id!r}.")

    def predecessor_of(self, node_id: str) -> str:
        for index, node in enumerate(self.nodes):
            if node.node_id == node_id:
                if index == 0:
                    raise ValueError(f"Pipeline node {node_id!r} has no upstream node.")
                return self.nodes[index - 1].node_id
        raise ValueError(f"Pipeline does not contain node {node_id!r}.")


def _optional_name(raw_node: Mapping[str, object], field: str) -> str | None:
    value = raw_node.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pipeline node {raw_node.get('id')!r}.{field} must be a non-empty string.")
    return value.strip()


def resolve_training_pipeline(config: Mapping[str, object]) -> TrainingPipeline:
    """Resolve a linear pipeline whose node IDs are their configuration keys.

    ``controller``, ``actor``, and ``learner`` retain their canonical IDs so
    their existing configuration sections and public queues remain stable.
    Every node between Actor and Learner may be a distinct ``Teacher`` named
    by its own top-level configuration key (for example ``teachera`` and
    ``teacherb``).
    """

    if not isinstance(config, Mapping):
        raise TypeError("Training configuration must be a mapping.")
    raw_pipeline = config.get("pipeline")
    if not isinstance(raw_pipeline, list) or not raw_pipeline:
        raise ValueError("Controller-managed training requires a non-empty top-level 'pipeline' list.")

    nodes: list[PipelineNode] = []
    seen_ids: set[str] = set()
    for index, raw_node in enumerate(raw_pipeline):
        if not isinstance(raw_node, Mapping):
            raise TypeError(f"pipeline[{index}] must be a mapping.")
        node_id = raw_node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(f"pipeline[{index}].id must be a non-empty string.")
        node_id = node_id.strip()
        if node_id in seen_ids:
            raise ValueError(f"pipeline contains duplicate node id {node_id!r}.")
        seen_ids.add(node_id)

        worker_class = raw_node.get("class")
        if not isinstance(worker_class, str) or worker_class not in _WORKER_CLASSES:
            supported = ", ".join(sorted(_WORKER_CLASSES))
            raise ValueError(f"pipeline node {node_id!r}.class must be one of: {supported}.")
        cpu_offload = raw_node.get("cpu_offload", False)
        if not isinstance(cpu_offload, bool):
            raise TypeError(f"pipeline node {node_id!r}.cpu_offload must be a boolean.")
        if worker_class in {"Controller", "Actor"} and cpu_offload:
            raise ValueError(f"{worker_class} does not support cpu_offload.")

        missing = object()
        node_config = config.get(node_id, missing)
        if node_config is missing:
            raise ValueError(f"pipeline node {node_id!r} requires a top-level {node_id!r} configuration mapping.")
        if not isinstance(node_config, Mapping):
            raise TypeError(f"Top-level configuration for pipeline node {node_id!r} must be a mapping.")

        plugin_hook = _optional_name(raw_node, "plugin_hook")
        objective_fn = _optional_name(raw_node, "objective_fn")
        if worker_class == "Controller" and (plugin_hook is not None or objective_fn is not None):
            raise ValueError("Controller does not bind plugin_hook or objective_fn.")
        if worker_class != "Controller" and plugin_hook is None:
            raise ValueError(f"pipeline node {node_id!r} requires plugin_hook.")
        if worker_class == "Learner" and objective_fn is None:
            raise ValueError(f"pipeline node {node_id!r} requires objective_fn.")
        if worker_class != "Learner" and objective_fn is not None:
            raise ValueError(f"pipeline node {node_id!r} is not a Learner and cannot bind objective_fn.")
        nodes.append(PipelineNode(node_id, worker_class, plugin_hook, objective_fn, cpu_offload))

    if len(nodes) < 3:
        raise ValueError("Pipeline must contain Controller, Actor, and Learner nodes.")
    classes = tuple(node.worker_class for node in nodes)
    if classes[0] != "Controller" or nodes[0].node_id != "controller":
        raise ValueError("Pipeline must begin with the canonical controller node.")
    if classes[1] != "Actor" or nodes[1].node_id != "actor":
        raise ValueError("Pipeline must place the canonical actor node immediately after controller.")
    if classes[-1] != "Learner" or nodes[-1].node_id != "learner":
        raise ValueError("Pipeline must end with the canonical learner node.")
    if any(worker_class != "Teacher" for worker_class in classes[2:-1]):
        raise ValueError("Only Teacher nodes may appear between Actor and Learner.")

    for worker_id, worker_config in config.items():
        if worker_id not in seen_ids and isinstance(worker_config, Mapping) and "gpu_ids" in worker_config:
            raise ValueError(
                f"Top-level worker configuration {worker_id!r} has no matching pipeline node. "
                "Remove the configuration or declare the node in pipeline."
            )
    return TrainingPipeline(tuple(nodes))


def resolve_worker_config(config: Mapping[str, object], node_id: str) -> dict[str, object]:
    """Merge one declared node's config over its optional class defaults.

    Node IDs remain the source of truth for concrete configuration sections.
    ``worker_defaults`` only supplies ordinary worker-class defaults (most
    usefully the generic placement for arbitrary ``Teacher`` IDs), so a
    Teacher-free pipeline never acquires an undeclared Worker configuration.
    """

    pipeline = resolve_training_pipeline(config)
    node = pipeline.node(node_id)
    node_config = config[node.node_id]
    if not isinstance(node_config, Mapping):  # Already checked above; keeps this public helper self-contained.
        raise TypeError(f"Top-level configuration for pipeline node {node.node_id!r} must be a mapping.")
    raw_defaults = config.get("worker_defaults", {})
    if not isinstance(raw_defaults, Mapping):
        raise TypeError("worker_defaults must be a mapping when provided.")
    class_defaults = raw_defaults.get(node.worker_class, {})
    if not isinstance(class_defaults, Mapping):
        raise TypeError(f"worker_defaults.{node.worker_class} must be a mapping.")

    merged: dict[str, object] = copy.deepcopy(dict(class_defaults))

    def overlay(target: dict[str, object], additions: Mapping[str, object]) -> None:
        for key, value in additions.items():
            current = target.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                nested = copy.deepcopy(dict(current))
                overlay(nested, value)
                target[str(key)] = nested
            else:
                target[str(key)] = copy.deepcopy(value)

    overlay(merged, node_config)
    return merged
