"""DAG-based lesion identity graph for tracking lesions across timepoints.

Implements ADR-010: each lesion observation is a node keyed as
``{lesion_id}@{timepoint_id}`` and directed edges represent identity links
from earlier to later observations.  The graph must remain acyclic (DAG).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from digital_twin_tumor.domain.models import Lesion, TimePoint


class LesionGraph:
    """Directed acyclic graph tracking lesion identity across timepoints.

    Nodes are keyed as ``"{lesion_id}@{timepoint_id}"`` and carry observation
    attributes.  Edges carry ``confidence``, ``method``, and
    ``user_override`` metadata.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Create an empty identity graph."""
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_observation(self, lesion: Lesion, timepoint: TimePoint) -> str:
        """Add a lesion observation as a graph node.

        Parameters
        ----------
        lesion:
            The ``Lesion`` domain object.
        timepoint:
            The ``TimePoint`` at which the lesion was observed.

        Returns
        -------
        str
            The node key ``"{lesion_id}@{timepoint_id}"``.
        """
        node_key = f"{lesion.lesion_id}@{timepoint.timepoint_id}"
        self._graph.add_node(
            node_key,
            lesion_id=lesion.lesion_id,
            timepoint_id=timepoint.timepoint_id,
            centroid=lesion.centroid,
            volume=lesion.volume_mm3,
            diameter=lesion.longest_diameter_mm,
            confidence=lesion.confidence,
            organ=lesion.organ,
            is_target=lesion.is_target,
            is_new=lesion.is_new,
            scan_date=str(timepoint.scan_date) if timepoint.scan_date else "",
        )
        return node_key

    def add_identity_link(
        self,
        source_key: str,
        target_key: str,
        confidence: float,
        method: str,
        user_override: bool = False,
    ) -> None:
        """Add a directed edge from an earlier observation to a later one.

        Parameters
        ----------
        source_key:
            Node key for the earlier observation.
        target_key:
            Node key for the later observation.
        confidence:
            Match confidence in ``[0, 1]``.
        method:
            Matching method that produced the link (e.g. ``"hungarian"``).
        user_override:
            ``True`` if a clinician manually confirmed / corrected the link.

        Raises
        ------
        ValueError
            If either node does not exist, or adding the edge would create a
            cycle.
        """
        if source_key not in self._graph:
            raise ValueError(f"Source node '{source_key}' not in graph")
        if target_key not in self._graph:
            raise ValueError(f"Target node '{target_key}' not in graph")

        # Validate temporal ordering: source must precede target
        src_tp = self._graph.nodes[source_key].get("timepoint_id", "")
        tgt_tp = self._graph.nodes[target_key].get("timepoint_id", "")
        if src_tp and tgt_tp and src_tp > tgt_tp:
            raise ValueError(
                f"Temporal ordering violation: source timepoint '{src_tp}' "
                f"must precede target timepoint '{tgt_tp}'"
            )

        # Tentatively add the edge, then check for cycles.
        self._graph.add_edge(
            source_key,
            target_key,
            confidence=confidence,
            method=method,
            user_override=user_override,
        )
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(source_key, target_key)
            raise ValueError(
                f"Adding edge {source_key} -> {target_key} would create a cycle"
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trajectory(self, lesion_id: str) -> list[dict[str, Any]]:
        """Return all observations of *lesion_id* sorted by timepoint.

        Parameters
        ----------
        lesion_id:
            The logical lesion identifier (not the node key).

        Returns
        -------
        list[dict]
            Observation attribute dicts ordered by ``timepoint_id``.
        """
        observations: list[dict[str, Any]] = []
        for node, attrs in self._graph.nodes(data=True):
            if attrs.get("lesion_id") == lesion_id:
                obs = dict(attrs)
                obs["node_key"] = node
                observations.append(obs)

        observations.sort(key=lambda o: o.get("timepoint_id", ""))
        return observations

    def get_active_lesions(self, timepoint_id: str) -> list[str]:
        """Return lesion IDs present at the given *timepoint_id*.

        Parameters
        ----------
        timepoint_id:
            The timepoint to query.

        Returns
        -------
        list[str]
            Unique lesion identifiers observed at the timepoint.
        """
        lesion_ids: list[str] = []
        for _node, attrs in self._graph.nodes(data=True):
            if attrs.get("timepoint_id") == timepoint_id:
                lid = attrs.get("lesion_id", "")
                if lid and lid not in lesion_ids:
                    lesion_ids.append(lid)
        return lesion_ids

    def get_new_lesions(self, timepoint_id: str) -> list[str]:
        """Return lesion IDs at *timepoint_id* that have no incoming edges.

        A lesion with no predecessor link is considered *new* at the given
        timepoint.

        Parameters
        ----------
        timepoint_id:
            The timepoint to query.

        Returns
        -------
        list[str]
            Lesion IDs that are new at this timepoint.
        """
        new_ids: list[str] = []
        for node, attrs in self._graph.nodes(data=True):
            if attrs.get("timepoint_id") != timepoint_id:
                continue
            if self._graph.in_degree(node) == 0:
                lid = attrs.get("lesion_id", "")
                if lid and lid not in new_ids:
                    new_ids.append(lid)
        return new_ids

    def get_disappeared_lesions(
        self, timepoint_id: str, prev_timepoint_id: str
    ) -> list[str]:
        """Lesions at *prev_timepoint_id* with no outgoing edge to *timepoint_id*.

        Parameters
        ----------
        timepoint_id:
            The current timepoint.
        prev_timepoint_id:
            The previous timepoint.

        Returns
        -------
        list[str]
            Lesion IDs that were present before but have no link to the
            current timepoint.
        """
        # Gather node keys at the current timepoint for fast lookup.
        current_nodes: set[str] = {
            node
            for node, attrs in self._graph.nodes(data=True)
            if attrs.get("timepoint_id") == timepoint_id
        }

        disappeared: list[str] = []
        for node, attrs in self._graph.nodes(data=True):
            if attrs.get("timepoint_id") != prev_timepoint_id:
                continue
            # Check if any successor of this node is in the current timepoint.
            has_link = any(
                succ in current_nodes for succ in self._graph.successors(node)
            )
            if not has_link:
                lid = attrs.get("lesion_id", "")
                if lid and lid not in disappeared:
                    disappeared.append(lid)
        return disappeared

    def get_lineage(self, node_key: str) -> list[str]:
        """Return the full chain of ancestor node keys.

        Traverses incoming edges recursively from *node_key* back to the
        earliest ancestor.

        Parameters
        ----------
        node_key:
            Starting observation node key.

        Returns
        -------
        list[str]
            Ancestor node keys ordered from earliest to most recent
            (excluding *node_key* itself).

        Raises
        ------
        ValueError
            If *node_key* is not in the graph.
        """
        if node_key not in self._graph:
            raise ValueError(f"Node '{node_key}' not in graph")

        ancestors: list[str] = []
        current = node_key
        visited: set[str] = {current}

        while True:
            preds = list(self._graph.predecessors(current))
            if not preds:
                break
            # Follow the highest-confidence predecessor when there are
            # multiple (e.g. merge scenarios).
            best_pred = max(
                preds,
                key=lambda p: self._graph.edges[p, current].get("confidence", 0.0),
            )
            if best_pred in visited:
                break  # Safety guard against unexpected state.
            visited.add(best_pred)
            ancestors.append(best_pred)
            current = best_pred

        ancestors.reverse()
        return ancestors

    def build_tracked_sets(self) -> list[dict[str, Any]]:
        """Build TrackedLesionSet-like dicts for all tracked lesions.

        Groups observations by following identity edges to discover
        which node observations belong to the same logical lesion.

        Returns
        -------
        list[dict]
            Each dict contains: canonical_id, observation_ids,
            timepoint_ids, volumes, diameters, confidences,
            is_target, organ, first_seen, last_seen, status.
        """
        # Find all root nodes (in-degree 0) and trace forward
        visited: set[str] = set()
        tracked_sets: list[dict[str, Any]] = []

        for node in self._graph.nodes:
            if self._graph.in_degree(node) == 0 and node not in visited:
                # Trace the chain forward from this root
                chain: list[str] = [node]
                visited.add(node)
                current = node
                while True:
                    succs = list(self._graph.successors(current))
                    if not succs:
                        break
                    # Follow highest-confidence successor
                    best = max(
                        succs,
                        key=lambda s: self._graph.edges[current, s].get(
                            "confidence", 0.0
                        ),
                    )
                    if best in visited:
                        break
                    visited.add(best)
                    chain.append(best)
                    current = best

                # Build the tracked set from the chain
                attrs_list = [self._graph.nodes[n] for n in chain]
                edge_confs: list[float] = []
                for i in range(len(chain) - 1):
                    edata = self._graph.edges.get((chain[i], chain[i + 1]), {})
                    edge_confs.append(edata.get("confidence", 0.0))

                volumes = [a.get("volume", 0.0) for a in attrs_list]
                diameters = [a.get("diameter", 0.0) for a in attrs_list]
                timepoints = [a.get("timepoint_id", "") for a in attrs_list]
                first_tp = timepoints[0] if timepoints else ""
                last_tp = timepoints[-1] if timepoints else ""

                # Determine status
                last_vol = volumes[-1] if volumes else 0.0
                status = "active" if last_vol > 0 else "resolved"

                canonical_id = attrs_list[0].get("lesion_id", "") if attrs_list else ""

                tracked_sets.append({
                    "canonical_id": canonical_id,
                    "observation_ids": chain,
                    "timepoint_ids": timepoints,
                    "volumes": volumes,
                    "diameters": diameters,
                    "confidences": edge_confs,
                    "is_target": attrs_list[0].get("is_target", False) if attrs_list else False,
                    "organ": attrs_list[0].get("organ", "") if attrs_list else "",
                    "first_seen": first_tp,
                    "last_seen": last_tp,
                    "status": status,
                })

        return tracked_sets

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_dag(self) -> list[str]:
        """Check the graph for structural issues.

        Returns
        -------
        list[str]
            Human-readable descriptions of detected issues.  An empty list
            means the graph is healthy.
        """
        issues: list[str] = []

        # 1. Cycle check
        if not nx.is_directed_acyclic_graph(self._graph):
            cycles = list(nx.simple_cycles(self._graph))
            for cycle in cycles:
                issues.append(f"Cycle detected: {' -> '.join(cycle)}")

        # 2. Orphan nodes (no edges at all and not the only observation
        #    of their lesion).
        lesion_counts: dict[str, int] = defaultdict(int)
        for _node, attrs in self._graph.nodes(data=True):
            lesion_counts[attrs.get("lesion_id", "")] += 1

        for node, attrs in self._graph.nodes(data=True):
            lid = attrs.get("lesion_id", "")
            in_deg = self._graph.in_degree(node)
            out_deg = self._graph.out_degree(node)
            if in_deg == 0 and out_deg == 0 and lesion_counts.get(lid, 0) > 1:
                issues.append(f"Orphan node: {node} (lesion has other observations)")

        # 3. Missing node attributes
        required_attrs = {"lesion_id", "timepoint_id", "centroid"}
        for node, attrs in self._graph.nodes(data=True):
            missing = required_attrs - set(attrs.keys())
            if missing:
                issues.append(
                    f"Node '{node}' missing attributes: {', '.join(sorted(missing))}"
                )

        # 4. Edge confidence range
        for u, v, edata in self._graph.edges(data=True):
            conf = edata.get("confidence", -1.0)
            if not (0.0 <= conf <= 1.0):
                issues.append(
                    f"Edge {u} -> {v} has confidence {conf} outside [0, 1]"
                )

        # 5. Required edge attributes
        required_edge_attrs = {"confidence", "method"}
        for u, v, edata in self._graph.edges(data=True):
            missing_edge = required_edge_attrs - set(edata.keys())
            if missing_edge:
                issues.append(
                    f"Edge {u} -> {v} missing attributes: {', '.join(sorted(missing_edge))}"
                )

        return issues

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the graph to a JSON string (node-link format).

        Returns
        -------
        str
            JSON representation of the graph.
        """
        data = json_graph.node_link_data(self._graph)
        return json.dumps(data, default=_json_default)

    @classmethod
    def from_json(cls, data: str) -> LesionGraph:
        """Deserialize a graph from a JSON string.

        Parameters
        ----------
        data:
            JSON string produced by :meth:`to_json`.

        Returns
        -------
        LesionGraph
            Reconstructed identity graph.
        """
        parsed = json.loads(data)
        instance = cls()
        instance._graph = json_graph.node_link_graph(parsed, directed=True)
        return instance

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Return aggregate counts describing the graph.

        Returns
        -------
        dict[str, int]
            Keys: ``total_nodes``, ``total_edges``, ``tracked_lesions``,
            ``new_lesions`` (nodes with in-degree 0).
        """
        tracked: set[str] = set()
        new_count = 0
        for node, attrs in self._graph.nodes(data=True):
            lid = attrs.get("lesion_id", "")
            if lid:
                tracked.add(lid)
            if self._graph.in_degree(node) == 0:
                new_count += 1

        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "tracked_lesions": len(tracked),
            "new_lesions": new_count,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def export_for_ui(self) -> dict[str, Any]:
        """Export graph data in the format expected by UI visualization.

        Returns
        -------
        dict
            Keys: nodes (list[dict]), edges (list[dict]).
        """
        nodes: list[dict[str, Any]] = []
        for node, attrs in self._graph.nodes(data=True):
            nodes.append({
                "id": node,
                "lesion_id": attrs.get("lesion_id", ""),
                "timepoint_id": attrs.get("timepoint_id", ""),
                "volume": attrs.get("volume", 0.0),
                "diameter": attrs.get("diameter", 0.0),
                "confidence": attrs.get("confidence", 0.0),
                "organ": attrs.get("organ", ""),
                "is_target": attrs.get("is_target", False),
                "is_new": attrs.get("is_new", False),
            })

        edges: list[dict[str, Any]] = []
        for u, v, edata in self._graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "confidence": edata.get("confidence", 0.0),
                "method": edata.get("method", ""),
                "user_override": edata.get("user_override", False),
            })

        return {"nodes": nodes, "edges": edges}

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX DiGraph (read-only intent)."""
        return self._graph

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"LesionGraph(nodes={s['total_nodes']}, edges={s['total_edges']}, "
            f"tracked={s['tracked_lesions']})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serializer for non-standard types in node attributes."""
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
