"""Tests for lesion tracking: identity graph, matching, and registration."""

from __future__ import annotations

import json

import numpy as np
import numpy.testing as npt
import pytest

from digital_twin_tumor.domain.models import Lesion, TimePoint
from digital_twin_tumor.tracking.identity_graph import LesionGraph
from digital_twin_tumor.tracking.matching import LesionMatcher
from digital_twin_tumor.tracking.registration import (
    register_volumes,
    transform_points,
)


# =====================================================================
# LesionGraph
# =====================================================================


class TestLesionGraph:
    """Test DAG-based lesion identity graph."""

    def _make_graph_with_two_nodes(self) -> tuple[LesionGraph, str, str]:
        graph = LesionGraph()
        les1 = Lesion(lesion_id="L1", centroid=(10.0, 10.0, 10.0), volume_mm3=100.0)
        les2 = Lesion(lesion_id="L1", centroid=(11.0, 11.0, 11.0), volume_mm3=120.0)
        tp1 = TimePoint(timepoint_id="TP1")
        tp2 = TimePoint(timepoint_id="TP2")
        key1 = graph.add_observation(les1, tp1)
        key2 = graph.add_observation(les2, tp2)
        return graph, key1, key2

    def test_add_observation(self):
        graph = LesionGraph()
        les = Lesion(lesion_id="L1")
        tp = TimePoint(timepoint_id="TP1")
        key = graph.add_observation(les, tp)
        assert key == "L1@TP1"
        summary = graph.summary()
        assert summary["total_nodes"] == 1

    def test_add_identity_link(self):
        graph, key1, key2 = self._make_graph_with_two_nodes()
        graph.add_identity_link(key1, key2, confidence=0.9, method="hungarian")
        summary = graph.summary()
        assert summary["total_edges"] == 1

    def test_cycle_prevention(self):
        graph, key1, key2 = self._make_graph_with_two_nodes()
        graph.add_identity_link(key1, key2, confidence=0.9, method="test")
        with pytest.raises(ValueError, match="(cycle|Temporal ordering)"):
            graph.add_identity_link(key2, key1, confidence=0.9, method="test")

    def test_get_trajectory(self):
        graph, key1, key2 = self._make_graph_with_two_nodes()
        graph.add_identity_link(key1, key2, confidence=0.9, method="test")
        trajectory = graph.get_trajectory("L1")
        assert len(trajectory) == 2
        # Should be sorted by timepoint_id
        assert trajectory[0]["timepoint_id"] == "TP1"
        assert trajectory[1]["timepoint_id"] == "TP2"

    def test_get_active_lesions(self):
        graph, key1, key2 = self._make_graph_with_two_nodes()
        active = graph.get_active_lesions("TP1")
        assert "L1" in active

    def test_get_new_lesions(self):
        graph, key1, key2 = self._make_graph_with_two_nodes()
        # Both have in_degree 0, so both are "new"
        new_tp1 = graph.get_new_lesions("TP1")
        assert "L1" in new_tp1

        # After linking, TP2's node has in_degree 1
        graph.add_identity_link(key1, key2, confidence=0.9, method="test")
        new_tp2 = graph.get_new_lesions("TP2")
        assert "L1" not in new_tp2

    def test_link_nonexistent_node_raises(self):
        graph = LesionGraph()
        with pytest.raises(ValueError, match="not in graph"):
            graph.add_identity_link("A@X", "B@Y", confidence=0.5, method="test")

    def test_serialize_roundtrip(self):
        graph, key1, key2 = self._make_graph_with_two_nodes()
        graph.add_identity_link(key1, key2, confidence=0.85, method="hungarian")

        json_str = graph.to_json()
        assert isinstance(json_str, str)

        restored = LesionGraph.from_json(json_str)
        assert restored.summary()["total_nodes"] == 2
        assert restored.summary()["total_edges"] == 1

    def test_validate_dag_clean(self):
        graph, key1, key2 = self._make_graph_with_two_nodes()
        graph.add_identity_link(key1, key2, confidence=0.9, method="test")
        issues = graph.validate_dag()
        assert len(issues) == 0

    def test_get_lineage(self):
        graph = LesionGraph()
        tp_a = TimePoint(timepoint_id="A")
        tp_b = TimePoint(timepoint_id="B")
        tp_c = TimePoint(timepoint_id="C")
        les = Lesion(lesion_id="L1")
        ka = graph.add_observation(les, tp_a)
        kb = graph.add_observation(les, tp_b)
        kc = graph.add_observation(les, tp_c)
        graph.add_identity_link(ka, kb, confidence=0.9, method="test")
        graph.add_identity_link(kb, kc, confidence=0.85, method="test")
        lineage = graph.get_lineage(kc)
        assert lineage == [ka, kb]


# =====================================================================
# LesionMatcher
# =====================================================================


class TestLesionMatcher:
    """Test lesion matching across timepoints."""

    def test_exact_match(self):
        """Two lesions at the same centroid should match with high confidence."""
        prev = [Lesion(lesion_id="P1", centroid=(10.0, 10.0, 10.0), volume_mm3=100.0)]
        curr = [Lesion(lesion_id="C1", centroid=(10.0, 10.0, 10.0), volume_mm3=100.0)]
        matches = LesionMatcher.match_lesions(curr, prev, distance_threshold=20.0)
        assert len(matches) == 1
        assert matches[0][0] == "C1"
        assert matches[0][1] == "P1"
        assert matches[0][2] > 0.5  # High confidence

    def test_close_match(self):
        """Two lesions 5mm apart should still match within 20mm threshold."""
        prev = [Lesion(lesion_id="P1", centroid=(10.0, 10.0, 10.0), volume_mm3=100.0)]
        curr = [Lesion(lesion_id="C1", centroid=(15.0, 10.0, 10.0), volume_mm3=100.0)]
        matches = LesionMatcher.match_lesions(curr, prev, distance_threshold=20.0)
        assert len(matches) == 1
        assert matches[0][0] == "C1"
        assert matches[0][1] == "P1"

    def test_no_match_beyond_threshold(self):
        """Two lesions 50mm apart should not match with 20mm threshold."""
        prev = [Lesion(lesion_id="P1", centroid=(10.0, 10.0, 10.0), volume_mm3=100.0)]
        curr = [Lesion(lesion_id="C1", centroid=(60.0, 10.0, 10.0), volume_mm3=100.0)]
        matches = LesionMatcher.match_lesions(curr, prev, distance_threshold=20.0)
        assert len(matches) == 0

    def test_multiple_lesion_matching(self):
        """Match multiple lesions correctly."""
        prev = [
            Lesion(lesion_id="P1", centroid=(10.0, 10.0, 10.0), volume_mm3=100.0),
            Lesion(lesion_id="P2", centroid=(50.0, 50.0, 50.0), volume_mm3=200.0),
        ]
        curr = [
            Lesion(lesion_id="C1", centroid=(11.0, 10.0, 10.0), volume_mm3=110.0),
            Lesion(lesion_id="C2", centroid=(51.0, 50.0, 50.0), volume_mm3=210.0),
        ]
        matches = LesionMatcher.match_lesions(curr, prev, distance_threshold=20.0)
        assert len(matches) == 2
        matched_pairs = {(m[0], m[1]) for m in matches}
        assert ("C1", "P1") in matched_pairs
        assert ("C2", "P2") in matched_pairs

    def test_empty_inputs(self):
        assert LesionMatcher.match_lesions([], [], distance_threshold=20.0) == []
        assert LesionMatcher.match_lesions(
            [Lesion(lesion_id="L1", centroid=(0, 0, 0))], [], distance_threshold=20.0
        ) == []

    def test_detect_new_lesions(self):
        current = [
            Lesion(lesion_id="C1"),
            Lesion(lesion_id="C2"),
        ]
        matched_ids = {"C1"}
        new = LesionMatcher.detect_new_lesions(current, matched_ids)
        assert len(new) == 1
        assert new[0].lesion_id == "C2"

    def test_detect_disappeared(self):
        previous = [
            Lesion(lesion_id="P1"),
            Lesion(lesion_id="P2"),
        ]
        matched_ids = {"P1"}
        disappeared = LesionMatcher.detect_disappeared(previous, matched_ids)
        assert len(disappeared) == 1
        assert disappeared[0].lesion_id == "P2"


class TestSplitMergeDetection:
    """Test split and merge detection in lesion matching."""

    def test_detect_splits(self):
        """One previous lesion matching two current lesions = split."""
        prev = [Lesion(lesion_id="P1", volume_mm3=1000.0)]
        curr = [
            Lesion(lesion_id="C1", volume_mm3=500.0),
            Lesion(lesion_id="C2", volume_mm3=500.0),
        ]
        # Simulate matches where both C1 and C2 matched to P1
        matches = [("C1", "P1", 0.8), ("C2", "P1", 0.7)]
        splits = LesionMatcher.detect_splits(curr, prev, matches)
        assert len(splits) == 1
        assert splits[0]["previous_id"] == "P1"
        assert set(splits[0]["current_ids"]) == {"C1", "C2"}

    def test_detect_merges(self):
        """Two previous lesions matching one current lesion = merge."""
        prev = [
            Lesion(lesion_id="P1", volume_mm3=500.0),
            Lesion(lesion_id="P2", volume_mm3=500.0),
        ]
        curr = [Lesion(lesion_id="C1", volume_mm3=1000.0)]
        matches = [("C1", "P1", 0.8), ("C1", "P2", 0.7)]
        merges = LesionMatcher.detect_merges(curr, prev, matches)
        assert len(merges) == 1
        assert merges[0]["current_id"] == "C1"
        assert set(merges[0]["previous_ids"]) == {"P1", "P2"}

    def test_no_splits_single_matches(self):
        matches = [("C1", "P1", 0.9)]
        splits = LesionMatcher.detect_splits([], [], matches)
        assert len(splits) == 0


# =====================================================================
# Registration
# =====================================================================


class TestRegistration:
    """Test image registration and point transformation."""

    def test_register_volumes_returns_4x4(self):
        rng = np.random.default_rng(42)
        fixed = rng.random((16, 16, 16)).astype(np.float64)
        moving = rng.random((16, 16, 16)).astype(np.float64)
        mat = register_volumes(fixed, moving)
        assert mat.shape == (4, 4)
        assert mat.dtype == np.float64

    def test_transform_points_roundtrip(self):
        """Applying identity matrix should return original points."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        identity = np.eye(4, dtype=np.float64)
        result = transform_points(points, identity)
        npt.assert_allclose(result, points, atol=1e-10)

    def test_transform_points_translation(self):
        """Test pure translation."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        mat = np.eye(4, dtype=np.float64)
        mat[:3, 3] = [10.0, 20.0, 30.0]
        result = transform_points(points, mat)
        expected = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
        npt.assert_allclose(result, expected, atol=1e-10)

    def test_transform_empty_points(self):
        points = np.empty((0, 3), dtype=np.float64)
        result = transform_points(points, np.eye(4))
        assert result.shape == (0, 3)

    def test_register_empty_volumes(self):
        fixed = np.empty((0, 0, 0), dtype=np.float64)
        moving = np.empty((0, 0, 0), dtype=np.float64)
        mat = register_volumes(fixed, moving)
        npt.assert_array_equal(mat, np.eye(4))
