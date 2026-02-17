"""Lesion matching across timepoints using a three-stage pipeline.

Implements ADR-010: spatial proximity (Hungarian algorithm), appearance
similarity (volume ratio), and confidence scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from digital_twin_tumor.domain.models import Lesion
from digital_twin_tumor.tracking.registration import transform_points


class LesionMatcher:
    """Match lesions between consecutive timepoints.

    The three-stage pipeline:
      1. **Spatial proximity** -- Hungarian algorithm on Euclidean distances
         with a configurable threshold.
      2. **Appearance similarity** -- volume-ratio tie-breaking for ambiguous
         matches.
      3. **Confidence scoring** -- composite score from distance and volume
         similarity.
    """

    # ------------------------------------------------------------------
    # Primary matching
    # ------------------------------------------------------------------

    @staticmethod
    def match_lesions(
        current_lesions: list[Lesion],
        previous_lesions: list[Lesion],
        transform: np.ndarray | None = None,
        distance_threshold: float = 20.0,
    ) -> list[tuple[str, str, float]]:
        """Run the full three-stage matching pipeline.

        Parameters
        ----------
        current_lesions:
            Lesions detected at the current timepoint.
        previous_lesions:
            Lesions detected at the previous timepoint.
        transform:
            Optional 4x4 rigid-body transform to apply to
            *previous_lesions* centroids before distance computation.
        distance_threshold:
            Maximum Euclidean distance (mm) to accept a match.

        Returns
        -------
        list[tuple[str, str, float]]
            Triples of ``(current_lesion_id, previous_lesion_id, confidence)``.
        """
        if not current_lesions or not previous_lesions:
            return []

        # Extract centroid arrays
        cur_centroids = np.array(
            [l.centroid for l in current_lesions], dtype=np.float64
        )
        prev_centroids = np.array(
            [l.centroid for l in previous_lesions], dtype=np.float64
        )

        # Apply spatial transform to previous centroids when available
        if transform is not None:
            prev_centroids = transform_points(prev_centroids, transform)

        # ------------------------------------------------------------------
        # Stage 1: Spatial proximity -- Hungarian algorithm
        # ------------------------------------------------------------------
        dist_matrix = cdist(cur_centroids, prev_centroids, metric="euclidean")

        # Build a cost matrix that penalises infeasible assignments heavily
        large_cost = distance_threshold * 10.0
        cost_matrix = dist_matrix.copy()
        cost_matrix[cost_matrix > distance_threshold] = large_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Keep only feasible assignments
        raw_matches: list[tuple[int, int, float]] = []
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= distance_threshold:
                raw_matches.append((r, c, float(dist_matrix[r, c])))

        # ------------------------------------------------------------------
        # Stage 2: Appearance similarity -- volume ratio tie-breaking
        # ------------------------------------------------------------------
        # For each current lesion that could match multiple previous ones
        # within the threshold, pick the one with the best volume ratio
        # (closest to 1.0).
        refined_matches = _refine_by_volume(
            raw_matches, current_lesions, previous_lesions, dist_matrix,
            distance_threshold,
        )

        # ------------------------------------------------------------------
        # Stage 3: Confidence scoring
        # ------------------------------------------------------------------
        results: list[tuple[str, str, float]] = []
        for cur_idx, prev_idx, distance in refined_matches:
            confidence = _compute_confidence(
                distance,
                distance_threshold,
                current_lesions[cur_idx].volume_mm3,
                previous_lesions[prev_idx].volume_mm3,
            )
            results.append((
                current_lesions[cur_idx].lesion_id,
                previous_lesions[prev_idx].lesion_id,
                confidence,
            ))

        return results

    # ------------------------------------------------------------------
    # New / disappeared detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_new_lesions(
        current_lesions: list[Lesion],
        matched_ids: set[str],
    ) -> list[Lesion]:
        """Return current lesions that were not matched to any previous lesion.

        Parameters
        ----------
        current_lesions:
            All lesions at the current timepoint.
        matched_ids:
            Set of ``lesion_id`` values from current that were matched.

        Returns
        -------
        list[Lesion]
            Unmatched (new) lesions.
        """
        return [l for l in current_lesions if l.lesion_id not in matched_ids]

    @staticmethod
    def detect_disappeared(
        previous_lesions: list[Lesion],
        matched_ids: set[str],
    ) -> list[Lesion]:
        """Return previous lesions that were not matched to any current lesion.

        Parameters
        ----------
        previous_lesions:
            All lesions at the previous timepoint.
        matched_ids:
            Set of ``lesion_id`` values from previous that were matched.

        Returns
        -------
        list[Lesion]
            Unmatched (disappeared) lesions.
        """
        return [l for l in previous_lesions if l.lesion_id not in matched_ids]

    # ------------------------------------------------------------------
    # Split / merge detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_splits(
        current_lesions: list[Lesion],
        previous_lesions: list[Lesion],
        matches: list[tuple[str, str, float]],
    ) -> list[dict[str, Any]]:
        """Detect when one previous lesion maps to multiple current lesions.

        A split is flagged when volume is approximately conserved (combined
        current volumes are within 50% of the previous volume).

        Parameters
        ----------
        current_lesions:
            All current timepoint lesions.
        previous_lesions:
            All previous timepoint lesions.
        matches:
            Output of :meth:`match_lesions`.

        Returns
        -------
        list[dict]
            Each dict contains ``previous_id``, ``current_ids``,
            ``previous_volume``, ``combined_current_volume``, and
            ``volume_ratio``.
        """
        if not matches:
            return []

        prev_map: dict[str, Lesion] = {l.lesion_id: l for l in previous_lesions}
        cur_map: dict[str, Lesion] = {l.lesion_id: l for l in current_lesions}

        # Group current IDs by their matched previous ID
        prev_to_cur: dict[str, list[str]] = {}
        for cur_id, prev_id, _conf in matches:
            prev_to_cur.setdefault(prev_id, []).append(cur_id)

        splits: list[dict[str, Any]] = []
        for prev_id, cur_ids in prev_to_cur.items():
            if len(cur_ids) < 2:
                continue
            prev_vol = prev_map[prev_id].volume_mm3 if prev_id in prev_map else 0.0
            combined_vol = sum(
                cur_map[cid].volume_mm3 for cid in cur_ids if cid in cur_map
            )
            ratio = combined_vol / prev_vol if prev_vol > 0 else 0.0

            # Flag as split if combined volume is within 50% of original
            if 0.5 <= ratio <= 1.5:
                splits.append({
                    "previous_id": prev_id,
                    "current_ids": cur_ids,
                    "previous_volume": prev_vol,
                    "combined_current_volume": combined_vol,
                    "volume_ratio": round(ratio, 4),
                })

        return splits

    @staticmethod
    def detect_merges(
        current_lesions: list[Lesion],
        previous_lesions: list[Lesion],
        matches: list[tuple[str, str, float]],
    ) -> list[dict[str, Any]]:
        """Detect when multiple previous lesions map to one current lesion.

        A merge is flagged when volume is approximately conserved (current
        volume is within 50% of combined previous volumes).

        Parameters
        ----------
        current_lesions:
            All current timepoint lesions.
        previous_lesions:
            All previous timepoint lesions.
        matches:
            Output of :meth:`match_lesions`.

        Returns
        -------
        list[dict]
            Each dict contains ``current_id``, ``previous_ids``,
            ``current_volume``, ``combined_previous_volume``, and
            ``volume_ratio``.
        """
        if not matches:
            return []

        prev_map: dict[str, Lesion] = {l.lesion_id: l for l in previous_lesions}
        cur_map: dict[str, Lesion] = {l.lesion_id: l for l in current_lesions}

        # Group previous IDs by their matched current ID
        cur_to_prev: dict[str, list[str]] = {}
        for cur_id, prev_id, _conf in matches:
            cur_to_prev.setdefault(cur_id, []).append(prev_id)

        merges: list[dict[str, Any]] = []
        for cur_id, prev_ids in cur_to_prev.items():
            if len(prev_ids) < 2:
                continue
            cur_vol = cur_map[cur_id].volume_mm3 if cur_id in cur_map else 0.0
            combined_prev = sum(
                prev_map[pid].volume_mm3 for pid in prev_ids if pid in prev_map
            )
            ratio = cur_vol / combined_prev if combined_prev > 0 else 0.0

            if 0.5 <= ratio <= 1.5:
                merges.append({
                    "current_id": cur_id,
                    "previous_ids": prev_ids,
                    "current_volume": cur_vol,
                    "combined_previous_volume": combined_prev,
                    "volume_ratio": round(ratio, 4),
                })

        return merges

    # ------------------------------------------------------------------
    # Convenience: match + graph update
    # ------------------------------------------------------------------

    @staticmethod
    def match_with_graph_update(
        graph: Any,
        current_lesions: list[Lesion],
        previous_lesions: list[Lesion],
        current_timepoint: Any,
        previous_timepoint: Any,
        transform: np.ndarray | None = None,
        distance_threshold: float = 20.0,
    ) -> dict[str, Any]:
        """Match lesions and update the identity graph in one step.

        Parameters
        ----------
        graph:
            A LesionGraph instance.
        current_lesions:
            Lesions at the current timepoint.
        previous_lesions:
            Lesions at the previous timepoint.
        current_timepoint:
            TimePoint for current observations.
        previous_timepoint:
            TimePoint for previous observations.
        transform:
            Optional spatial transform.
        distance_threshold:
            Maximum matching distance in mm.

        Returns
        -------
        dict
            Keys: matches, new_lesions, disappeared, splits, merges.
        """
        # Ensure previous observations are in graph
        for les in previous_lesions:
            key = f"{les.lesion_id}@{previous_timepoint.timepoint_id}"
            if key not in graph.graph:
                graph.add_observation(les, previous_timepoint)

        # Add current observations
        for les in current_lesions:
            graph.add_observation(les, current_timepoint)

        # Run matching
        matches = LesionMatcher.match_lesions(
            current_lesions, previous_lesions, transform, distance_threshold,
        )

        # Add identity links
        for cur_id, prev_id, conf in matches:
            src = f"{prev_id}@{previous_timepoint.timepoint_id}"
            tgt = f"{cur_id}@{current_timepoint.timepoint_id}"
            try:
                graph.add_identity_link(src, tgt, conf, "hungarian")
            except ValueError:
                pass  # Skip if would create cycle

        matched_cur = {m[0] for m in matches}
        matched_prev = {m[1] for m in matches}

        return {
            "matches": matches,
            "new_lesions": LesionMatcher.detect_new_lesions(current_lesions, matched_cur),
            "disappeared": LesionMatcher.detect_disappeared(previous_lesions, matched_prev),
            "splits": LesionMatcher.detect_splits(current_lesions, previous_lesions, matches),
            "merges": LesionMatcher.detect_merges(current_lesions, previous_lesions, matches),
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _refine_by_volume(
    raw_matches: list[tuple[int, int, float]],
    current_lesions: list[Lesion],
    previous_lesions: list[Lesion],
    dist_matrix: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Refine raw matches by volume-ratio when spatial proximity is ambiguous.

    For each matched current lesion, if there are other previous lesions
    also within the distance threshold and closer in volume ratio, swap
    the assignment.

    Returns the refined list of ``(cur_idx, prev_idx, distance)`` tuples.
    """
    if not raw_matches:
        return []

    # Build sets of already-assigned indices
    matched_cur: set[int] = set()
    matched_prev: set[int] = set()
    result_map: dict[int, tuple[int, float]] = {}

    for cur_idx, prev_idx, dist in raw_matches:
        result_map[cur_idx] = (prev_idx, dist)
        matched_cur.add(cur_idx)
        matched_prev.add(prev_idx)

    # Check for ambiguous cases and apply volume-ratio tie-breaking
    for cur_idx in list(matched_cur):
        assigned_prev, assigned_dist = result_map[cur_idx]
        cur_vol = current_lesions[cur_idx].volume_mm3

        # Find all previous candidates within threshold
        candidates: list[tuple[int, float, float]] = []
        for prev_idx in range(len(previous_lesions)):
            d = float(dist_matrix[cur_idx, prev_idx])
            if d <= threshold:
                prev_vol = previous_lesions[prev_idx].volume_mm3
                vol_ratio = _volume_ratio(cur_vol, prev_vol)
                candidates.append((prev_idx, d, vol_ratio))

        if len(candidates) <= 1:
            continue

        # Sort by combined score: distance weight + volume ratio weight
        # Lower is better for both
        best = min(
            candidates,
            key=lambda c: (c[1] / threshold) * 0.6 + (1.0 - c[2]) * 0.4,
        )

        best_prev_idx = best[0]
        if best_prev_idx != assigned_prev:
            # Only swap if the new candidate is not already assigned to
            # something with a better overall score
            if best_prev_idx not in matched_prev:
                result_map[cur_idx] = (best_prev_idx, best[1])
                matched_prev.discard(assigned_prev)
                matched_prev.add(best_prev_idx)

    return [
        (cur_idx, prev_idx, dist)
        for cur_idx, (prev_idx, dist) in result_map.items()
    ]


def _volume_ratio(vol_a: float, vol_b: float) -> float:
    """Return a similarity score in [0, 1] based on the ratio of volumes.

    A ratio of 1.0 means the volumes are identical; 0.0 means maximally
    dissimilar (one volume is zero or infinitely larger).
    """
    if vol_a <= 0.0 or vol_b <= 0.0:
        return 0.0
    ratio = min(vol_a, vol_b) / max(vol_a, vol_b)
    return float(ratio)


def _compute_confidence(
    distance: float,
    threshold: float,
    cur_volume: float,
    prev_volume: float,
) -> float:
    """Composite confidence from spatial distance and volume similarity.

    The score is a weighted combination:
      - 60% spatial:  ``1 - distance / threshold``
      - 40% volume:   ``min(a,b) / max(a,b)``

    Clamped to ``[0, 1]``.

    Parameters
    ----------
    distance:
        Euclidean distance between centroids in mm.
    threshold:
        Maximum acceptable distance.
    cur_volume:
        Current lesion volume in mm^3.
    prev_volume:
        Previous lesion volume in mm^3.

    Returns
    -------
    float
        Confidence in ``[0, 1]``.
    """
    spatial_score = max(0.0, 1.0 - distance / threshold) if threshold > 0 else 0.0
    vol_score = _volume_ratio(cur_volume, prev_volume)
    confidence = 0.6 * spatial_score + 0.4 * vol_score
    return round(min(1.0, max(0.0, confidence)), 4)
