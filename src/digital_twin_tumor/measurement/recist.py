"""RECIST 1.1 and iRECIST classification engine.

Implements ADR-009: target selection, sum-of-diameters computation,
response classification, and the iRECIST immunotherapy extension.
"""

from __future__ import annotations

import logging
from typing import Sequence

from digital_twin_tumor.domain.models import Lesion, Measurement, RECISTResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RECIST 1.1 thresholds
_PR_THRESHOLD: float = -30.0   # >= 30% decrease from baseline
_PD_THRESHOLD: float = 20.0    # >= 20% increase from nadir
_PD_ABSOLUTE_MM: float = 5.0   # minimum 5 mm absolute increase

# Nodal lesion thresholds
_NODAL_TARGET_MIN_SA_MM: float = 15.0   # short axis >= 15 mm to be target
_NON_NODAL_TARGET_MIN_LD_MM: float = 10.0  # longest diameter >= 10 mm
_NODAL_CR_THRESHOLD_MM: float = 10.0  # nodal < 10 mm SA = normal

# RECIST categories
CR = "CR"
PR = "PR"
SD = "SD"
PD = "PD"
NE = "NE"

# iRECIST categories
ICR = "iCR"
IPR = "iPR"
ISD = "iSD"
IUPD = "iUPD"
ICPD = "iCPD"


# ---------------------------------------------------------------------------
# Helper: determine if a lesion is nodal
# ---------------------------------------------------------------------------

def _is_nodal(lesion: Lesion) -> bool:
    """Heuristic: a lesion is nodal if its organ field contains 'node' or 'lymph'."""
    organ_lower = lesion.organ.lower()
    return "node" in organ_lower or "lymph" in organ_lower


# ---------------------------------------------------------------------------
# RECISTClassifier
# ---------------------------------------------------------------------------

class RECISTClassifier:
    """RECIST 1.1 and iRECIST response classification.

    Provides methods for target lesion selection, sum-of-diameters
    computation, and timepoint-level response classification per
    the RECIST 1.1 guidelines with iRECIST immunotherapy extension.
    """

    # -- Target selection --------------------------------------------------

    @staticmethod
    def select_targets(
        lesions: list[Lesion],
        max_total: int = 5,
        max_per_organ: int = 2,
    ) -> list[Lesion]:
        """Select target lesions per RECIST 1.1 rules.

        Selection criteria:
            - Non-nodal: longest diameter >= 10 mm.
            - Nodal: short axis >= 15 mm.
            - Prefer the largest measurable lesions.
            - At most *max_per_organ* per organ, *max_total* overall.

        Parameters
        ----------
        lesions:
            All candidate lesions identified at baseline.
        max_total:
            Maximum number of target lesions (RECIST default: 5).
        max_per_organ:
            Maximum targets per organ (RECIST default: 2).

        Returns
        -------
        list[Lesion]
            Selected target lesions, sorted by descending size.
        """
        if not lesions:
            return []

        # Filter measurable lesions
        eligible: list[Lesion] = []
        for lesion in lesions:
            if _is_nodal(lesion):
                if lesion.short_axis_mm >= _NODAL_TARGET_MIN_SA_MM:
                    eligible.append(lesion)
            else:
                if lesion.longest_diameter_mm >= _NON_NODAL_TARGET_MIN_LD_MM:
                    eligible.append(lesion)

        # Sort by primary measurement descending (short axis for nodal,
        # longest diameter for non-nodal)
        def _sort_key(les: Lesion) -> float:
            if _is_nodal(les):
                return les.short_axis_mm
            return les.longest_diameter_mm

        eligible.sort(key=_sort_key, reverse=True)

        # Apply organ limits and total cap
        selected: list[Lesion] = []
        organ_counts: dict[str, int] = {}

        for lesion in eligible:
            if len(selected) >= max_total:
                break
            organ = lesion.organ or "__unknown__"
            count = organ_counts.get(organ, 0)
            if count >= max_per_organ:
                continue
            selected.append(lesion)
            organ_counts[organ] = count + 1

        logger.info(
            "Selected %d target lesions from %d candidates.",
            len(selected),
            len(lesions),
        )
        return selected

    # -- Sum of diameters --------------------------------------------------

    @staticmethod
    def compute_sum_of_diameters(
        measurements: list[Measurement],
        lesions: list[Lesion] | None = None,
    ) -> float:
        """Compute the sum of diameters for target lesions.

        Per RECIST 1.1:
            - Non-nodal targets: sum of longest diameters.
            - Nodal targets: sum of short-axis diameters.

        When *lesions* is provided each measurement is matched by
        ``lesion_id`` to determine nodal status.  Otherwise all
        measurements are assumed non-nodal and ``diameter_mm`` is used.

        Parameters
        ----------
        measurements:
            Measurements for all target lesions at a single time-point.
        lesions:
            Optional list of target lesions for nodal classification.

        Returns
        -------
        float
            Sum of diameters in mm.
        """
        if not measurements:
            return 0.0

        lesion_map: dict[str, Lesion] = {}
        if lesions:
            lesion_map = {les.lesion_id: les for les in lesions}

        total = 0.0
        for m in measurements:
            if m.lesion_id in lesion_map and _is_nodal(lesion_map[m.lesion_id]):
                # Use short axis from metadata if available, else diameter
                short_axis = m.metadata.get("short_axis_mm", m.diameter_mm)
                total += float(short_axis)
            else:
                total += m.diameter_mm

        return total

    # -- Response classification -------------------------------------------

    @staticmethod
    def classify_response(
        current_sum: float,
        baseline_sum: float,
        nadir_sum: float,
        new_lesions: bool = False,
        nodal_short_axes: list[float] | None = None,
    ) -> RECISTResponse:
        """Classify target-lesion response per RECIST 1.1.

        Parameters
        ----------
        current_sum:
            Sum of diameters at the current time-point (mm).
        baseline_sum:
            Sum of diameters at baseline (mm).
        nadir_sum:
            Smallest sum of diameters observed so far (mm).
        new_lesions:
            Whether any unequivocal new lesions have appeared.
        nodal_short_axes:
            Short-axis measurements of all nodal target lesions (mm).
            Required to determine CR when nodal targets are present.

        Returns
        -------
        RECISTResponse
            Classification with all supporting measurements.
        """
        if baseline_sum <= 0 and current_sum <= 0:
            # No measurable disease at any time
            return RECISTResponse(
                sum_of_diameters=current_sum,
                baseline_sum=baseline_sum,
                nadir_sum=nadir_sum,
                percent_change_from_baseline=0.0,
                percent_change_from_nadir=0.0,
                category=CR if not new_lesions else PD,
            )

        # Percent changes
        pct_from_baseline = (
            ((current_sum - baseline_sum) / baseline_sum * 100.0)
            if baseline_sum > 0
            else 0.0
        )
        pct_from_nadir = (
            ((current_sum - nadir_sum) / nadir_sum * 100.0)
            if nadir_sum > 0
            else 0.0
        )

        # --- PD check (highest priority after new lesions) ---
        if new_lesions:
            category = PD
        elif (
            pct_from_nadir >= _PD_THRESHOLD
            and (current_sum - nadir_sum) >= _PD_ABSOLUTE_MM
        ):
            category = PD
        # --- CR check ---
        elif current_sum == 0.0:
            # Non-nodal: all disappeared.  Nodal: check short axes.
            if nodal_short_axes and any(sa >= _NODAL_CR_THRESHOLD_MM for sa in nodal_short_axes):
                # At least one nodal target still >= 10 mm → not CR
                category = PR if pct_from_baseline <= _PR_THRESHOLD else SD
            else:
                category = CR
        # --- PR check ---
        elif pct_from_baseline <= _PR_THRESHOLD:
            category = PR
        # --- SD (default) ---
        else:
            category = SD

        return RECISTResponse(
            sum_of_diameters=current_sum,
            baseline_sum=baseline_sum,
            nadir_sum=nadir_sum,
            percent_change_from_baseline=round(pct_from_baseline, 2),
            percent_change_from_nadir=round(pct_from_nadir, 2),
            category=category,
        )

    # -- Overall response --------------------------------------------------

    @staticmethod
    def classify_overall(
        target_response: str,
        non_target_response: str,
        new_lesions: bool,
    ) -> str:
        """Combine target, non-target, and new-lesion status.

        Implements the RECIST 1.1 overall response decision table.

        Parameters
        ----------
        target_response:
            Target lesion category (``CR``, ``PR``, ``SD``, ``PD``,
            ``NE``).
        non_target_response:
            Non-target lesion category (``CR``, ``non-CR/non-PD``,
            ``PD``, ``NE``).
        new_lesions:
            Whether unequivocal new lesions are present.

        Returns
        -------
        str
            Overall RECIST category.
        """
        # New lesions always → PD
        if new_lesions:
            return PD

        # PD in either target or non-target → PD
        if target_response == PD or non_target_response == PD:
            return PD

        # NE in targets → NE
        if target_response == NE:
            return NE

        # CR in targets
        if target_response == CR:
            if non_target_response == CR:
                return CR
            if non_target_response in ("non-CR/non-PD", NE):
                return PR
            # non-target NE with target CR → PR
            return PR

        # PR in targets
        if target_response == PR:
            if non_target_response != PD:
                return PR
            return PD

        # SD in targets
        if target_response == SD:
            if non_target_response != PD:
                return SD
            return PD

        # Fallback
        logger.warning(
            "Unhandled RECIST combination: target=%s, non_target=%s",
            target_response,
            non_target_response,
        )
        return NE

    # -- Non-target classification -----------------------------------------

    @staticmethod
    def classify_non_target(
        lesions: list[Lesion],
        previous_lesions: list[Lesion] | None = None,
        new_lesions_present: bool = False,
    ) -> str:
        """Classify non-target lesion response per RECIST 1.1.

        Non-target response categories:
            - CR: All non-target lesions disappeared, all lymph nodes < 10mm.
            - non-CR/non-PD: Persistence of one or more non-target lesions.
            - PD: Unequivocal progression of existing non-target lesions.

        Parameters
        ----------
        lesions:
            Non-target lesions at the current time-point.
        previous_lesions:
            Non-target lesions at the previous time-point (for progression check).
        new_lesions_present:
            Whether any new lesions have been detected.

        Returns
        -------
        str
            One of "CR", "non-CR/non-PD", or "PD".
        """
        if new_lesions_present:
            return PD

        if not lesions:
            return CR

        # Check if all lesions have disappeared or all nodal are < 10mm SA
        all_resolved = True
        for les in lesions:
            if _is_nodal(les):
                if les.short_axis_mm >= _NODAL_CR_THRESHOLD_MM:
                    all_resolved = False
                    break
            else:
                if les.volume_mm3 > 0 or les.longest_diameter_mm > 0:
                    all_resolved = False
                    break

        if all_resolved:
            return CR

        # Check for unequivocal progression
        if previous_lesions:
            prev_total_vol = sum(l.volume_mm3 for l in previous_lesions)
            cur_total_vol = sum(l.volume_mm3 for l in lesions)
            if prev_total_vol > 0:
                pct_increase = ((cur_total_vol - prev_total_vol) / prev_total_vol) * 100
                # Unequivocal progression: substantial increase and clinically meaningful
                if pct_increase >= 73:  # Roughly 20% diameter increase cubed
                    return PD

        return "non-CR/non-PD"

    # -- iRECIST extension -------------------------------------------------

    @staticmethod
    def classify_irecist(
        current_category: str,
        previous_category: str | None = None,
        on_immunotherapy: bool = False,
    ) -> str:
        """Apply iRECIST immunotherapy-specific classification.

        iRECIST modifies RECIST 1.1 for patients on immunotherapy:
            - First PD during immunotherapy -> iUPD (unconfirmed PD).
            - If PD is confirmed 4-8 weeks later -> iCPD.
            - If subsequent assessment is not PD -> back to iSD/iPR/iCR.
            - Non-PD categories are prefixed with 'i'.

        Parameters
        ----------
        current_category:
            Current RECIST 1.1 category (``CR``, ``PR``, ``SD``,
            ``PD``).
        previous_category:
            Previous iRECIST category (e.g., ``iUPD``).  ``None``
            for the first assessment.
        on_immunotherapy:
            Whether the patient is currently on immunotherapy.

        Returns
        -------
        str
            iRECIST category.
        """
        if not on_immunotherapy:
            # Not on immunotherapy → standard RECIST categories
            return current_category

        # Map standard RECIST to iRECIST prefixed equivalents
        irecist_map = {
            CR: ICR,
            PR: IPR,
            SD: ISD,
        }

        if current_category == PD:
            if previous_category == IUPD:
                # Confirmed PD after prior unconfirmed PD
                return ICPD
            else:
                # First PD on immunotherapy → unconfirmed
                return IUPD

        # Non-PD after iUPD → reset (lesion may have pseudo-progressed)
        if previous_category == IUPD:
            return irecist_map.get(current_category, ISD)

        return irecist_map.get(current_category, current_category)

    # -- Best overall response ---------------------------------------------

    @staticmethod
    def compute_best_overall_response(
        responses: list[RECISTResponse],
    ) -> str:
        """Determine best overall response from a sequence of assessments.

        Per RECIST 1.1, the best overall response is the best response
        recorded from the start of treatment until disease progression,
        taking into account confirmation requirements.

        Parameters
        ----------
        responses:
            Time-ordered RECIST responses.

        Returns
        -------
        str
            Best overall response category.
        """
        if not responses:
            return NE

        # Priority order: CR > PR > SD > PD > NE
        priority = {CR: 0, PR: 1, SD: 2, PD: 3, NE: 4}
        best = NE
        best_priority = priority.get(NE, 4)

        for r in responses:
            cat = r.category
            p = priority.get(cat, 4)
            if p < best_priority:
                best = cat
                best_priority = p
            if cat == PD:
                break  # Stop at first progression

        return best

    # -- Full RECIST 1.1 classification ------------------------------------

    def classify(
        self,
        measurements: list[Measurement],
        baseline: list[Measurement],
        nadir: list[Measurement],
        lesions: list[Lesion] | None = None,
        baseline_lesions: list[Lesion] | None = None,
        new_lesions: bool = False,
    ) -> RECISTResponse:
        """Full RECIST 1.1 classification satisfying RECISTProtocol.

        Parameters
        ----------
        measurements:
            Current time-point target measurements.
        baseline:
            Baseline time-point target measurements.
        nadir:
            Nadir (smallest sum) target measurements.
        lesions:
            Optional current lesions for nodal classification.
        baseline_lesions:
            Optional baseline lesions for target selection.
        new_lesions:
            Whether unequivocal new lesions are present.

        Returns
        -------
        RECISTResponse
            The computed classification.
        """
        current_sum = self.compute_sum_of_diameters(measurements, lesions)
        baseline_sum = self.compute_sum_of_diameters(baseline, baseline_lesions)
        nadir_sum = self.compute_sum_of_diameters(nadir, baseline_lesions)

        # Extract nodal short axes for CR determination
        nodal_sa: list[float] | None = None
        if lesions:
            nodal_sa = [
                m.metadata.get("short_axis_mm", m.diameter_mm)
                for m, l in zip(measurements, lesions)
                if l and _is_nodal(l)
            ]

        return self.classify_response(
            current_sum=current_sum,
            baseline_sum=baseline_sum,
            nadir_sum=nadir_sum,
            new_lesions=new_lesions,
            nodal_short_axes=nodal_sa,
        )
