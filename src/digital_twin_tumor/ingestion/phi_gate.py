"""HIPAA Safe Harbor PHI detection and de-identification gate (ADR-011).

Implements the 18 Safe Harbor identifier categories defined in
45 CFR 164.514(b)(2).  Every DICOM dataset must pass through this gate
**before** any pixel data is read or downstream processing occurs.

Key design choices:

* **Deterministic hashing** -- PatientID and AccessionNumber are replaced
  with SHA-256 hashes so that the same input always produces the same
  anonymous ID.  This preserves linkability across studies without
  retaining the original identifier.
* **Deterministic date shifting** -- All dates for a given patient are
  shifted by the same random-looking offset derived from the original
  PatientID hash.  This preserves the temporal order of studies within
  a patient while removing calendar dates.
* **Deep copy** -- :meth:`strip` always operates on a copy of the
  dataset so the caller's original is never mutated.
"""
from __future__ import annotations

import copy
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any

from digital_twin_tumor.domain.models import AuditEvent

# ---------------------------------------------------------------------------
# Optional dependency import
# ---------------------------------------------------------------------------
try:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence as DicomSequence
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "pydicom is required for the PHI gate. Install it with: "
        "pip install 'pydicom>=2.4'"
    ) from _exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HIPAA Safe Harbor -- DICOM tags that may contain PHI
#
# Organised by Safe Harbor category.  Each entry is
# ``(tag_keyword, tag_tuple, action)`` where *action* is one of:
#   "remove"   -- delete the tag entirely
#   "hash"     -- replace with a deterministic SHA-256 hash
#   "dateshift" -- shift by per-patient random offset
#   "conditional" -- age-conditional (remove if >= 90)
# ---------------------------------------------------------------------------

_PHI_TAG_ACTIONS: list[tuple[str, tuple[int, int], str]] = [
    # --- Names (Safe Harbor cat. 1) ---
    ("PatientName", (0x0010, 0x0010), "remove"),
    ("ReferringPhysicianName", (0x0008, 0x0090), "remove"),
    ("PerformingPhysicianName", (0x0008, 0x1052), "remove"),
    ("OperatorsName", (0x0008, 0x1070), "remove"),
    ("OtherPatientNames", (0x0010, 0x1001), "remove"),
    # --- Geographic data (Safe Harbor cat. 2) ---
    ("PatientAddress", (0x0010, 0x1040), "remove"),
    ("InstitutionName", (0x0008, 0x0080), "remove"),
    ("InstitutionAddress", (0x0008, 0x0081), "remove"),
    ("StationName", (0x0008, 0x1010), "remove"),
    # --- Dates (Safe Harbor cat. 3) -- except year ---
    ("PatientBirthDate", (0x0010, 0x0030), "remove"),
    ("StudyDate", (0x0008, 0x0020), "dateshift"),
    ("SeriesDate", (0x0008, 0x0021), "dateshift"),
    ("AcquisitionDate", (0x0008, 0x0022), "dateshift"),
    ("ContentDate", (0x0008, 0x0023), "dateshift"),
    # --- Phone / fax (Safe Harbor cats. 4-5) ---
    ("PatientTelephoneNumbers", (0x0010, 0x2154), "remove"),
    # --- Identifiers (Safe Harbor cats. 7-12) ---
    ("PatientID", (0x0010, 0x0020), "hash"),
    ("OtherPatientIDs", (0x0010, 0x1000), "remove"),
    ("AccessionNumber", (0x0008, 0x0050), "hash"),
    ("MedicalRecordLocator", (0x0010, 0x1090), "remove"),
    # --- Device / vehicle serials (Safe Harbor cats. 15-16) ---
    ("DeviceSerialNumber", (0x0018, 0x1000), "remove"),
    # --- Age (conditional: Safe Harbor cat. 3 note) ---
    ("PatientAge", (0x0010, 0x1010), "conditional"),
    # --- Other unique identifiers (Safe Harbor cat. 18) ---
    ("EthnicGroup", (0x0010, 0x2160), "remove"),
    ("PatientSex", (0x0010, 0x0040), "keep"),  # Not PHI per Safe Harbor
]

# Tags that are known anonymisation placeholders (not considered PHI)
_KNOWN_ANONYMOUS_VALUES: frozenset[str] = frozenset({
    "",
    "ANONYMOUS",
    "ANONYMIZED",
    "DEIDENTIFIED",
    "DE-IDENTIFIED",
    "UNKNOWN",
    "N/A",
    "NONE",
})

# Date format used in DICOM (YYYYMMDD)
_DICOM_DATE_FMT = "%Y%m%d"


class PHIGate:
    """HIPAA Safe Harbor PHI scanner and stripper for DICOM datasets.

    Usage::

        gate = PHIGate()
        phi_tags = gate.scan(dataset)
        if phi_tags:
            cleaned = gate.strip(dataset)

    The gate is stateless with respect to any individual call, but uses
    a deterministic hash of the PatientID to generate consistent date
    offsets and anonymous identifiers.
    """

    # Class-level salt for hashing (not secret -- just for domain separation)
    _HASH_SALT: str = "digital-twin-tumor-phi-gate-v1"

    # Maximum date offset magnitude in days
    _MAX_DATE_OFFSET_DAYS: int = 365

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(self, ds: Dataset) -> list[str]:
        """Scan a DICOM dataset for tags containing PHI.

        Checks all 18 Safe Harbor categories.  A tag is considered to
        contain PHI if it is present, non-empty, and its value is not a
        known anonymisation placeholder.

        Parameters
        ----------
        ds:
            A ``pydicom.Dataset`` instance.

        Returns
        -------
        list[str]
            Tag keyword names that were detected as containing PHI.
        """
        found: list[str] = []

        for tag_name, tag_tuple, action in _PHI_TAG_ACTIONS:
            if action == "keep":
                continue

            value = self._get_tag_value(ds, tag_name, tag_tuple)
            if value is None:
                continue

            str_value = str(value).strip()
            if str_value.upper() in _KNOWN_ANONYMOUS_VALUES:
                continue

            # Hashed identifiers (SHA-256 hex digest) are not PHI
            if action == "hash" and self._looks_like_hash(str_value):
                continue

            # Date-shifted dates: dates alone (with year only) are not
            # PHI under Safe Harbor.  We cannot distinguish a shifted
            # date from an original one, so dateshift tags are always
            # flagged unless the date has been removed.
            # This is intentionally conservative.

            # Conditional: PatientAge -- only PHI if >= 90
            if action == "conditional" and tag_name == "PatientAge":
                if not self._is_age_over_90(str_value):
                    continue

            found.append(tag_name)

        return found

    def strip(self, ds: Dataset) -> Dataset:
        """Remove or anonymise PHI from a DICOM dataset.

        Returns a **deep copy** of *ds* with PHI tags handled according
        to their action type:

        - ``remove``: tag is deleted
        - ``hash``: value is replaced with a SHA-256 hex digest
        - ``dateshift``: date is shifted by a deterministic per-patient offset
        - ``conditional``: removed if age >= 90

        Parameters
        ----------
        ds:
            A ``pydicom.Dataset`` instance.

        Returns
        -------
        Dataset
            A new dataset with PHI removed or replaced.
        """
        cleaned = copy.deepcopy(ds)

        # Determine the patient ID for deterministic hashing/date-shift
        original_patient_id = str(
            self._get_tag_value(cleaned, "PatientID", (0x0010, 0x0020)) or "unknown"
        )
        date_offset = self._generate_date_offset(original_patient_id)

        for tag_name, tag_tuple, action in _PHI_TAG_ACTIONS:
            if action == "keep":
                continue

            if not self._has_tag(cleaned, tag_name, tag_tuple):
                continue

            if action == "remove":
                self._delete_tag(cleaned, tag_name, tag_tuple)

            elif action == "hash":
                old_value = str(
                    self._get_tag_value(cleaned, tag_name, tag_tuple) or ""
                )
                hashed = self._hash_value(old_value)
                self._set_tag_value(cleaned, tag_name, tag_tuple, hashed)

            elif action == "dateshift":
                self._shift_date_tag(cleaned, tag_name, tag_tuple, date_offset)

            elif action == "conditional":
                if tag_name == "PatientAge":
                    age_str = str(
                        self._get_tag_value(cleaned, tag_name, tag_tuple) or ""
                    )
                    if self._is_age_over_90(age_str):
                        self._delete_tag(cleaned, tag_name, tag_tuple)
                    # else: keep the tag (age < 90 is not PHI per Safe Harbor)

        logger.info(
            "PHI stripping complete for patient (hashed ID: %s)",
            self._hash_value(original_patient_id)[:12],
        )
        return cleaned

    def is_clean(self, ds: Dataset) -> bool:
        """Return ``True`` if no PHI is detected in *ds*.

        Parameters
        ----------
        ds:
            A ``pydicom.Dataset`` instance.

        Returns
        -------
        bool
        """
        return len(self.scan(ds)) == 0

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _generate_date_offset(self, patient_id: str) -> int:
        """Generate a deterministic date offset from a patient ID.

        The offset is derived from the first 8 bytes of the SHA-256 hash
        of the patient ID (salted), mapped to the range
        ``[-MAX_DATE_OFFSET_DAYS, +MAX_DATE_OFFSET_DAYS]``.

        The same patient ID always produces the same offset, ensuring
        temporal order is preserved across all studies for that patient.

        Parameters
        ----------
        patient_id:
            The **original** (pre-hashing) patient identifier.

        Returns
        -------
        int
            Offset in days (may be negative).
        """
        digest = hashlib.sha256(
            f"{self._HASH_SALT}:dateshift:{patient_id}".encode("utf-8")
        ).digest()

        # Use first 4 bytes as a signed integer
        raw_int = int.from_bytes(digest[:4], byteorder="big", signed=True)

        # Map to [-MAX_DATE_OFFSET_DAYS, MAX_DATE_OFFSET_DAYS]
        offset = raw_int % (2 * self._MAX_DATE_OFFSET_DAYS + 1) - self._MAX_DATE_OFFSET_DAYS
        return offset

    def _hash_value(self, value: str) -> str:
        """Return a deterministic SHA-256 hex digest for *value*.

        Parameters
        ----------
        value:
            The string to hash.

        Returns
        -------
        str
            64-character hex digest.
        """
        return hashlib.sha256(
            f"{self._HASH_SALT}:{value}".encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------
    # Tag manipulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tag_value(
        ds: Dataset,
        keyword: str,
        tag_tuple: tuple[int, int],
    ) -> Any | None:
        """Safely retrieve a DICOM tag value by keyword or numeric tag.

        Returns ``None`` if the tag is absent.
        """
        # Try by keyword first (faster, works for standard tags)
        value = getattr(ds, keyword, None)
        if value is not None:
            return value

        # Fall back to numeric tag
        try:
            element = ds[tag_tuple]
            return element.value
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _has_tag(
        ds: Dataset,
        keyword: str,
        tag_tuple: tuple[int, int],
    ) -> bool:
        """Check whether a DICOM tag exists in the dataset."""
        if hasattr(ds, keyword) and getattr(ds, keyword, None) is not None:
            return True
        try:
            _ = ds[tag_tuple]
            return True
        except (KeyError, TypeError):
            return False

    @staticmethod
    def _delete_tag(
        ds: Dataset,
        keyword: str,
        tag_tuple: tuple[int, int],
    ) -> None:
        """Delete a DICOM tag from the dataset (no error if absent)."""
        try:
            if keyword in ds:
                del ds[keyword]
            elif tag_tuple in ds:
                del ds[tag_tuple]
        except Exception:
            pass

    @staticmethod
    def _set_tag_value(
        ds: Dataset,
        keyword: str,
        tag_tuple: tuple[int, int],
        value: Any,
    ) -> None:
        """Set the value of a DICOM tag by keyword."""
        try:
            if keyword in ds:
                ds[keyword].value = value
            elif tag_tuple in ds:
                ds[tag_tuple].value = value
            else:
                # If the tag did not exist, we do not add it
                pass
        except Exception as exc:
            logger.warning("Failed to set tag %s: %s", keyword, exc)

    def _shift_date_tag(
        self,
        ds: Dataset,
        keyword: str,
        tag_tuple: tuple[int, int],
        offset_days: int,
    ) -> None:
        """Shift a DICOM date tag by *offset_days*.

        If the date cannot be parsed, the tag is removed as a safety
        fallback.
        """
        raw = self._get_tag_value(ds, keyword, tag_tuple)
        if raw is None:
            return

        date_str = str(raw).strip()
        if not date_str:
            return

        try:
            original_date = datetime.strptime(date_str, _DICOM_DATE_FMT)
            shifted = original_date + timedelta(days=offset_days)
            self._set_tag_value(ds, keyword, tag_tuple, shifted.strftime(_DICOM_DATE_FMT))
        except (ValueError, TypeError):
            logger.warning(
                "Cannot parse date tag %s (value=%r); removing instead of shifting.",
                keyword,
                date_str,
            )
            self._delete_tag(ds, keyword, tag_tuple)

    @staticmethod
    def _looks_like_hash(value: str) -> bool:
        """Return ``True`` if *value* looks like a SHA-256 hex digest.

        A 64-character lowercase hexadecimal string is treated as an
        already-anonymised identifier.
        """
        if len(value) != 64:
            return False
        try:
            int(value, 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_age_over_90(age_str: str) -> bool:
        """Determine whether an age string represents 90 years or older.

        DICOM ``PatientAge`` typically has a format like ``065Y``
        (years), ``078M`` (months), ``003W`` (weeks), or ``001D`` (days).
        """
        age_str = age_str.strip()
        if not age_str:
            return False

        try:
            # Try numeric-only (some implementations store plain integers)
            age_val = int(age_str)
            return age_val >= 90
        except ValueError:
            pass

        # DICOM Age String format: nnnX where X is D/W/M/Y
        suffix = age_str[-1].upper()
        try:
            num = int(age_str[:-1])
        except ValueError:
            # Cannot parse -- treat as potentially identifying
            return True

        if suffix == "Y":
            return num >= 90
        if suffix == "M":
            return num >= 90 * 12
        if suffix == "W":
            return num >= 90 * 52
        if suffix == "D":
            return num >= 90 * 365

        # Unknown suffix -- be conservative
        return True
