"""Main Gradio Blocks application -- seven-tab interface (ADR-002, ADR-015).

Tabs: Dashboard, Upload & Preprocess, Annotate & Measure, Track Lesions,
      Simulate, Narrate, Agentic Workflow.
State is shared across tabs via ``gr.State``.
"""
from __future__ import annotations

import json, logging, os, tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import spaces  # type: ignore[import-untyped]
    _HF_SPACES = True
except ImportError:
    _HF_SPACES = False

try:
    import gradio as gr  # type: ignore[import-untyped]
except ImportError as _exc:  # pragma: no cover
    raise ImportError("Gradio is required: pip install 'gradio>=4.0'") from _exc

from digital_twin_tumor.ui.components import (
    create_disclaimer_banner, create_growth_plot,
    create_identity_graph_plot, create_measurement_overlay,
    create_recist_waterfall, extract_slice, format_measurement_table,
)
from digital_twin_tumor.ui.theme import get_theme, get_css
from digital_twin_tumor.ui.agentic import pipeline_html, run_pipeline
from digital_twin_tumor.ui import callbacks as _cb

# Backend imports with graceful degradation --------------------------------
from digital_twin_tumor.domain.models import ProcessedVolume

_OK: dict[str, bool] = {}
try:
    from digital_twin_tumor.ingestion.dicom_loader import load_dicom_series
    from digital_twin_tumor.ingestion.nifti_loader import load_nifti
    _OK["ingest"] = True
except Exception:
    _OK["ingest"] = False
try:
    from digital_twin_tumor.preprocessing.normalize import normalize_volume
    _OK["preproc"] = True
except Exception:
    _OK["preproc"] = False
try:
    from digital_twin_tumor.measurement.segmentation import SegmentationEngine
    _OK["seg"] = True
except Exception:
    _OK["seg"] = False
try:
    from digital_twin_tumor.twin_engine.growth_models import (
        ExponentialGrowth, GompertzGrowth, LogisticGrowth,
    )
    _OK["twin"] = True
except Exception:
    _OK["twin"] = False

_narr_gen = None
_reasoner = None
try:
    from digital_twin_tumor.reasoning import NarrativeGenerator, MedGemmaClient
    from digital_twin_tumor.reasoning.medgemma import MedGemmaReasoner
    _narr_gen = NarrativeGenerator()
    _OK["medgemma"] = isinstance(_narr_gen.client, MedGemmaClient)
    # Lazy-init: MedGemmaReasoner no longer loads the model in __init__,
    # so this is safe even without a GPU (ZeroGPU compatibility).
    _reasoner = MedGemmaReasoner()
    _OK["reasoner"] = False  # will become True on first GPU call
except Exception:
    _narr_gen = None
    _reasoner = None
    _OK["medgemma"] = False
    _OK["reasoner"] = False

_DEMO_DB = os.environ.get("DTT_DEMO_DB", ".cache/demo.db")

# Sample images for Upload tab ------------------------------------------------
_SAMPLE_DIR = Path(__file__).resolve().parents[3] / "data" / "sample_images"

_SAMPLE_IMAGES: dict[str, tuple[str, str]] = {}  # label -> (relative_path, modality)
_nifti_dir = _SAMPLE_DIR / "nifti"
_dicom_dir = _SAMPLE_DIR / "dicom"
if _nifti_dir.is_dir():
    for _n, _m, _d in [
        ("synthetic_tumor_ct.nii.gz", "CT", "CT Baseline -- 15 mm tumor (128x128x64)"),
        ("synthetic_tumor_ct_followup.nii.gz", "CT", "CT Follow-up -- 19 mm tumor (+105% growth)"),
        ("synthetic_tumor_mr.nii.gz", "MRI", "MR T1-Contrast Tumor (128x128x64)"),
    ]:
        _p = _nifti_dir / _n
        if _p.exists():
            _SAMPLE_IMAGES[_d] = (str(_p), _m)
if _dicom_dir.is_dir():
    for _n, _m, _d in [
        ("ct_series/CT_small.dcm", "CT", "DICOM CT Slice (128x128)"),
        ("mr_series/MR_small.dcm", "MRI", "DICOM MR Slice (64x64)"),
    ]:
        _p = _dicom_dir / _n
        if _p.exists():
            _SAMPLE_IMAGES[_d] = (str(_p), _m)
_demo_loader = None
try:
    if os.path.exists(_DEMO_DB):
        from digital_twin_tumor.data.demo_loader import DemoLoader
        _demo_loader = DemoLoader(_DEMO_DB)
        _OK["demo"] = True
except Exception:
    _OK["demo"] = False


def _empty() -> dict[str, Any]:
    return {"volume": None, "volume_metadata": {}, "lesions": [],
            "measurements": [], "tracking_graph_json": None,
            "growth_results": [], "simulations": [],
            "recist_responses": [], "narrative": None, "timepoints": [],
            "therapy_events": [], "patient_metadata": {},
            "simulation_results_full": []}


def _mg_label() -> str:
    return "MedGemma: Active" if _OK.get("medgemma") else "MedGemma: Template Mode"


# Dashboard ----------------------------------------------------------------
def _get_demo_choices() -> list[str]:
    if _demo_loader is None:
        return ["No demo data available -- run scripts/generate_demo_data.py"]
    return _demo_loader.get_patient_choices()


def _on_patient_select(choice_label, state):
    if _demo_loader is None:
        return "<p style='color:#f87171;'>Demo DB not found.</p>", "", _empty()
    patients, choices = _demo_loader.list_patients(), _demo_loader.get_patient_choices()
    try:
        idx = choices.index(choice_label)
    except ValueError:
        return "<p>Select a patient.</p>", "", state or _empty()
    pid = patients[idx]["patient_id"]
    ns = _demo_loader.build_ui_state(pid)
    if not ns:
        return "<p style='color:#f87171;'>Failed to load.</p>", "", _empty()
    gr.Info(f"Loaded: {patients[idx]['label']}")
    return _demo_loader.get_patient_dashboard_html(pid), _demo_loader.get_timeline_html(pid), ns


def _demo_slice(index, state):
    vol = (state or {}).get("volume")
    return extract_slice(np.asarray(vol), int(index)) if vol is not None else np.zeros((256, 256), dtype=np.uint8)


def _demo_tp(tp_choice, state):
    return _cb.demo_timepoint_volume(tp_choice, state, _empty)


def _demo_wf(s):
    return create_recist_waterfall((s or {}).get("recist_responses", []))


def _demo_tg(s):
    gj = (s or {}).get("tracking_graph_json")
    return create_identity_graph_plot(json.loads(gj) if gj else None)


def _demo_gp(s):
    return _cb.demo_growth_plot(s, _empty)


# Tab callbacks ------------------------------------------------------------
def _is_nifti(path: Path) -> bool:
    """Check whether *path* looks like a NIfTI file (.nii or .nii.gz)."""
    suffixes = [s.lower() for s in path.suffixes]
    return suffixes == [".nii"] or suffixes[-2:] == [".nii", ".gz"]


def _resolve_upload_path(file_entry) -> Path:
    """Turn a Gradio file entry into an absolute Path.

    Gradio 6 returns ``NamedString`` objects (str subclass where ``.name``
    equals the full path).  Older Gradio / ``TemporaryUploadedFile`` objects
    expose ``.name`` as the filesystem path as well.  Plain ``str`` and
    ``pathlib.Path`` objects are also handled.  We always fall back to
    ``str()`` so that ``Path.name`` (which returns only the basename) is
    never used accidentally.
    """
    raw = str(file_entry)
    return Path(raw)


def _on_process_or_sample(files, modality, window, state, sample_choice=None):
    """Process Scan button -- uses uploaded file, or falls back to sample."""
    if not files and sample_choice and sample_choice in _SAMPLE_IMAGES:
        return _on_sample_load(sample_choice, state)
    img, meta, status, slider, ns = _on_upload(files, modality, window, state)
    return img, meta, status, slider, modality, ns


def _on_upload(files, modality, window, state):
    if not files:
        gr.Warning("No files selected.")
        return None, {}, "No files uploaded.", gr.update(value=0), state or _empty()
    state = dict(state) if state else _empty()
    first = _resolve_upload_path(files[0])
    try:
        if first.suffix.lower() == ".dcm" or first.is_dir():
            if not _OK.get("ingest"):
                raise RuntimeError("DICOM loader unavailable.")
            # Single .dcm file -> use its parent directory for series loading
            vol = load_dicom_series(first.parent if first.suffix.lower() == ".dcm" else first)
        elif _is_nifti(first):
            if not _OK.get("ingest"):
                raise RuntimeError("NIfTI loader unavailable.")
            vol = load_nifti(first)
        else:
            gr.Warning(f"Unsupported format: {first.name}")
            return None, {}, f"Unsupported format: {first.name}", gr.update(value=0), state

        # Normalise -- fall back to simple [0,1] rescaling for unsupported modalities
        if vol is not None and _OK.get("preproc"):
            vol_modality = vol.modality.upper().strip()
            if vol_modality in ("CT", "MR", "MRI"):
                vol = normalize_volume(vol, ct_window=window if modality == "CT" else "soft_tissue")
            else:
                # PET / other: percentile-clip and rescale to [0,1]
                logger.info("Modality '%s' not CT/MR; using percentile normalisation.", vol.modality)
                pdata = vol.pixel_data.astype(np.float32)
                lo, hi = float(np.percentile(pdata, 1)), float(np.percentile(pdata, 99))
                if hi - lo > 1e-8:
                    pdata = np.clip(pdata, lo, hi)
                    pdata = (pdata - lo) / (hi - lo)
                else:
                    pdata = np.zeros_like(pdata)
                vol = ProcessedVolume(
                    pixel_data=pdata, spacing=vol.spacing, origin=vol.origin,
                    direction=vol.direction, modality=vol.modality, metadata=vol.metadata,
                    mask=vol.mask, selected_slices=list(vol.selected_slices),
                )

        if vol is not None:
            state["volume"] = vol.pixel_data
            state["volume_metadata"] = {
                k: (v if isinstance(v, (str, int, float, bool, list)) else str(v))
                for k, v in vol.metadata.items()}
            state["volume_metadata"]["shape"] = list(vol.pixel_data.shape)
            state["volume_metadata"]["spacing"] = [vol.spacing.x, vol.spacing.y, vol.spacing.z]
            n_slices = vol.pixel_data.shape[0]
            mid = n_slices // 2
            gr.Info(f"Upload complete. {n_slices} slices loaded.")
            return (
                extract_slice(vol.pixel_data, mid),
                state["volume_metadata"],
                f"Loaded {n_slices} slices ({vol.pixel_data.shape[1]}x{vol.pixel_data.shape[2]}).",
                gr.update(value=mid, minimum=0, maximum=max(n_slices - 1, 0)),
                state,
            )
    except Exception as exc:
        logger.exception("Upload failed")
        gr.Warning(f"Upload error: {exc}")
    return None, {}, "Upload failed. Check the file and try again.", gr.update(value=0), state


def _on_sample_load(choice, state):
    """Load a sample image by label and process it through the ingestion pipeline."""
    if not choice or choice not in _SAMPLE_IMAGES:
        gr.Warning("Select a sample image first.")
        return None, {}, "No sample selected.", gr.update(value=0), "CT", state or _empty()
    fpath, modality = _SAMPLE_IMAGES[choice]
    # Delegate to _on_upload with a fake file list containing the real path
    class _FakePath:
        def __init__(self, p): self._p = p
        def __str__(self): return self._p
    img, meta, status, slider, ns = _on_upload([_FakePath(fpath)], modality, "soft_tissue", state)
    return img, meta, status, slider, modality, ns


def _on_slice(index, state):
    vol = (state or {}).get("volume")
    return extract_slice(np.asarray(vol), int(index)) if vol is not None else None


# MedGemma integration callbacks -------------------------------------------

def _mg_analyze_slice(slice_idx, state):
    """Analyze an imaging slice with MedGemma multimodal (MedSigLIP encoder)."""
    vol = (state or {}).get("volume")
    if vol is None:
        return "*Load a volume first.*"
    slc = extract_slice(np.asarray(vol), int(slice_idx))
    meta = (state or {}).get("volume_metadata", {})
    modality = meta.get("modality", "CT")
    context = f"Modality: {modality}. Shape: {meta.get('shape', 'unknown')}. Slice {int(slice_idx)}."
    if _reasoner is not None:
        try:
            return _reasoner.analyze_imaging_slice(slc, context=context)
        except Exception as exc:
            logger.warning("MedGemma slice analysis failed: %s", exc)
            return f"*Analysis error: {exc}*"
    return "*MedGemma not available. Load model for imaging analysis.*"


def _mg_tumor_board(state):
    """Generate a tumor board summary from current state."""
    state = dict(state) if state else _empty()
    if _reasoner is None:
        return "*MedGemma reasoner not available.*"
    patient_data = {
        "patient_metadata": state.get("patient_metadata", {}),
        "measurements": state.get("measurements", []),
        "recist_responses": state.get("recist_responses", []),
        "therapy_events": state.get("therapy_events", []),
        "growth_results": state.get("growth_results", []),
        "timepoints": state.get("timepoints", []),
        "simulation_results_full": state.get("simulation_results_full", []),
    }
    if not patient_data["measurements"] and not patient_data["patient_metadata"]:
        return "*Load patient data or add measurements first.*"
    try:
        return _reasoner.generate_tumor_board_summary(patient_data)
    except Exception as exc:
        logger.warning("MedGemma tumor board failed: %s", exc)
        return f"*Tumor board error: {exc}*"


def _mg_interpret_sim(state):
    """Interpret simulation results with MedGemma counterfactual reasoning."""
    state = dict(state) if state else _empty()
    if _reasoner is None:
        return "*MedGemma reasoner not available.*"
    ms = state.get("measurements", [])
    sims = state.get("simulation_results_full", [])
    if len(ms) < 2:
        return "*Need >=2 measurements to interpret simulations.*"
    # Build baseline trajectory from measurements
    times = [float(i) for i in range(len(ms))]
    vols = [float(m.get("volume_mm3", 0)) for m in ms]
    baseline = {"name": "Observed", "times": times, "volumes": vols}
    # Build counterfactual trajectories from simulation state
    cf_list = []
    for s in state.get("simulations", []):
        name = s.get("scenario_name", "Scenario")
        cf_list.append({"name": name, "times": times,
                        "volumes": [v * 1.2 for v in vols]})  # fallback
    # Extract any richer sim data
    for s in sims:
        if hasattr(s, "time_points") and hasattr(s, "predicted_volumes"):
            cf_list.append({"name": getattr(s, "scenario_name", "Sim"),
                            "times": list(s.time_points),
                            "volumes": list(s.predicted_volumes)})
    if not cf_list:
        cf_list.append({"name": "Natural History (projected)",
                        "times": times + [times[-1] + 4, times[-1] + 8],
                        "volumes": vols + [vols[-1] * 1.3, vols[-1] * 1.7]})
    therapy = ", ".join(e.get("dose", e.get("therapy_type", ""))
                        for e in state.get("therapy_events", [])) or "No therapy recorded"
    try:
        return _reasoner.generate_counterfactual_interpretation(baseline, cf_list, therapy)
    except Exception as exc:
        logger.warning("MedGemma CF interpretation failed: %s", exc)
        return f"*Interpretation error: {exc}*"


def _mg_model_info_html() -> str:
    """Return HTML badge with MedGemma model details."""
    if _reasoner is None:
        return ""
    info = _reasoner.get_model_info()
    avail = info.get("is_available", False)
    bc = "#10b981" if avail else "#f59e0b"
    status = "Active (GPU)" if avail else f"Template ({info.get('load_error', 'N/A')})"
    return (
        f'<div style="background:rgba(15,17,23,0.85);border:1px solid {bc};'
        f'border-radius:10px;padding:10px 16px;margin:8px 0;font-size:0.82rem;">'
        f'<span style="color:{bc};font-weight:700;">MedGemma</span> '
        f'<span style="color:#94a3b8;">| {info.get("model_name","?")} '
        f'({info.get("parameters","?")}, {info.get("type","?")}) '
        f'| Encoder: {info.get("image_encoder","N/A")} '
        f'| MedQA: {info.get("medqa_score","?")} '
        f'| Status: {status}</span></div>'
    )


def _on_segment(bbox_str, slice_idx, state):
    state = dict(state) if state else _empty()
    vol = state.get("volume")
    if vol is None:
        gr.Warning("Load a volume first."); return None, "0.0", "No volume.", state
    slc = extract_slice(np.asarray(vol), int(slice_idx))
    # Auto-generate a centre bbox when the field is empty or blank
    if not bbox_str or not bbox_str.strip():
        h, w = slc.shape[:2]
        cx, cy = w // 2, h // 2
        sz = min(h, w) // 4
        bbox_str = f"{cx - sz},{cy - sz},{cx + sz},{cy + sz}"
    try:
        coords = [int(x.strip()) for x in bbox_str.split(",")]
        if len(coords) != 4: raise ValueError("Need x0,y0,x1,y1")
        bbox = tuple(coords)
    except Exception as exc:
        gr.Warning(f"Bad bbox: {exc}")
        return create_measurement_overlay(slc), "0.0", str(exc), state
    mask = np.zeros_like(slc, dtype=np.uint8)
    if _OK.get("seg"):
        try: mask = SegmentationEngine(model_name="medsam", device="cpu").segment_with_box(slc, bbox)
        except Exception: pass
    vc = int(np.sum(mask > 0))
    sp = state.get("volume_metadata", {}).get("spacing", [1, 1, 1])
    ev = vc * float(sp[0]) * float(sp[1]) * float(sp[2])
    import uuid
    state.setdefault("measurements", []).append(
        {"lesion_id": str(uuid.uuid4())[:8], "diameter_mm": 0.0,
         "volume_mm3": ev, "method": "semi-auto", "timepoint_id": "current"})
    return create_measurement_overlay(slc, mask=mask), f"{ev:.1f}", f"Voxels:{vc} Vol:{ev:.1f}", state


def _on_manual_diam(val, state):
    state = dict(state) if state else _empty()
    ms = state.get("measurements", [])
    if ms:
        ms[-1] = {**ms[-1], "diameter_mm": val, "method": "manual"}
        state["measurements"] = ms; gr.Info(f"Diameter: {val:.1f} mm")
        return f"Recorded: {val:.1f} mm", state
    gr.Warning("No measurement."); return "No measurement.", state


def _classify_recist(state):
    state = dict(state) if state else _empty()
    ms = state.get("measurements", [])
    if not ms: return "No measurements.", state
    ds = [float(m.get("diameter_mm", 0)) for m in ms]
    cur = sum(ds) if any(d > 0 for d in ds) else 0
    if cur == 0: return "Provide diameters first.", state
    rs = state.get("recist_responses", [])
    bl = rs[0]["baseline_sum"] if rs else cur
    nadir = min([r.get("nadir_sum", bl) for r in rs] + [cur])
    pbl, pna = ((cur - bl) / max(bl, 0.01)) * 100, ((cur - nadir) / max(nadir, 0.01)) * 100
    cat = "CR" if cur == 0 else "PR" if pbl <= -30 else "PD" if pna >= 20 else "SD"
    state.setdefault("recist_responses", []).append(
        {"timepoint_id": "current", "sum_of_diameters": cur, "baseline_sum": bl,
         "nadir_sum": nadir, "percent_change_from_baseline": pbl,
         "percent_change_from_nadir": pna, "category": cat})
    return f"**{cat}** | Sum: {cur:.1f} mm | BL: {pbl:+.1f}% | Nadir: {pna:+.1f}%", state


def _build_tracking(state):
    state = dict(state) if state else _empty()
    gj = state.get("tracking_graph_json")
    if gj:
        gd = json.loads(gj) if isinstance(gj, str) else gj
        n, e = gd.get("nodes", []), gd.get("edges", [])
        bl: dict[str, int] = {}
        for nd in n:
            lid = nd.get("lesion_id", "")
            bl[lid.rsplit("_tp", 1)[0] if "_tp" in lid else lid] = bl.get(lid, 0) + 1
        return create_identity_graph_plot(gd), f"N:{len(n)} E:{len(e)} L:{len(bl)}", state
    ms = state.get("measurements", [])
    if not ms:
        gr.Warning("No measurements."); return create_identity_graph_plot(None), "No data.", state
    nl = [{"id": f"{m.get('lesion_id','')}@{m.get('timepoint_id','')}",
           "lesion_id": m.get("lesion_id", ""), "timepoint_id": m.get("timepoint_id", ""),
           "volume": float(m.get("volume_mm3", 0))} for m in ms]
    bl2: dict[str, list[int]] = {}
    for i, nd in enumerate(nl): bl2.setdefault(nd["lesion_id"], []).append(i)
    el = [{"source": nl[idx[j]]["id"], "target": nl[idx[j+1]]["id"], "confidence": 0.95}
          for idx in bl2.values() for j in range(len(idx)-1)]
    gd2 = {"nodes": nl, "edges": el}; state["tracking_graph_json"] = json.dumps(gd2)
    return create_identity_graph_plot(gd2), f"N:{len(nl)} E:{len(el)} L:{len(bl2)}", state


def _run_sim(scenario, shift, mult, resist, state):
    models = (ExponentialGrowth, LogisticGrowth, GompertzGrowth) if _OK.get("twin") else None
    gr.Warning("Need >=2 measurements.") if len((dict(state) if state else _empty()).get("measurements", [])) < 2 else None
    return _cb.run_sim(scenario, shift, mult, resist, state, _empty, _OK, models)


def _gen_narrative(state):
    state = dict(state) if state else _empty()
    mode = _mg_label()
    if _narr_gen is not None and state.get("measurements"):
        try:
            from digital_twin_tumor.domain.models import Measurement, UncertaintyEstimate
            ms = [Measurement(lesion_id=m.get("lesion_id", ""), timepoint_id=m.get("timepoint_id", ""),
                              diameter_mm=float(m.get("diameter_mm", 0)),
                              volume_mm3=float(m.get("volume_mm3", 0)), method=m.get("method", "auto"))
                  for m in state.get("measurements", [])]
            unc = UncertaintyEstimate(sigma_manual=0.5, sigma_auto=0.3, sigma_scan=0.2,
                                      total_sigma=0.6, reliability="MEDIUM")
            r = _narr_gen.generate_narrative(measurements=ms, therapy_events=[], growth_results=[],
                                             simulations=[], uncertainty=unc, recist_responses=[])
            state["narrative"] = r.text
            return r.text, f"{mode} | Grounding: {'OK' if r.grounding_check else 'WARN'}", state
        except Exception as exc:
            logger.warning("MedGemma narrative failed: %s", exc)
    if state.get("patient_metadata"):
        text, info, state = _cb.demo_narrative(state, _empty)
        return text, f"{mode} | {info}", state
    ms = state.get("measurements", [])
    rs, gr_res = state.get("recist_responses", []), state.get("growth_results", [])
    secs = ["## Digital Twin Tumor Assessment Report\n",
            "> **Disclaimer:** AI-generated for research only.\n"]
    if ms:
        secs += [f"- Lesion {i}: {float(m.get('diameter_mm',0)):.1f} mm, "
                 f"{float(m.get('volume_mm3',0)):.1f} mm^3" for i, m in enumerate(ms, 1)]
    if rs: secs.append(f"\n**RECIST: {rs[-1].get('category','?')}**")
    if gr_res:
        secs += [f"- {g.get('model_name','?')}: w={g.get('weight',0):.3f}" for g in gr_res]
    text = "\n".join(secs); state["narrative"] = text
    return text, f"{mode} | {len(ms)} measurements.", state


def _export_report(state):
    text = (state or {}).get("narrative", "")
    if not text: gr.Warning("Generate narrative first."); return None
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", prefix="dtt_report_", delete=False)
    f.write(text); f.close(); gr.Info(f"Saved: {f.name}"); return f.name


def _run_agentic(state):
    return run_pipeline(dict(state) if state else _empty(), _OK.get("medgemma", False), _gen_narrative, _reasoner)


# -- ZeroGPU decorators (HF Spaces) ----------------------------------------
# GPU is only available inside @spaces.GPU-decorated functions on ZeroGPU.
if _HF_SPACES:
    _mg_tumor_board = spaces.GPU(_mg_tumor_board)
    _mg_analyze_slice = spaces.GPU(_mg_analyze_slice)
    _mg_interpret_sim = spaces.GPU(_mg_interpret_sim)
    _run_agentic = spaces.GPU(_run_agentic)


# App builder ==============================================================
def create_app() -> gr.Blocks:
    """Build the seven-tab Gradio application with demo dashboard."""
    theme, css = get_theme(), get_css()
    _js = "() => { document.body.classList.add('dark'); }"
    _kw: dict[str, Any] = {
        "title": "Digital Twin: Tumor Response Assessment",
        "theme": theme, "css": css, "js": _js,
    }
    _mg = _OK.get("medgemma", False)
    _bc, _bt = ("#10b981", "Active") if _mg else ("#f59e0b", "Template Mode")

    with gr.Blocks(**_kw) as app:
        app._dtt_theme = theme; app._dtt_css = css; app._dtt_js = _js  # type: ignore[attr-defined]
        gr.HTML(
            '<div style="text-align:center;padding:20px 0 8px;">'
            '<h1 style="font-size:2.2rem;font-weight:800;margin:0;'
            'background:linear-gradient(135deg,#22d3ee,#06b6d4,#14b8a6);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
            'background-clip:text;">Digital Twin: Tumor Response Assessment</h1>'
            '<p style="color:#94a3b8;font-size:0.95rem;margin:6px 0 0;">'
            'Medical Imaging &bull; RECIST 1.1 &bull; Growth Modelling &bull; AI Narratives'
            f'&nbsp;&nbsp;<span style="display:inline-block;background:rgba(15,17,23,0.8);'
            f'border:1px solid {_bc};border-radius:12px;padding:2px 10px;'
            f'font-size:0.75rem;color:{_bc};font-weight:600;">MedGemma: {_bt}</span></p></div>')
        gr.HTML(create_disclaimer_banner())
        st = gr.State(value=_empty())

        # Tab 0: Dashboard
        with gr.Tab("Dashboard"):
            gr.Markdown("### Patient Dashboard\nSelect a demo patient to load data across all tabs.")
            pr = gr.Radio(choices=_get_demo_choices(), label="Demo Patient", value=None, interactive=True)
            lb = gr.Button("Load Patient", variant="primary", size="lg")
            dh = gr.HTML('<div style="text-align:center;padding:40px;color:#475569;">Select a patient.</div>')
            th = gr.HTML("")
            with gr.Accordion("Volume Viewer", open=False):
                with gr.Row():
                    tdd = gr.Dropdown(choices=[], label="Timepoint", interactive=True)
                    tlb = gr.Button("Load Timepoint", variant="secondary")
                with gr.Row():
                    di = gr.Image(label="Axial Slice", type="numpy", interactive=False)
                    ds = gr.Slider(0, 63, step=1, value=32, label="Slice")
                    ti = gr.Textbox(label="Info", interactive=False)
            with gr.Row():
                with gr.Column(): dwf = gr.Plot(label="RECIST Waterfall")
                with gr.Column(): dtg = gr.Plot(label="Lesion Tracking")
            with gr.Row():
                with gr.Column():
                    dgp = gr.Plot(show_label=False); dgi = gr.Textbox(label="Model Info", interactive=False, lines=3)
            with gr.Accordion("MedGemma Tumor Board", open=False):
                gr.HTML(_mg_model_info_html())
                dtb_btn = gr.Button("Generate Tumor Board Summary", variant="secondary")
                dtb_md = gr.Markdown("*Load a patient, then click to generate a MedGemma-powered tumor board summary.*")
                dtb_btn.click(_mg_tumor_board, [st], [dtb_md])

            def _on_load(choice, state):
                d, t, ns = _on_patient_select(choice, state)
                tps = ns.get("timepoints", []) if isinstance(ns, dict) else []
                tc = [f"Week {tp.get('week','?')} ({tp.get('scan_date','')})" for tp in tps]
                wf, tg, (gp, gt) = _demo_wf(ns), _demo_tg(ns), _demo_gp(ns)
                vol = ns.get("volume") if isinstance(ns, dict) else None
                img = extract_slice(np.asarray(vol), 32) if vol is not None else np.zeros((256, 256), dtype=np.uint8)
                return d, t, ns, gr.update(choices=tc, value=tc[0] if tc else None), img, wf, tg, gp, gt

            lb.click(_on_load, [pr, st], [dh, th, st, tdd, di, dwf, dtg, dgp, dgi])
            tlb.click(_demo_tp, [tdd, st], [di, ti, st])
            ds.change(_demo_slice, [ds, st], [di])

        # Tab 1: Upload & Preprocess
        with gr.Tab("Upload & Preprocess"):
            _sample_choices = list(_SAMPLE_IMAGES.keys())
            if _sample_choices:
                gr.Markdown("### Quick Start -- Sample Images\n"
                            "Click a sample to load it instantly, or upload your own below.")
                with gr.Row():
                    with gr.Column(scale=2):
                        sample_radio = gr.Radio(
                            choices=_sample_choices, label="Sample Images",
                            value=None, interactive=True,
                        )
                    with gr.Column(scale=1):
                        sample_btn = gr.Button("Load Sample", variant="primary", size="lg")
            gr.Markdown("---\n### Upload Your Own Scan")
            with gr.Row():
                with gr.Column(scale=1):
                    fi = gr.File(label="Upload", file_types=[".dcm",".nii",".nii.gz",".zip"], file_count="multiple")
                    mod = gr.Dropdown(["CT","MRI","PET"], value="CT", label="Modality")
                    win = gr.Dropdown(["soft_tissue","lung","bone","brain","liver"], value="soft_tissue", label="Window")
                    ub = gr.Button("Process Scan", variant="primary")
                with gr.Column(scale=2):
                    si = gr.Image(label="Axial Slice", type="numpy", interactive=False)
                    ss = gr.Slider(0, 200, step=1, value=0, label="Slice")
            us = gr.Textbox(label="Status", interactive=False, lines=2); mj = gr.JSON(label="Metadata")
            if _sample_choices:
                sample_btn.click(_on_sample_load, [sample_radio, st], [si, mj, us, ss, mod, st])
                ub.click(_on_process_or_sample, [fi, mod, win, st, sample_radio], [si, mj, us, ss, mod, st])
            else:
                ub.click(_on_upload, [fi, mod, win, st], [si, mj, us, ss, st])
            ss.change(_on_slice, [ss, st], [si])
            with gr.Accordion("MedGemma Imaging Analysis", open=False):
                gr.Markdown("Use MedGemma's **MedSigLIP** encoder for multimodal medical image analysis.")
                mg_up_btn = gr.Button("Analyze Slice with MedGemma", variant="secondary")
                mg_up_md = gr.Markdown("*Load a scan first, then analyze the current slice.*")
                mg_up_btn.click(_mg_analyze_slice, [ss, st], [mg_up_md])

        # Tab 2: Annotate & Measure
        with gr.Tab("Annotate & Measure"):
            gr.Markdown("### HITL Annotation\nBBox -> segment -> measure -> RECIST.")
            with gr.Row():
                with gr.Column(scale=2):
                    ai = gr.Image(label="Annotated Slice", type="numpy", interactive=False)
                with gr.Column(scale=1):
                    a_ss = gr.Slider(0, 200, step=1, value=0, label="Slice")
                    bb = gr.Textbox(label="BBox (x0,y0,x1,y1)", value="32,32,96,96",
                                    placeholder="x0,y0,x1,y1 or leave for auto-centre")
                    sb = gr.Button("Segment", variant="primary")
                    diam = gr.Number(label="Manual Diameter (mm)", value=0.0, precision=1)
                    ob = gr.Button("Save Diameter")
                    vd = gr.Textbox(label="Volume", interactive=False)
                    si2 = gr.Textbox(label="Info", interactive=False, lines=2)
            with gr.Row():
                rb = gr.Button("RECIST", variant="secondary"); rm = gr.Markdown("*Result here.*")
            mt = gr.Dataframe(label="Measurements", headers=["ID","D(mm)","V(mm3)","Method","TP"], interactive=False)
            _u = lambda s: format_measurement_table(s.get("measurements") if s else None)
            sb.click(_on_segment, [bb, a_ss, st], [ai, vd, si2, st]).then(_u, [st], [mt])
            ob.click(_on_manual_diam, [diam, st], [si2, st]).then(_u, [st], [mt])
            rb.click(_classify_recist, [st], [rm, st])
            with gr.Accordion("MedGemma Lesion Analysis", open=False):
                gr.Markdown("MedGemma interprets segmented lesions with clinical context and RECIST criteria.")
                mg_an_btn = gr.Button("Analyze Lesion with MedGemma", variant="secondary")
                mg_an_md = gr.Markdown("*Segment a lesion first, then run MedGemma analysis.*")
                mg_an_btn.click(_mg_analyze_slice, [a_ss, st], [mg_an_md])

        # Tab 3: Track Lesions
        with gr.Tab("Track Lesions"):
            gr.Markdown("### Lesion Identity Tracking")
            tb = gr.Button("Build Graph", variant="primary")
            tp = gr.Plot(label="Identity Graph"); ts = gr.Textbox(label="Summary", interactive=False)
            wp = gr.Plot(label="RECIST Waterfall")
            tb.click(_build_tracking, [st], [tp, ts, st]).then(
                lambda s: create_recist_waterfall(s.get("recist_responses") if s else None), [st], [wp])
            with gr.Accordion("MedGemma Tracking Interpretation", open=False):
                gr.Markdown("MedGemma analyzes lesion tracking trajectories and provides clinical interpretation.")
                mg_tk_btn = gr.Button("Interpret Tracking with MedGemma", variant="secondary")
                mg_tk_md = gr.Markdown("*Build the tracking graph first, then interpret.*")
                mg_tk_btn.click(_mg_tumor_board, [st], [mg_tk_md])

        # Tab 4: Simulate
        with gr.Tab("Simulate"):
            gr.Markdown("### Digital Twin Growth Simulation")
            with gr.Row():
                with gr.Column(scale=1):
                    sd = gr.Dropdown(["Natural history (no treatment)","Earlier treatment",
                                      "Later treatment","Treatment resistance","Dose escalation"],
                                     value="Natural history (no treatment)", label="Scenario")
                    shs = gr.Slider(-12, 12, step=0.5, value=0, label="Therapy Shift")
                    ems = gr.Slider(0, 2, step=0.05, value=0.5, label="Effect Mult.")
                    ros = gr.Slider(0, 52, step=1, value=12, label="Resist. Onset")
                    smb = gr.Button("Run Simulation", variant="primary")
                with gr.Column(scale=2):
                    gp = gr.Plot(show_label=False)
                    mi = gr.Textbox(label="Model Info", interactive=False, lines=4)
            smb.click(_run_sim, [sd, shs, ems, ros, st], [gp, mi, st])
            with gr.Accordion("MedGemma Counterfactual Analysis", open=False):
                gr.Markdown("MedGemma interprets simulation results with **counterfactual reasoning** "
                            "-- comparing observed vs. hypothetical treatment trajectories.")
                mg_cf_btn = gr.Button("Interpret Simulation with MedGemma", variant="secondary")
                mg_cf_md = gr.Markdown("*Run a simulation first, then interpret the counterfactual scenarios.*")
                mg_cf_btn.click(_mg_interpret_sim, [st], [mg_cf_md])

        # Tab 5: Narrate
        with gr.Tab("Narrate"):
            gr.HTML(create_disclaimer_banner())
            gr.Markdown("### AI Narrative -- **Not for clinical use.**")
            gr.HTML(_mg_model_info_html() or
                    f'<div style="margin-bottom:8px;"><span style="background:rgba(15,17,23,0.8);'
                    f'border:1px solid {_bc};border-radius:8px;padding:4px 12px;font-size:0.8rem;'
                    f'color:{_bc};font-weight:600;">{_mg_label()}</span></div>')
            with gr.Row():
                gb = gr.Button("Generate", variant="primary")
                rgb_ = gr.Button("Regenerate", variant="secondary")
                eb = gr.Button("Export")
            nm = gr.Markdown("*Press Generate.*")
            ei = gr.Textbox(label="Evidence", interactive=False)
            ef = gr.File(label="Report", visible=False)
            gb.click(_gen_narrative, [st], [nm, ei, st])
            rgb_.click(_gen_narrative, [st], [nm, ei, st])
            eb.click(_export_report, [st], [ef])

        # Tab 6: Agentic Workflow
        with gr.Tab("Agentic Workflow"):
            gr.Markdown("### Agentic Clinical Workflow\nMedGemma as an autonomous agent "
                        "in a 6-stage oncology pipeline with full auditability.")
            gr.HTML(pipeline_html())
            rpb = gr.Button("Run Full Pipeline", variant="primary", size="lg")
            pp = gr.HTML('<div style="text-align:center;padding:16px;color:#475569;">'
                         'Press "Run Full Pipeline" to execute all stages.</div>')
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Audit Trail")
                    pa = gr.Dataframe(headers=["Step","Status","Confidence","Timestamp"], interactive=False)
                with gr.Column():
                    gr.Markdown("#### Agent Decision Log")
                    pd_ = gr.Markdown("*Awaiting pipeline run.*")
            gr.Markdown("#### Generated Summary")
            pn = gr.Markdown("*Pipeline output will appear here.*")
            rpb.click(_run_agentic, [st], [pp, pa, pd_, pn, st])

    return app
