"""Reusable UI components for the Digital Twin Tumor system."""
from __future__ import annotations
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False
    go = None  # type: ignore[assignment]
try:
    import pandas as pd  # type: ignore[import-untyped]
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False
    pd = None  # type: ignore[assignment]
try:
    from PIL import Image, ImageDraw  # type: ignore[import-untyped]
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

# -- Theme constants --
_BG = "rgba(10,14,26,0.95)"
_CARD = "#141722"
_CYAN = "#22d3ee"
_TEAL = "#06b6d4"
_LCYAN = "#67e8f9"
_TEXT = "#e2e8f0"
_GRID = "rgba(30,41,59,0.4)"
_ZERO = "#1e293b"
_EMER = "#10b981"
_AMBER = "#f59e0b"
_ROSE = "#ef4444"
_VIOLET = "#a855f7"
_FUCH = "#ec4899"
_SC_COLORS = [_AMBER, _ROSE, _EMER, _VIOLET, _FUCH, "#f97316"]

_DARK_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(10,14,26,0)", plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, system-ui, sans-serif", size=12),
    title_font=dict(color=_LCYAN, size=16, family="Inter, system-ui, sans-serif"),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_ZERO, linecolor=_ZERO),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_ZERO, linecolor=_ZERO),
    legend=dict(bgcolor="rgba(10,14,26,0.7)", bordercolor=_ZERO, borderwidth=1, font=dict(size=11, color=_TEXT)),
    margin=dict(l=60, r=30, t=65, b=50),
    hoverlabel=dict(bgcolor=_CARD, bordercolor=_TEAL, font_color=_TEXT, font_size=12),
)

_DISCLAIMER_HTML = (
    '<div style="background:linear-gradient(135deg,rgba(234,179,8,0.08),rgba(234,179,8,0.03));'
    "border:1px solid rgba(234,179,8,0.25);border-left:3px solid #eab308;"
    "border-radius:8px;padding:14px 20px;margin-bottom:16px;"
    'font-size:13px;color:#fbbf24;backdrop-filter:blur(8px);">'
    '<span style="color:#fde68a;font-weight:700;letter-spacing:0.05em;">'
    "DISCLAIMER</span>&nbsp;&nbsp;This tool is for <strong>research and "
    "educational purposes only</strong>. It is <strong>not</strong> a "
    "medical device and must <strong>not</strong> be used for clinical "
    "decision-making. All outputs require independent verification by a "
    "qualified healthcare professional.</div>"
)

_NAMED_RGB: dict[str, str] = {
    _AMBER: "245,158,11", _ROSE: "239,68,68", _EMER: "16,185,129",
    _VIOLET: "168,85,247", _FUCH: "236,72,153", "#f97316": "249,115,22",
    _CYAN: "34,211,238", _TEAL: "6,182,212",
}


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}" if len(h) == 6 else "128,128,128"


def _named_to_rgb(name: str) -> str:
    return _NAMED_RGB.get(name, _hex_rgb(name))


def _nl() -> dict:
    return {k: v for k, v in _DARK_LAYOUT.items() if k != "legend"}


def create_disclaimer_banner() -> str:
    return _DISCLAIMER_HTML


# -- Image helpers --

def extract_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """Extract axial slice *index* from 3-D *volume* as uint8 (H,W)."""
    if volume is None or volume.ndim < 3 or volume.shape[0] == 0:
        return np.zeros((256, 256), dtype=np.uint8)
    idx = max(0, min(index, volume.shape[0] - 1))
    slc = volume[idx].astype(np.float64)
    lo, hi = float(slc.min()), float(slc.max())
    slc = (slc - lo) / (hi - lo) * 255.0 if hi - lo > 1e-8 else np.zeros_like(slc)
    return slc.astype(np.uint8)


def create_measurement_overlay(
    image: np.ndarray, mask: np.ndarray | None = None,
    diameter_endpoints: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray:
    """Overlay segmentation mask and diameter line on a 2-D image."""
    if image is None or image.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1).astype(np.uint8)
    elif image.ndim == 3 and image.shape[2] == 1:
        rgb = np.repeat(image, 3, axis=2).astype(np.uint8)
    else:
        rgb = image.astype(np.uint8).copy()
    if mask is not None and mask.shape[:2] == rgb.shape[:2]:
        alpha, oc = 0.35, np.array([0, 200, 0], dtype=np.uint8)
        mb = mask.astype(bool)
        for c in range(3):
            ch = rgb[:, :, c].astype(np.float64)
            ch[mb] = ch[mb] * (1 - alpha) + oc[c] * alpha
            rgb[:, :, c] = ch.astype(np.uint8)
    if diameter_endpoints is not None and _HAS_PIL:
        try:
            pil_img = Image.fromarray(rgb, "RGB")
            draw = ImageDraw.Draw(pil_img)
            p1, p2 = diameter_endpoints
            draw.line([p1, p2], fill=(255, 50, 50), width=2)
            for p in (p1, p2):
                draw.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(255, 50, 50))
            rgb = np.array(pil_img)
        except Exception:
            logger.debug("PIL diameter draw failed", exc_info=True)
    return rgb


# =========================================================================
# 1. GROWTH PLOT -- hero visualization
# =========================================================================

def create_growth_plot(
    times: np.ndarray | None = None, volumes: np.ndarray | None = None,
    fitted: np.ndarray | None = None, lower: np.ndarray | None = None,
    upper: np.ndarray | None = None, scenarios: list[dict[str, Any]] | None = None,
    therapy_start_week: float | None = None,
) -> Any:
    """Observed points, fitted curve with glow, CI bands, breakpoints, scenarios."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    t = np.asarray(times) if times is not None else np.array([])
    ok = len(t) > 0
    ht = "Week %{x:.1f}<br>Volume: %{y:.0f} mm\u00b3<extra></extra>"
    # -- 95% CI band --
    if ok and lower is not None and upper is not None:
        lo, hi = np.asarray(lower), np.asarray(upper)
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]), y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=f"rgba({_named_to_rgb(_CYAN)},0.10)",
            line=dict(width=0), name="95% CI", hoverinfo="skip"))
    # -- Fitted curve (glow + sharp) --
    if ok and fitted is not None:
        f = np.asarray(fitted)
        fig.add_trace(go.Scatter(x=t, y=f, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=f"rgba({_named_to_rgb(_TEAL)},0.25)", width=8)))
        fig.add_trace(go.Scatter(x=t, y=f, mode="lines", name="Fitted Model",
                                 line=dict(color=_CYAN, width=2.5, shape="spline"), hovertemplate=ht))
    # -- Observed markers --
    if ok and volumes is not None:
        fig.add_trace(go.Scatter(
            x=t, y=np.asarray(volumes), mode="markers", name="Observed",
            marker=dict(color=_CYAN, size=10, line=dict(width=2, color="#ffffff")),
            hovertemplate=ht))
    # -- Breakpoints (slope change > 30%) --
    if ok and volumes is not None and len(t) >= 3:
        v = np.asarray(volumes)
        dt, dv = np.diff(t), np.diff(v)
        rates = np.where(dt > 0, dv / dt, 0.0)
        first_bp = True
        for i in range(1, len(rates)):
            if abs(rates[i]) > 0 and abs(rates[i] - rates[i-1]) / (abs(rates[i-1]) + 1e-9) > 0.3:
                fig.add_trace(go.Scatter(
                    x=[t[i+1]], y=[v[i+1]], mode="markers",
                    marker=dict(color=_AMBER, size=14, symbol="diamond-open", line=dict(width=2, color=_AMBER)),
                    name="Slope Change" if first_bp else None, showlegend=first_bp,
                    hovertemplate="Breakpoint Wk %{x:.1f}<br>Vol: %{y:.0f}<extra></extra>"))
                first_bp = False
    # -- Therapy start --
    if therapy_start_week is not None:
        fig.add_vline(x=therapy_start_week, line_dash="dash", line_color=_VIOLET, line_width=1.5,
                      annotation_text="Tx Start", annotation_font_color=_VIOLET, annotation_font_size=11)
    # -- Scenarios --
    for i, sc in enumerate(scenarios or []):
        if isinstance(sc, dict):
            st, sv = np.asarray(sc.get("times", [])), np.asarray(sc.get("volumes", []))
            sn, sl, su = sc.get("name", f"Scenario {i+1}"), sc.get("lower"), sc.get("upper")
        else:
            st = np.asarray(getattr(sc, "time_points", []))
            sv = np.asarray(getattr(sc, "predicted_volumes", []))
            sn = getattr(sc, "scenario_name", f"Scenario {i+1}")
            sl, su = getattr(sc, "lower_bound", None), getattr(sc, "upper_bound", None)
        c = _SC_COLORS[i % len(_SC_COLORS)]
        if len(st) > 0 and len(sv) > 0:
            fig.add_trace(go.Scatter(
                x=st, y=sv, mode="lines", name=sn, line=dict(color=c, width=2, dash="dash"),
                hovertemplate=f"{sn}<br>Wk %{{x:.1f}}<br>Vol: %{{y:.0f}}<extra></extra>"))
            if sl is not None and su is not None and len(np.asarray(sl)) > 0:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([st, st[::-1]]),
                    y=np.concatenate([np.asarray(su), np.asarray(sl)[::-1]]),
                    fill="toself", fillcolor=f"rgba({_named_to_rgb(c)},0.07)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
    _base = {k: v for k, v in _nl().items() if k != "margin"}
    fig.update_layout(
        **_base, title=dict(text="Tumor Growth Trajectory \u2014 Digital Twin Projection",
                            x=0.5, y=0.97, yanchor="top"),
        xaxis_title="Time (weeks)", yaxis_title="Volume (mm\u00b3)", height=480,
        margin=dict(l=60, r=30, t=90, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                    bgcolor="rgba(10,14,26,0.7)", bordercolor=_ZERO, borderwidth=1, font=dict(size=11)))
    return fig


# =========================================================================
# 2. RECIST WATERFALL
# =========================================================================

def create_recist_waterfall(responses: list[dict[str, Any]] | None = None) -> Any:
    """Waterfall bar chart of RECIST percent-change from baseline."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    if not responses:
        fig.add_annotation(text="No RECIST data", xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=_TEXT, size=14))
        fig.update_layout(**_DARK_LAYOUT, height=320, title="RECIST Response Assessment")
        return fig
    _cc = {"CR": _EMER, "PR": _CYAN, "SD": _AMBER, "PD": _ROSE}

    def _a(r: Any, k: str, d: Any = "") -> Any:
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    labels = [str(_a(r, "timepoint_id", "?"))[:10] for r in responses]
    vals = [float(_a(r, "percent_change_from_baseline", 0)) for r in responses]
    cats = [str(_a(r, "category", "SD")) for r in responses]
    cols = [_cc.get(c, "#94a3b8") for c in cats]
    fig.add_trace(go.Bar(
        x=labels, y=vals,
        marker=dict(color=cols, line=dict(width=1.5, color=[f"rgba({_named_to_rgb(c)},0.6)" for c in cols])),
        text=[f"<b>{c}</b><br>{v:+.1f}%" for c, v in zip(cats, vals)],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="Timepoint: %{x}<br>Change: %{y:+.1f}%<extra></extra>"))
    fig.add_hline(y=-30, line_dash="dot", line_color=_EMER, line_width=1.5,
                  annotation_text="PR threshold (-30%)", annotation_font_color=_EMER, annotation_font_size=10)
    fig.add_hline(y=20, line_dash="dot", line_color=_ROSE, line_width=1.5,
                  annotation_text="PD threshold (+20%)", annotation_font_color=_ROSE, annotation_font_size=10)
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="RECIST Response Assessment", x=0.5),
                      height=360, xaxis_title="Timepoint", yaxis_title="Change from Baseline (%)")
    return fig


# =========================================================================
# 3. IDENTITY GRAPH
# =========================================================================

def create_identity_graph_plot(graph_data: dict[str, Any] | None = None) -> Any:
    """Plotly DAG of the lesion-tracking identity graph."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    if graph_data is None or not graph_data.get("nodes"):
        fig.add_annotation(text="No tracking data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=_TEXT))
        fig.update_layout(**_DARK_LAYOUT, height=360, title="Lesion Identity Tracking Graph")
        return fig
    nodes, edges = graph_data["nodes"], graph_data.get("edges", [])
    tp_order: list[str] = []
    lid_order: list[str] = []
    for n in nodes:
        tp = n.get("timepoint_id", "")
        if tp and tp not in tp_order:
            tp_order.append(tp)
        lid = n.get("lesion_id", "")
        can = lid.rsplit("_tp", 1)[0] if "_tp" in lid else lid
        if can and can not in lid_order:
            lid_order.append(can)
    pos: dict[str, tuple[float, float]] = {}
    for n in nodes:
        lid = n.get("lesion_id", "")
        can = lid.rsplit("_tp", 1)[0] if "_tp" in lid else lid
        tp = n.get("timepoint_id", "")
        pos[n.get("id", "")] = (
            float(tp_order.index(tp)) if tp in tp_order else 0.0,
            float(lid_order.index(can)) if can in lid_order else 0.0)
    # Edges colored by confidence
    for e in edges:
        s, t = e.get("source", ""), e.get("target", "")
        conf = float(e.get("confidence", 0.5))
        if s in pos and t in pos:
            ec = _CYAN if conf >= 0.8 else _AMBER if conf >= 0.5 else _ROSE
            fig.add_trace(go.Scatter(
                x=[pos[s][0], pos[t][0]], y=[pos[s][1], pos[t][1]], mode="lines",
                line=dict(width=max(1.0, conf*3), color=ec),
                hoverinfo="none", showlegend=False, opacity=0.7))
    # Nodes sized by volume
    valid = [n for n in nodes if n.get("id") in pos]
    vols = [float(n.get("volume", 100)) for n in valid]
    mx = max(vols) if vols else 1.0
    szs = [max(8, min(28, 8 + 20*(v/mx))) for v in vols]
    fig.add_trace(go.Scatter(
        x=[pos[n["id"]][0] for n in valid], y=[pos[n["id"]][1] for n in valid],
        mode="markers+text",
        marker=dict(size=szs, color=_CYAN, line=dict(width=1.5, color="#fff"), opacity=0.9),
        text=[n.get("lesion_id", "?")[:6] for n in valid],
        textposition="top center", textfont=dict(size=9, color=_TEXT),
        hovertext=[f"Lesion: {n.get('lesion_id','?')[:10]}<br>Vol: {n.get('volume',0):.0f} mm\u00b3<br>Organ: {n.get('organ','?')}" for n in valid],
        hoverinfo="text", showlegend=False))
    lk = {k: v for k, v in _DARK_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(**lk, title=dict(text="Lesion Identity Tracking Graph", x=0.5), height=380,
        xaxis=dict(title="Timepoint", tickvals=list(range(len(tp_order))),
                   ticktext=[t[:8] for t in tp_order], gridcolor=_GRID, zerolinecolor=_ZERO, linecolor=_ZERO),
        yaxis=dict(title="Lesion", tickvals=list(range(len(lid_order))),
                   ticktext=[l[:10] for l in lid_order], gridcolor=_GRID, zerolinecolor=_ZERO, linecolor=_ZERO))
    return fig


# =========================================================================
# 4. HETEROGENEITY PLOT (NEW)
# =========================================================================

def create_heterogeneity_plot(
    measurements: list[dict[str, Any]] | None = None,
    timepoints: list[dict[str, Any]] | None = None,
) -> Any:
    """Multiple lesion trajectories showing divergent/mixed response."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    if not measurements or not timepoints:
        fig.add_annotation(text="No multi-lesion data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=_TEXT))
        fig.update_layout(**_DARK_LAYOUT, height=380, title="Lesion Heterogeneity")
        return fig
    tp_wk = {tp.get("timepoint_id", ""): float(tp.get("week", 0)) for tp in timepoints}
    ld: dict[str, list[tuple[float, float]]] = {}
    for m in measurements:
        lid = m.get("lesion_id", "")
        can = lid.rsplit("_tp", 1)[0] if "_tp" in lid else lid
        wk = tp_wk.get(m.get("timepoint_id", ""), float(m.get("week", 0)))
        ld.setdefault(can, []).append((wk, float(m.get("volume_mm3", 0))))
    pal = [_CYAN, _AMBER, _EMER, _ROSE, _VIOLET, _FUCH, "#f97316", "#60a5fa"]
    for i, (lid, pts) in enumerate(sorted(ld.items())):
        pts.sort()
        ws, vs = [p[0] for p in pts], [p[1] for p in pts]
        c = pal[i % len(pal)]
        tr = "progressing" if len(vs) >= 2 and vs[-1] > vs[0] else "responding"
        fig.add_trace(go.Scatter(
            x=ws, y=vs, mode="lines+markers", name=f"{lid[:10]} ({tr})",
            line=dict(color=c, width=2, dash="solid" if tr == "progressing" else "dot"),
            marker=dict(size=6, color=c, line=dict(width=1, color="#fff")),
            hovertemplate=f"{lid[:10]}<br>Wk %{{x:.0f}}: %{{y:.0f}} mm\u00b3<extra></extra>"))
    fig.update_layout(
        **_nl(), title=dict(text="Lesion Heterogeneity \u2014 Mixed Response Detection", x=0.5),
        xaxis_title="Time (weeks)", yaxis_title="Volume (mm\u00b3)", height=400,
        legend=dict(bgcolor="rgba(10,14,26,0.7)", bordercolor=_ZERO, borderwidth=1, font=dict(size=10)))
    return fig


# =========================================================================
# 5. THERAPY TIMELINE (NEW)
# =========================================================================

def create_therapy_timeline(
    therapy_events: list[dict[str, Any]] | None = None,
    recist_responses: list[dict[str, Any]] | None = None,
    timepoints: list[dict[str, Any]] | None = None,
) -> Any:
    """Horizontal timeline of treatments with RECIST status markers."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure()
    if not therapy_events and not timepoints:
        fig.add_annotation(text="No therapy data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=_TEXT))
        fig.update_layout(**_DARK_LAYOUT, height=250, title="Therapy Timeline")
        return fig
    _cc = {"CR": _EMER, "PR": _CYAN, "SD": _AMBER, "PD": _ROSE}
    tp_pal = [_VIOLET, _TEAL, _FUCH, "#f97316", "#60a5fa"]
    for i, te in enumerate(therapy_events or []):
        s_wk, e_wk = float(i * 8), float(i * 8 + 12)
        sd, ed = te.get("start_date", ""), te.get("end_date", "")
        if timepoints and sd:
            try:
                from datetime import date as _d
                t0 = timepoints[0].get("scan_date", "")
                if t0:
                    d0 = _d.fromisoformat(str(t0))
                    ds = _d.fromisoformat(str(sd))
                    de = _d.fromisoformat(str(ed)) if ed else ds
                    s_wk, e_wk = (ds - d0).days / 7.0, (de - d0).days / 7.0
            except (ValueError, TypeError):
                pass
        c = tp_pal[i % len(tp_pal)]
        lbl = te.get("drug_name", te.get("therapy_type", f"Therapy {i+1}"))
        fig.add_trace(go.Bar(
            x=[e_wk - s_wk], y=[lbl], base=[s_wk], orientation="h", name=lbl,
            marker=dict(color=f"rgba({_named_to_rgb(c)},0.5)", line=dict(width=1, color=c)),
            hovertemplate=f"{lbl}<br>Wk {s_wk:.0f}\u2013{e_wk:.0f}<extra></extra>", showlegend=False))
    if timepoints:
        r_map = {r.get("timepoint_id", ""): r.get("category", "SD") for r in (recist_responses or [])}
        for tp in timepoints:
            wk = float(tp.get("week", 0))
            cat = r_map.get(tp.get("timepoint_id", ""), "")
            col = _cc.get(cat, "#94a3b8")
            fig.add_trace(go.Scatter(
                x=[wk], y=["RECIST"], mode="markers+text",
                marker=dict(size=16, color=col, symbol="diamond", line=dict(width=1, color="#fff")),
                text=[cat], textposition="top center", textfont=dict(size=9, color=col),
                showlegend=False, hovertemplate=f"Wk {wk:.0f}: {cat}<extra></extra>"))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Treatment Timeline & RECIST Status", x=0.5),
                      height=max(200, 80 + 40 * len(therapy_events or [])),
                      xaxis_title="Time (weeks)", yaxis_title="", barmode="stack", showlegend=False)
    return fig


# =========================================================================
# 6. UNCERTAINTY DECOMPOSITION (NEW)
# =========================================================================

def create_uncertainty_decomposition(
    measurement_uncertainty: float = 0.3, model_uncertainty: float = 0.4,
    scenario_uncertainty: float = 0.3,
) -> Any:
    """Stacked bar showing sources of uncertainty."""
    if not _HAS_PLOTLY:
        return None
    total = measurement_uncertainty + model_uncertainty + scenario_uncertainty
    total = max(total, 1e-9)
    items = [
        ("Measurement Noise", measurement_uncertainty / total * 100, _AMBER),
        ("Model Epistemic", model_uncertainty / total * 100, _CYAN),
        ("Scenario Spread", scenario_uncertainty / total * 100, _VIOLET),
    ]
    fig = go.Figure()
    for lbl, pct, c in items:
        fig.add_trace(go.Bar(
            x=[pct], y=["Uncertainty"], orientation="h", name=lbl,
            marker=dict(color=f"rgba({_named_to_rgb(c)},0.7)", line=dict(width=1, color=c)),
            text=[f"{lbl}: {pct:.0f}%"], textposition="inside", textfont=dict(size=11, color="#fff"),
            hovertemplate=f"{lbl}: {pct:.1f}%<extra></extra>"))
    base = {k: v for k, v in _DARK_LAYOUT.items() if k not in ("legend", "xaxis", "yaxis")}
    fig.update_layout(
        **base, title=dict(text="Uncertainty Decomposition", x=0.5),
        height=160, barmode="stack", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5,
                    bgcolor="rgba(10,14,26,0.7)", bordercolor=_ZERO, borderwidth=1, font=dict(size=10)),
        xaxis=dict(title="Contribution (%)", range=[0, 100], gridcolor=_GRID,
                   zerolinecolor=_ZERO, linecolor=_ZERO),
        yaxis=dict(visible=False))
    return fig


# =========================================================================
# DATA TABLE
# =========================================================================

def format_measurement_table(measurements: list[dict[str, Any]] | None = None) -> Any:
    """Format measurements as a DataFrame (or nested list fallback)."""
    hdrs = ["Lesion ID", "Diameter (mm)", "Volume (mm\u00b3)", "Method", "Timepoint"]
    if not measurements:
        return pd.DataFrame(columns=hdrs) if _HAS_PANDAS else [hdrs]
    _badge = {"auto": "Auto", "semi-auto": "Semi-auto", "manual": "Manual"}
    def _attr(m: Any, key: str, default: Any = "") -> Any:
        return m.get(key, default) if isinstance(m, dict) else getattr(m, key, default)
    rows = [
        [str(_attr(m, "lesion_id", ""))[:8], f"{float(_attr(m, 'diameter_mm', 0)):.1f}",
         f"{float(_attr(m, 'volume_mm3', 0)):.1f}",
         _badge.get(str(_attr(m, "method", "auto")).lower(), str(_attr(m, "method", "")).capitalize()),
         str(_attr(m, "timepoint_id", ""))[:10]]
        for m in measurements]
    return pd.DataFrame(rows, columns=hdrs) if _HAS_PANDAS else [hdrs] + rows
