"""UI sub-package.

Gradio-based web interface for the Digital Twin Tumor system.

Quick usage::

    from digital_twin_tumor.ui import create_app, launch

    app = create_app()
    app.launch(server_port=7860)

Or use the convenience launcher::

    from digital_twin_tumor.ui import launch
    launch(port=7860, share=False)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr

logger = logging.getLogger(__name__)


def create_app() -> "gr.Blocks":
    """Build and return the Gradio Blocks application.

    Lazy-imports the app module so that ``import digital_twin_tumor.ui``
    does not require Gradio to be installed until the app is actually
    created.

    Returns
    -------
    gr.Blocks
        The configured seven-tab Gradio application.
    """
    from digital_twin_tumor.ui.app import create_app as _create_app

    return _create_app()


def launch(
    port: int = 7860,
    share: bool = False,
    debug: bool = False,
) -> None:
    """Build the app and launch the Gradio server.

    Parameters
    ----------
    port:
        Port number for the Gradio server.
    share:
        If ``True``, create a publicly shareable Gradio link.
    debug:
        If ``True``, enable Gradio debug mode.
    """
    app = create_app()
    logger.info("Launching Digital Twin Tumor UI on port %d", port)
    launch_kw: dict = dict(server_port=port, share=share, debug=debug)
    # Gradio 6+ accepts theme/css/js in launch() instead of Blocks()
    if hasattr(app, "_dtt_theme"):
        import inspect as _insp
        _lp = _insp.signature(app.launch).parameters
        if "theme" in _lp:
            launch_kw["theme"] = app._dtt_theme  # type: ignore[attr-defined]
        if "css" in _lp:
            launch_kw["css"] = app._dtt_css  # type: ignore[attr-defined]
        if "js" in _lp:
            launch_kw["js"] = app._dtt_js  # type: ignore[attr-defined]
    app.launch(**launch_kw)


__all__ = ["create_app", "launch"]
