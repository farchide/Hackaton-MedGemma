"""Entry point for the Digital Twin Tumor Response Assessment system.

Launch the Gradio web interface with::

    python -m digital_twin_tumor
    python -m digital_twin_tumor --port 7860 --share --debug
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="digital_twin_tumor",
        description="Digital Twin for Tumor Response Assessment -- Gradio UI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number for the Gradio server (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable Gradio link.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with verbose logging.",
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        default=False,
        help="Generate demo data before launching (creates 5 patient scenarios).",
    )
    parser.add_argument(
        "--demo-db",
        type=str,
        default=".cache/demo.db",
        help="Path to the demo database (default: .cache/demo.db).",
    )
    return parser


def _configure_logging(debug: bool) -> None:
    """Set up root logger.

    Parameters
    ----------
    debug:
        If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_app() -> "gradio.Blocks":  # noqa: F821 -- lazy import
    """Build and return the Gradio Blocks application.

    Delegates to the full five-tab implementation in
    :mod:`digital_twin_tumor.ui.app`.

    Returns
    -------
    gradio.Blocks
        The configured Gradio application.
    """
    from digital_twin_tumor.ui import create_app as _ui_create_app

    return _ui_create_app()


def main() -> None:
    """Parse arguments and launch the Gradio application."""
    import os

    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.debug)

    # Set the demo DB path for the UI to pick up
    os.environ["DTT_DEMO_DB"] = args.demo_db

    # Generate demo data if requested
    if args.generate_demo:
        if os.path.exists(args.demo_db):
            logger.info("Demo database already exists at %s", args.demo_db)
        else:
            logger.info("Generating demo data to %s ...", args.demo_db)
            from digital_twin_tumor.data.synthetic import generate_all_demo_data

            generate_all_demo_data(
                db_path=args.demo_db, seed=42, verbose=True,
            )
            logger.info("Demo data generation complete.")

    logger.info("Starting Digital Twin Tumor UI on port %d", args.port)

    app = create_app()
    launch_kw: dict = dict(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )
    # Gradio 6+ accepts theme/css/js in launch() instead of Blocks()
    import inspect as _insp
    _lp = _insp.signature(app.launch).parameters
    if hasattr(app, "_dtt_theme") and "theme" in _lp:
        launch_kw["theme"] = app._dtt_theme
    if hasattr(app, "_dtt_css") and "css" in _lp:
        launch_kw["css"] = app._dtt_css
    if hasattr(app, "_dtt_js") and "js" in _lp:
        launch_kw["js"] = app._dtt_js
    app.launch(**launch_kw)


if __name__ == "__main__":
    main()
