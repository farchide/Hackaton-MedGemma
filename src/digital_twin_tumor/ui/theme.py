"""Custom dark theme for the Digital Twin Tumor system.

A sleek black medical-grade UI with cyan/teal accent colors,
subtle gradients, and glassmorphism effects.
"""
from __future__ import annotations

from typing import Iterable

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class MedicalDarkTheme(Base):
    """Dark medical-grade theme with cyan accents and glassmorphism."""

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.cyan,
        secondary_hue: colors.Color | str = colors.teal,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_lg,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Inter"),
            fonts.GoogleFont("IBM Plex Sans"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("JetBrains Mono"),
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

        # ------------------------------------------------------------------
        # Core surface colors
        # ------------------------------------------------------------------
        super().set(
            # Body / canvas
            body_background_fill="#0a0a0f",
            body_background_fill_dark="#050508",
            body_text_color="#e2e8f0",
            body_text_color_dark="#e2e8f0",
            body_text_color_subdued="#94a3b8",
            body_text_color_subdued_dark="#94a3b8",

            # Blocks / panels
            block_background_fill="#0f1117",
            block_background_fill_dark="#0f1117",
            block_border_color="#1e293b",
            block_border_color_dark="#1e293b",
            block_border_width="1px",
            block_label_background_fill="#141722",
            block_label_background_fill_dark="#141722",
            block_label_text_color="#67e8f9",
            block_label_text_color_dark="#67e8f9",
            block_label_border_color="#1e293b",
            block_label_border_color_dark="#1e293b",
            block_shadow="0 4px 24px rgba(0, 0, 0, 0.4)",
            block_shadow_dark="0 4px 24px rgba(0, 0, 0, 0.6)",
            block_title_text_color="#f1f5f9",
            block_title_text_color_dark="#f1f5f9",

            # Inputs
            input_background_fill="#141722",
            input_background_fill_dark="#141722",
            input_border_color="#1e293b",
            input_border_color_dark="#1e293b",
            input_border_color_focus="*primary_500",
            input_border_color_focus_dark="*primary_500",
            input_placeholder_color="#475569",
            input_placeholder_color_dark="#475569",
            input_shadow="inset 0 1px 4px rgba(0, 0, 0, 0.3)",
            input_shadow_dark="inset 0 1px 4px rgba(0, 0, 0, 0.5)",
            input_shadow_focus="0 0 0 3px rgba(34, 211, 238, 0.15)",
            input_shadow_focus_dark="0 0 0 3px rgba(34, 211, 238, 0.2)",

            # Buttons — primary
            button_primary_background_fill="linear-gradient(135deg, #0891b2 0%, #06b6d4 50%, #22d3ee 100%)",
            button_primary_background_fill_dark="linear-gradient(135deg, #0891b2 0%, #06b6d4 50%, #22d3ee 100%)",
            button_primary_background_fill_hover="linear-gradient(135deg, #06b6d4 0%, #22d3ee 50%, #67e8f9 100%)",
            button_primary_background_fill_hover_dark="linear-gradient(135deg, #06b6d4 0%, #22d3ee 50%, #67e8f9 100%)",
            button_primary_text_color="#0a0a0f",
            button_primary_text_color_dark="#0a0a0f",
            button_primary_border_color="transparent",
            button_primary_border_color_dark="transparent",
            button_primary_shadow="0 2px 12px rgba(6, 182, 212, 0.3)",
            button_primary_shadow_dark="0 2px 12px rgba(6, 182, 212, 0.4)",

            # Buttons — secondary
            button_secondary_background_fill="#141722",
            button_secondary_background_fill_dark="#141722",
            button_secondary_background_fill_hover="#1e293b",
            button_secondary_background_fill_hover_dark="#1e293b",
            button_secondary_text_color="#67e8f9",
            button_secondary_text_color_dark="#67e8f9",
            button_secondary_border_color="#1e293b",
            button_secondary_border_color_dark="#1e293b",

            # Buttons — cancel / stop
            button_cancel_background_fill="#1e1e2a",
            button_cancel_background_fill_dark="#1e1e2a",
            button_cancel_text_color="#f87171",
            button_cancel_text_color_dark="#f87171",
            button_cancel_border_color="#7f1d1d",
            button_cancel_border_color_dark="#7f1d1d",

            # Borders / shadows
            border_color_accent="#22d3ee",
            border_color_accent_dark="#22d3ee",
            border_color_primary="#1e293b",
            border_color_primary_dark="#1e293b",
            shadow_spread="8px",
            shadow_spread_dark="12px",

            # Tabs
            background_fill_primary="#0f1117",
            background_fill_primary_dark="#0f1117",
            background_fill_secondary="#141722",
            background_fill_secondary_dark="#141722",

            # Links
            link_text_color="#22d3ee",
            link_text_color_dark="#22d3ee",
            link_text_color_hover="#67e8f9",
            link_text_color_hover_dark="#67e8f9",
            link_text_color_active="#a5f3fc",
            link_text_color_active_dark="#a5f3fc",
            link_text_color_visited="#06b6d4",
            link_text_color_visited_dark="#06b6d4",

            # Table
            table_border_color="#1e293b",
            table_border_color_dark="#1e293b",
            table_even_background_fill="#0f1117",
            table_even_background_fill_dark="#0f1117",
            table_odd_background_fill="#141722",
            table_odd_background_fill_dark="#141722",
            table_row_focus="#1e293b",
            table_row_focus_dark="#1e293b",

            # Slider
            slider_color="*primary_500",
            slider_color_dark="*primary_400",

            # Checkbox / radio
            checkbox_background_color="#141722",
            checkbox_background_color_dark="#141722",
            checkbox_border_color="#334155",
            checkbox_border_color_dark="#334155",
            checkbox_label_background_fill="#0f1117",
            checkbox_label_background_fill_dark="#0f1117",
            checkbox_label_text_color="#e2e8f0",
            checkbox_label_text_color_dark="#e2e8f0",

            # Loader
            loader_color="*primary_500",
            loader_color_dark="*primary_400",

            # Prose (Markdown) styling
            prose_text_size="*text_md",
            prose_header_text_weight="700",
        )


# ---------------------------------------------------------------------------
# Custom CSS overlay for effects the theme API cannot express
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Force dark mode */
:root, .dark {
    color-scheme: dark;
}

/* Animated gradient background */
.gradio-container {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 25%, #0a0f1a 50%, #0d1117 75%, #0a0a0f 100%) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 20s ease infinite !important;
    min-height: 100vh;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism panels */
.block {
    background: rgba(15, 17, 23, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(34, 211, 238, 0.08) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

.block:hover {
    border-color: rgba(34, 211, 238, 0.15) !important;
    box-shadow: 0 4px 30px rgba(6, 182, 212, 0.06) !important;
}

/* Tab styling */
.tabs > .tab-nav > button {
    color: #94a3b8 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 20px !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.02em !important;
}

.tabs > .tab-nav > button:hover {
    color: #22d3ee !important;
    background: rgba(34, 211, 238, 0.05) !important;
}

.tabs > .tab-nav > button.selected {
    color: #22d3ee !important;
    border-bottom: 2px solid #22d3ee !important;
    background: rgba(34, 211, 238, 0.08) !important;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.3) !important;
}

.tab-nav {
    border-bottom: 1px solid #1e293b !important;
    background: rgba(10, 10, 15, 0.6) !important;
    backdrop-filter: blur(8px) !important;
}

/* Primary buttons — glow effect */
.primary {
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
}

.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(6, 182, 212, 0.4), 0 0 40px rgba(6, 182, 212, 0.1) !important;
}

.primary::after {
    content: '' !important;
    position: absolute !important;
    top: -50% !important;
    left: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.08) 50%, transparent 70%) !important;
    transform: rotate(45deg) translateX(-100%) !important;
    transition: transform 0.6s ease !important;
}

.primary:hover::after {
    transform: rotate(45deg) translateX(100%) !important;
}

/* Secondary buttons */
.secondary {
    transition: all 0.3s ease !important;
}

.secondary:hover {
    border-color: rgba(34, 211, 238, 0.3) !important;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.1) !important;
}

/* Input focus glow */
input:focus, textarea:focus, select:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.15), inset 0 1px 4px rgba(0, 0, 0, 0.3) !important;
}

/* Slider track */
input[type=range] {
    accent-color: #22d3ee !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0a0a0f;
}

::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* Markdown headings */
.prose h1, .prose h2, .prose h3 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

.prose h1 {
    background: linear-gradient(135deg, #22d3ee, #06b6d4, #14b8a6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

.prose h2 {
    color: #67e8f9 !important;
}

.prose h3 {
    color: #a5f3fc !important;
}

.prose strong {
    color: #22d3ee !important;
}

.prose code {
    background: #141722 !important;
    color: #67e8f9 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid #1e293b !important;
}

.prose blockquote {
    border-left: 3px solid #22d3ee !important;
    background: rgba(34, 211, 238, 0.05) !important;
    padding: 8px 16px !important;
    border-radius: 0 8px 8px 0 !important;
}

/* Plotly chart background */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Image viewer */
.image-container {
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Dataframe */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

table th {
    background: #141722 !important;
    color: #67e8f9 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #22d3ee !important;
}

table td {
    border-bottom: 1px solid #1e293b !important;
}

table tr:hover td {
    background: rgba(34, 211, 238, 0.04) !important;
}

/* File upload area */
.upload-container {
    border: 2px dashed #1e293b !important;
    background: rgba(15, 17, 23, 0.6) !important;
    transition: all 0.3s ease !important;
}

.upload-container:hover {
    border-color: #22d3ee !important;
    background: rgba(34, 211, 238, 0.03) !important;
}

/* JSON viewer */
.json-holder {
    background: #0f1117 !important;
    border: 1px solid #1e293b !important;
}

/* Accordion */
.accordion {
    border: 1px solid #1e293b !important;
    background: rgba(15, 17, 23, 0.6) !important;
}

/* Toast notifications */
.toast-wrap .toast-body {
    background: #141722 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
}

/* Label styling */
label span {
    font-size: 0.875rem !important;
    letter-spacing: 0.02em !important;
}

/* Footer */
footer {
    background: transparent !important;
    border-top: 1px solid #1e293b !important;
}
"""


def get_theme() -> MedicalDarkTheme:
    """Return the dark medical theme instance."""
    return MedicalDarkTheme()


def get_css() -> str:
    """Return the custom CSS string."""
    return CUSTOM_CSS
