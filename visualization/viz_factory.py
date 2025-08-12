"""
Centralized helpers for creating and rendering Matplotlib visuals
with consistent quality controls based on APP_CONFIG.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Tuple
from config.app_config import APP_CONFIG


def _scaled_figsize(width: float, height: float) -> Tuple[float, float]:
    """Scale figsize if hi-res mode is enabled."""
    w, h = width, height
    if APP_CONFIG.get("hi_res_plots", False):
        w *= float(APP_CONFIG.get("hi_res_figsize_scale", 1.25))
        h *= float(APP_CONFIG.get("hi_res_figsize_scale", 1.25))
    if APP_CONFIG.get("compact_plots", False):
        w *= float(APP_CONFIG.get("compact_figsize_scale", 0.85))
        h *= float(APP_CONFIG.get("compact_figsize_scale", 0.85))
    return w, h


def create_matplotlib_figure(width: float = 8, height: float = 4, facecolor: str = "#1a1a1a"):
    """Create a Matplotlib figure honoring hi-res/compact settings and background style."""
    w, h = _scaled_figsize(width, height)
    if APP_CONFIG.get("transparent_plots", False):
        facecolor = "none"
    else:
        facecolor = APP_CONFIG.get("matplotlib_facecolor", facecolor)
    fig = plt.figure(figsize=(w, h), facecolor=facecolor)
    if APP_CONFIG.get("hi_res_plots", False):
        try:
            fig.set_dpi(int(APP_CONFIG.get("hi_res_dpi", 300)))
        except Exception:
            pass
    return fig


def finalize_matplotlib_figure(fig):
    """Apply tight layout for better spacing and return the figure."""
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def apply_plotly_theme(fig):
    """Apply global Plotly background and sizing theme based on APP_CONFIG."""
    try:
        bgcolor = APP_CONFIG.get("plotly_bgcolor", "#2a2a2a")
        layout_updates = {
            "plot_bgcolor": bgcolor,
            "paper_bgcolor": bgcolor,
        }
        if getattr(fig.layout, "height", None):
            try:
                scale = float(APP_CONFIG.get("plotly_height_scale", 1.0))
                if scale != 1.0:
                    layout_updates["height"] = int(fig.layout.height * scale)
            except Exception:
                pass
        fig.update_layout(**layout_updates)
    except Exception:
        pass
    return fig


