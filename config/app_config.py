"""
Application-wide configuration settings.
"""

from pathlib import Path

APP_CONFIG = {
    "page_title": "ML Pipeline Explorer",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "artifact_dir": Path("artifacts"),
    "default_test_size": 0.2,
    "default_random_state": 42,
    "max_shap_samples": 1000,
    "default_shap_samples": 500,
    "max_ice_samples": 30,
    "max_feature_importance": 40,
    "max_input_features": 20,  # Maximum number of features shown in input forms
    "model_selection_sleep": 10,  # Seconds to wait for UI feedback (increased to 10)
    # High-resolution rendering controls
    "hi_res_plots": False,          # Toggle for high-DPI Matplotlib figures
    "hi_res_dpi": 400,              # DPI to use when hi_res_plots is True
    "hi_res_figsize_scale": 1.1,   # Scale existing figsize by this factor for hi-res
    # Compact and background styling
    "compact_plots": True,          # Reduce default figure sizes
    "compact_figsize_scale": 0.7,   # Scale down figsize when compact_plots is True
    "transparent_plots": False,     # Use non-transparent backgrounds
    "matplotlib_facecolor": "#2a2a2a",  # Grey background for Matplotlib figures
    "plotly_bgcolor": "#2a2a2a",        # Grey background for Plotly figures
    "plotly_height_scale": 0.8,     # Scale factor for Plotly figure heights
    # Approved color palette
    "colors": {
        "mediumblue": "#005E7B",
        "darkred": "#D0073A",
        "lightblue": "#008CA5",
        "gray": "#777777",
        "yellow": "#FBC15E",
        "darkgreen": "#307F42",
        "pink": "#FFB5B8",
        "darkblue": "#063157",
        "brightred": "#EA0D49",
        "brown": "#603534",
        "lightgreen": "#70B73F",
        "orange": "#F7941D",
        # Aliases
        "primary_blue": "#005E7B",   # mediumblue
        "accent_blue": "#008CA5",    # lightblue
        "primary_grey": "#777777",
    },
}
