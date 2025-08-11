"""
Reusable UI components for ML Pipeline Explorer.
"""

# Metric card component
import streamlit as st
from typing import Any, Optional

def metric_card(label: str, value: Any, delta: Optional[str] = None, helptext: Optional[str] = None):
    """Display a stylized metric card using Streamlit.

    This wrapper ensures a consistent look & feel for all summary metrics across
    the application. Additional styling tweaks can be centralized here so that
    they automatically propagate everywhere without touching individual views.

    Parameters
    ----------
    label : str
        The primary label for the metric (e.g., "Accuracy").
    value : Any
        The value to display. Streamlit will convert this to string internally.
    delta : Optional[str], default None
        An optional delta value to show alongside the metric (e.g., "+3.2%").
    helptext : Optional[str], default None
        Optional tooltip/hover help text.
    """

    # Apply a consistent container width and formatting.
    # Future styling (icons, colors) can be added here.
    st.metric(label=label, value=value, delta=delta, help=helptext)

# Add more reusable UI/data components as needed
