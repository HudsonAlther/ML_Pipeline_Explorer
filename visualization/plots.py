"""
Model evaluation plots for ML Pipeline Explorer.
"""

import numpy as np
import matplotlib.pyplot as plt
from config.app_config import APP_CONFIG
from visualization.viz_factory import create_matplotlib_figure, apply_plotly_theme
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set dark theme for all matplotlib plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

def create_confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix"):
    """Create a confusion matrix plot with dark theme"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig = create_matplotlib_figure(6, 4, facecolor='#1a1a1a')
        ax = fig.gca()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel("Predicted", color='white')
        ax.set_ylabel("Actual", color='white')
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        return None

# Confusion matrix heatmap
def plot_confusion_heatmap(cm, title="Confusion Matrix"):
    fig = create_matplotlib_figure(4, 3, facecolor='#1a1a1a')
    ax = fig.gca()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
               cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted", color='white')
    ax.set_ylabel("Actual", color='white')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    return fig

# Threshold dashboard (requires y_true, y_prob)
def plot_threshold_dashboard(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve, roc_curve
    import plotly.graph_objects as go
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    fig = go.Figure()
    green = APP_CONFIG.get('colors', {}).get('lightgreen', '#70B73F')
    blue = APP_CONFIG.get('colors', {}).get('primary_blue', '#005E7B')
    fig.add_trace(go.Scatter(x=thr, y=prec[:-1], mode="lines", name="Precision", line=dict(color=green)))
    fig.add_trace(go.Scatter(x=thr, y=rec[:-1], mode="lines", name="Recall", line=dict(color=blue)))
    # Theme-aware font color
    import streamlit as st
    base = str(st.get_option("theme.base") or "dark").lower()
    font_color = '#FFFFFF' if base == 'dark' else '#000000'
    fig.update_layout(
        title="Precision-Recall vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=280,
        font=dict(color=font_color),
        title_font_color=font_color,
    )
    return apply_plotly_theme(fig)
