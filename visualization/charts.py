"""
Chart creation utilities for ML Pipeline Explorer.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.app_config import APP_CONFIG
from visualization.viz_factory import apply_plotly_theme

# Dark theme configuration for all charts
DARK_THEME = {
    'plot_bgcolor': '#1a1a1a',
    'paper_bgcolor': '#1a1a1a', 
    'font': {'color': 'white'},
    'title_font_color': 'white'
}

def create_model_comparison_chart(df, ds_key):
    """Create a comprehensive model comparison chart"""
    if df is None or df.empty:
        return None
    
    # Select metrics to display
    metrics = ["accuracy", "roc_auc", "f1_score", "precision", "recall"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return None
    
    fig = go.Figure()
    
    for metric in available_metrics:
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=df["model_key"], 
            y=df[metric],
            hovertemplate=f"{metric.title()}: %{{y:.4f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"Model Comparison — {ds_key.title()}",
        xaxis_title="Model", 
        yaxis_title="Score",
        barmode="group", 
        height=420, 
        legend_title="Metric",
        showlegend=True,
        **DARK_THEME
    )
    # Theme-aware bar color: light theme -> accent blue; dark -> primary blue
    import streamlit as st
    base = str(st.get_option("theme.base") or "dark").lower()
    bar_color = APP_CONFIG.get('colors', {}).get('primary_blue', '#005E7B')
    if base == 'light':
        bar_color = APP_CONFIG.get('colors', {}).get('lightblue', '#008CA5')
    fig.update_traces(marker_color=bar_color)
    fig = apply_plotly_theme(fig)
    
    return fig

# Model comparison bar chart
def create_simple_model_comparison(df, ds_key):
    if df is None or df.empty:
        return None
    metrics = ["accuracy", "roc_auc", "f1_score"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric.replace('_',' ').title(),
            x=df["model_key"], y=df[metric],
            hovertemplate=f"{metric.title()}: %{{y:.4f}}<extra></extra>"
        ))
    fig.update_layout(
        title=f"Model Comparison — {ds_key.title()}",
        xaxis_title="Model", yaxis_title="Score",
        barmode="group", height=420, legend_title="Metric",
        **DARK_THEME
    )
    import streamlit as st
    base = str(st.get_option("theme.base") or "dark").lower()
    bar_color = APP_CONFIG.get('colors', {}).get('primary_blue', '#005E7B')
    if base == 'light':
        bar_color = APP_CONFIG.get('colors', {}).get('lightblue', '#008CA5')
    fig.update_traces(marker_color=bar_color)
    fig = apply_plotly_theme(fig)
    return fig

# Feature importance chart
def create_feature_importance_plot(model, feature_names, model_name):
    try:
        model_obj = model.named_steps["model"] if hasattr(model, "named_steps") else model
        if hasattr(model_obj, "feature_importances_"):
            imp = model_obj.feature_importances_
        elif hasattr(model_obj, "coef_"):
            coef = model_obj.coef_
            imp = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return None
        df_imp = pd.DataFrame({"feature": feature_names[:len(imp)], "importance": imp})
        df_imp = df_imp.sort_values("importance", ascending=True).tail(40)
        fig = px.bar(
            df_imp, x="importance", y="feature", orientation="h",
            title=f"Feature Importance — {model_name}",
            labels={"importance": "Importance", "feature": "Feature"},
            height=520,
            color_discrete_sequence=['#2196F3']
        )
        fig.update_layout(**{**DARK_THEME, 'plot_bgcolor': APP_CONFIG.get('plotly_bgcolor', '#1a1a1a'), 'paper_bgcolor': APP_CONFIG.get('plotly_bgcolor', '#1a1a1a')})
        return fig
    except Exception as e:
        return None

# PDP/ICE chart
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

def create_pdp_ice(model, X, feature, max_ice=30):
    try:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1a1a1a')
        display = PartialDependenceDisplay.from_estimator(
            model, X, [feature], kind=["average", "individual"],
            n_jobs=1, ax=ax, ice_lines_kw={"alpha": 0.2}, pd_line_kw={"color": "#4CAF50", "lw": 2}
        )
        ax.set_title(f"PDP/ICE for {feature}", color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel(ax.get_xlabel(), color='white')
        ax.set_ylabel(ax.get_ylabel(), color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a1a')
        return fig
    except Exception as e:
        return None

# SHAP summary plot
import shap
import numpy as np
from visualization.viz_factory import create_matplotlib_figure, finalize_matplotlib_figure

def _prepare_for_shap(model, X_sample):
    """Return (core_model, processed_df) ready for SHAP."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preproc = model.named_steps["preprocessor"]
        core_model = model.named_steps.get("model", model)
        try:
            # Ensure X_sample is a DataFrame for consistent handling
            if not isinstance(X_sample, pd.DataFrame):
                X_sample = pd.DataFrame(X_sample)
            
            # Apply preprocessing transformation
            X_proc = preproc.transform(X_sample)
            
            # Handle feature names more robustly
            try:
                if hasattr(preproc, "get_feature_names_out"):
                    feature_names = preproc.get_feature_names_out()
                elif hasattr(preproc, "get_feature_names"):
                    feature_names = preproc.get_feature_names()
                else:
                    # For ColumnTransformer, manually build feature names
                    feature_names = []
                    if hasattr(preproc, 'transformers_'):
                        for name, transformer, columns in preproc.transformers_:
                            if name == 'remainder':
                                continue
                            if hasattr(transformer, 'get_feature_names_out'):
                                trans_names = transformer.get_feature_names_out(columns)
                                feature_names.extend(trans_names)
                            elif len(columns) > 0:
                                # Fallback for transformers without get_feature_names_out
                                if name == 'num':  # numeric features keep original names
                                    feature_names.extend(columns)
                                elif name == 'cat':  # categorical features get expanded
                                    # Estimate expanded feature count (rough approximation)
                                    estimated_count = len(columns) * 3  # rough estimate for one-hot
                                    feature_names.extend([f"{name}_feature_{i}" for i in range(estimated_count)])
                    
                    # If still no feature names, use generic ones
                    if not feature_names or len(feature_names) != X_proc.shape[1]:
                        feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]
                        
            except Exception as e:
                print(f"Feature name extraction failed: {e}")
                feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]
            
            # Ensure we have the right number of feature names
            if len(feature_names) != X_proc.shape[1]:
                print(f"Feature name mismatch: {len(feature_names)} names vs {X_proc.shape[1]} features")
                feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]
            
            # Convert to DataFrame with proper feature names
            X_df = pd.DataFrame(X_proc, columns=feature_names)
            
            # Ensure no NaN or infinite values that could cause SHAP issues
            X_df = X_df.fillna(0)
            X_df = X_df.replace([np.inf, -np.inf], 0)
            
            return core_model, X_df
            
        except Exception as e:
            print(f"SHAP preprocessing failed: {e}")
            # Fallback to raw if transform fails
            return model, X_sample
    # Already a core estimator
    return model, X_sample


def shap_summary_plot(model, X_sample, max_display: int | None = None):
    def _force_white_text(fig):
        try:
            # Set white on all axes (including colorbar axes)
            for _ax in fig.axes:
                try:
                    _ax.tick_params(colors='white')
                    if hasattr(_ax, 'xaxis'):
                        _ax.xaxis.label.set_color('white')
                    if hasattr(_ax, 'yaxis'):
                        _ax.yaxis.label.set_color('white')
                    if hasattr(_ax, 'title'):
                        _ax.title.set_color('white')
                except Exception:
                    pass
            # Set all text artists to white (covers colorbar labels and misc text)
            for txt in fig.findobj(lambda o: isinstance(o, mpl.text.Text)):
                try:
                    txt.set_color('white')
                except Exception:
                    pass
        except Exception:
            pass
    try:
        core_model, X_proc = _prepare_for_shap(model, X_sample)
        
        # More robust explainer selection
        if hasattr(core_model, "estimators_"):  # Random Forest
            explainer = shap.TreeExplainer(core_model)
        elif core_model.__class__.__name__.lower().startswith("xgb"):  # XGBoost
            explainer = shap.TreeExplainer(core_model)
        elif hasattr(core_model, "coef_"):  # Linear models
            explainer = shap.Explainer(core_model, X_proc)
        else:
            # Try TreeExplainer first, fallback to Explainer
            try:
                explainer = shap.TreeExplainer(core_model)
            except:
                explainer = shap.Explainer(core_model, X_proc)
        
        shap_values = explainer(X_proc)
        
        # Special case for Random Forest - use beeswarm plot instead of summary plot
        if hasattr(core_model, "estimators_"):  # Random Forest
            # Handle multi-class SHAP values for beeswarm plot
            if len(shap_values.shape) == 3:  # (samples, features, classes)
                # For binary classification, use the positive class (index 1)
                if shap_values.shape[2] == 2:  # Binary classification
                    shap_values_single = shap_values[:, :, 1]  # Use positive class for all samples
                else:
                    # For multi-class, sum the absolute values
                    shap_values_single = np.sum(shap_values, axis=2)
            else:
                shap_values_single = shap_values
            
            fig = create_matplotlib_figure(8, 4, facecolor="#1a1a1a")
            # Limit features shown if requested
            try:
                if max_display is not None:
                    shap.plots.beeswarm(shap_values_single, max_display=max_display, show=False)
                else:
                    shap.plots.beeswarm(shap_values_single, show=False)
            except Exception:
                shap.plots.beeswarm(shap_values_single, show=False)
            
            # Set white text for all elements
            ax = plt.gca()
            if ax.get_facecolor() != (0.0, 0.0, 0.0, 0.0):
                # Keep transparent if configured
                ax.set_facecolor('none')
            _force_white_text(fig)
            
            return finalize_matplotlib_figure(fig)
        else:
            # Original summary plot for other models
            fig = create_matplotlib_figure(8, 4, facecolor="#1a1a1a")
            # Limit features shown if requested
            try:
                if max_display is not None:
                    shap.summary_plot(shap_values, X_proc, max_display=max_display, show=False)
                else:
                    shap.summary_plot(shap_values, X_proc, show=False)
            except Exception:
                shap.summary_plot(shap_values, X_proc, show=False)
            
            # Set white text for all elements
            ax = plt.gca()
            if ax.get_facecolor() != (0.0, 0.0, 0.0, 0.0):
                ax.set_facecolor('none')
            _force_white_text(fig)
            
            return finalize_matplotlib_figure(fig)
    except Exception as e:
        print(f"SHAP summary plot failed: {e}")
        return None

# SHAP waterfall plot

def shap_waterfall_plot(model, X_sample, row_idx):
    try:
        core_model, X_proc = _prepare_for_shap(model, X_sample)
        
        # More robust explainer selection
        if hasattr(core_model, "estimators_"):  # Random Forest
            explainer = shap.TreeExplainer(core_model)
        elif core_model.__class__.__name__.lower().startswith("xgb"):  # XGBoost
            explainer = shap.TreeExplainer(core_model)
        elif hasattr(core_model, "coef_"):  # Linear models
            explainer = shap.Explainer(core_model, X_proc)
        else:
            # Try TreeExplainer first, fallback to Explainer
            try:
                explainer = shap.TreeExplainer(core_model)
            except:
                explainer = shap.Explainer(core_model, X_proc)
        
        shap_values = explainer(X_proc)
        
        if row_idx >= len(X_proc):
            row_idx = 0
        
        # Handle multi-class SHAP values
        if len(shap_values.shape) == 3:  # (samples, features, classes)
            # For binary classification, use the positive class (index 1)
            # For multi-class, we could choose a specific class or sum them
            if shap_values.shape[2] == 2:  # Binary classification
                shap_values_single = shap_values[row_idx, :, 1]  # Use positive class
            else:
                # For multi-class, sum the absolute values
                shap_values_single = np.sum(shap_values[row_idx], axis=1)
        else:
            shap_values_single = shap_values[row_idx]
        
        # Try different waterfall plot methods
        try:
            # Method 1: Try the newer plots.waterfall
            ax = shap.plots.waterfall(shap_values_single, show=False)
            if ax is None:
                # If waterfall returns None, create our own
                fig = plt.figure(figsize=(6, 3), facecolor='#1a1a1a')
                ax = plt.gca()
                values = shap_values_single.values if hasattr(shap_values_single, 'values') else shap_values_single
                feature_names = X_proc.columns if hasattr(X_proc, 'columns') else [f"Feature {i}" for i in range(len(values))]
                
                # Sort by absolute values and take top 10
                abs_values = np.abs(values)
                top_indices = np.argsort(abs_values)[-10:]
                
                y_pos = np.arange(len(top_indices))
                colors = ['red' if v < 0 else 'blue' for v in values[top_indices]]
                ax.barh(y_pos, values[top_indices], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([feature_names[i] for i in top_indices])
                ax.set_xlabel('SHAP Value')
                ax.set_title('SHAP Values (Top 10 Features)')
            else:
                # Get the figure from the axes
                fig = ax.figure
        except Exception as e1:
            try:
                # Method 2: Try the older waterfall_plot
                fig = plt.figure(figsize=(6, 3), facecolor='#1a1a1a')
                shap.waterfall_plot(shap_values_single, show=False)
            except Exception as e2:
                # Method 3: Create a simple bar plot as fallback
                fig = plt.figure(figsize=(6, 3), facecolor='#1a1a1a')
                ax = plt.gca()
                values = shap_values_single.values if hasattr(shap_values_single, 'values') else shap_values_single
                feature_names = X_proc.columns if hasattr(X_proc, 'columns') else [f"Feature {i}" for i in range(len(values))]
                
                # Sort by absolute values and take top 10
                abs_values = np.abs(values)
                top_indices = np.argsort(abs_values)[-10:]
                
                y_pos = np.arange(len(top_indices))
                colors = ['red' if v < 0 else 'blue' for v in values[top_indices]]
                ax.barh(y_pos, values[top_indices], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([feature_names[i] for i in top_indices])
                ax.set_xlabel('SHAP Value')
                ax.set_title('SHAP Values (Top 10 Features)')
        
        # Set white text for all elements
        ax = plt.gca()
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"SHAP waterfall plot failed: {e}")
        return None
