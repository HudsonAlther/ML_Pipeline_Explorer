"""
Model analysis view for ML Pipeline Explorer.
"""

import streamlit as st
from visualization.components import metric_card
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
from core.data_handler import load_dataset, load_trained_pipeline, load_model_metadata
from core.model_manager import get_model
from core.session_manager import set_session
from utils.metrics import compute_predictions
from visualization.charts import create_feature_importance_plot, create_pdp_ice, shap_summary_plot
from visualization.plots import plot_confusion_heatmap, plot_threshold_dashboard
from config.app_config import APP_CONFIG
from config.datasets import DATASETS

def load_concept_image(model_key):
    """Load concept image for the model type."""
    image_urls = {
        "logreg": "https://miro.medium.com/v2/resize:fit:1400/1*H1e9zJQ5ZQZQZQZQZQZQ.png",
        "random_forest": "https://miro.medium.com/v2/resize:fit:1400/1*QJZ6QZQZQZQZQZQZQZQ.png",
        "xgboost": "https://miro.medium.com/v2/resize:fit:1400/1*XGBoost_Logo.png"
    }
    
    try:
        response = requests.get(image_urls.get(model_key, ""))
        return Image.open(BytesIO(response.content))
    except:
        return None

def plot_learning_curve(history=None):
    """Plot learning curve if available in model history."""
    if not history or not hasattr(history, 'history'):
        return None
        
    metrics = history.history
    fig = go.Figure()
    
    # Add training metrics
    for metric in metrics:
        if not metric.startswith('val_'):
            fig.add_trace(go.Scatter(
                x=list(range(len(metrics[metric]))),
                y=metrics[metric],
                mode='lines+markers',
                name=f'train_{metric}'
            ))
    
    # Add validation metrics
    for metric in metrics:
        if metric.startswith('val_'):
            fig.add_trace(go.Scatter(
                x=list(range(len(metrics[metric]))),
                y=metrics[metric],
                mode='lines+markers',
                name=metric
            ))
    
    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis_title='Metric Value',
        legend_title='Metrics',
        template='plotly_white'
    )
    return fig

def model_analysis_view():
    st.header("Step 4: Model Analysis")
    ds_key = st.session_state.get("dataset")
    model_key = st.session_state.get("selected_model")
    
    if ds_key is None or model_key is None:
        st.error("Please select a dataset and model first.")
        if st.button("Back to Model Selection"):
            set_session("current_view", "Model Selection")
            st.rerun()
        return
    
    # Concept image (optional) without verbose description
    concept_img = load_concept_image(model_key)
    if concept_img:
        st.image(concept_img, width=150)

    # Load model and data
    try:
        model = load_trained_pipeline(ds_key, model_key)
        df = load_dataset(ds_key)
        metadata = load_model_metadata(ds_key, model_key)
        
        if model is None or metadata is None:
            st.error("Model not found. Please train the model first.")
            if st.button("Back to Model Selection"):
                set_session("current_view", "Model Selection")
                st.rerun()
            return
        
        # Model performance metrics (no separate prediction UI)
        tab1, tab2 = st.tabs(["Performance", "Interpretability"])
        
        with tab1:
            st.subheader("Model Performance")
            # Styled, responsive metric boxes: label (small) and big number beneath
            primary_blue = APP_CONFIG.get('colors', {}).get('primary_blue', '#005E7B')
            st.markdown(
                f"""
                <style>
                .metric-box {{
                  background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
                  border: 1px solid #777777;
                  border-radius: 12px;
                  padding: clamp(10px, 1.2vw, 16px);
                  text-align: center;
                }}
                .metric-label {{
                  color: #bfbfbf;
                  font-size: clamp(12px, 1vw + 6px, 16px);
                  letter-spacing: 0.02em;
                  margin-bottom: 6px;
                }}
                .metric-value {{
                  font-size: clamp(28px, 3vw + 14px, 48px);
                  font-weight: 800;
                  color: #ffffff;
                  line-height: 1.1;
                }}
                .metric-accent {{
                  height: 4px; border-radius: 2px; margin-top: 10px; opacity: 0.9;
                  background: {primary_blue};
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown(
                    f"""
                    <div class='metric-box'>
                      <div class='metric-label'>Accuracy</div>
                      <div class='metric-value'>{metadata.get('accuracy', 0):.4f}</div>
                      <div class='metric-accent'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f"""
                    <div class='metric-box'>
                      <div class='metric-label'>ROC AUC</div>
                      <div class='metric-value'>{metadata.get('roc_auc', 0):.4f}</div>
                      <div class='metric-accent'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_c:
                st.markdown(
                    f"""
                    <div class='metric-box'>
                      <div class='metric-label'>F1 Score</div>
                      <div class='metric-value'>{metadata.get('f1_score', 0):.4f}</div>
                      <div class='metric-accent'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_d:
                st.markdown(
                    f"""
                    <div class='metric-box'>
                      <div class='metric-label'>Precision</div>
                      <div class='metric-value'>{metadata.get('precision', 0):.4f}</div>
                      <div class='metric-accent'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Data distribution
            with st.expander("Data Distribution"):
                target = DATASETS[ds_key]["target"]
                fig1 = px.pie(
                    values=df[target].value_counts().values,
                    names=df[target].value_counts().index,
                    title=f"Target Distribution - {target}"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Feature distributions
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    fig2 = px.histogram(
                        df, 
                        x=numeric_cols[0], 
                        color=target,
                        title=f"Feature Distribution - {numeric_cols[0]}"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Model Interpretability")
            
            # Feature importance
            core_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
            if hasattr(core_model, 'feature_importances_') or hasattr(core_model, 'coef_'):
                feature_names = metadata.get('feature_names', [])
                if feature_names:
                    importance_plot = create_feature_importance_plot(model, feature_names, model_key)
                    if importance_plot:
                        st.plotly_chart(importance_plot, use_container_width=True)

            # SHAP explanations (only under Interpretability)
            st.subheader("SHAP Explanations")
            try:
                # Sample data for SHAP (aligned with gallery)
                sample_data = df.sample(min(100, len(df)), random_state=42)
                target = DATASETS[ds_key]["target"]
                X_sample = sample_data.drop(columns=[target])
                
                # If dataset is Netflix, show only top 5 features
                max_disp = 5 if ds_key == "netflix" else None
                shap_plot = shap_summary_plot(model, X_sample, max_display=max_disp)
                if shap_plot:
                    st.pyplot(shap_plot)
            except Exception as e:
                st.warning(f"SHAP analysis not available: {str(e)}")
        
        # Navigation
        st.markdown("---")
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button("Back to Model Selection"):
                set_session("current_view", "Model Selection")
                st.rerun()
        with col2:
            if st.button("Continue to Model Playground", help="Open an interactive space to test predictions and probabilities"):
                set_session("current_view", "Model Playground")
                st.rerun()
                
    except Exception as e:
        st.error(f"Error in Model Analysis: {str(e)}")
        if st.button("Back to Model Selection"):
            set_session("current_view", "Model Selection")
            st.rerun()
