"""
Model playground view for interactive model predictions.
"""

import streamlit as st
from visualization.components import metric_card
import pandas as pd
import numpy as np
import plotly.express as px
from core.data_handler import load_dataset
from core.data_processor import load_preprocessor
from core.session_manager import set_session
from services.ml_service import MLService

def create_model_playground_view():
    st.header("Model Playground")
    st.markdown("**Test your trained model with real-world scenarios**")
    
    # Check if model is selected
    ds_key = st.session_state.get("dataset")
    model_key = st.session_state.get("selected_model")
    
    if ds_key is None or model_key is None:
        st.error("Please select a dataset and train a model first.")
        if st.button("Back to Model Selection"):
            set_session("current_view", "Model Selection")
            st.rerun()
        return
    
    try:
        # Initialize service
        ml_service = MLService()
        
        # Load model and dataset
        model = ml_service.load_model(ds_key, model_key)
        df = load_dataset(ds_key)
        
        # Get dataset context
        from config.datasets import DATASETS
        dataset_config = DATASETS.get(ds_key, {})
        target = dataset_config.get("target")
        prediction_context = dataset_config.get("prediction_context", {})
        features_explanation = dataset_config.get("features_explanation", {})
        
        if target is None or target not in df.columns:
            st.error("Could not determine target variable. Please check dataset configuration.")
            return
            
        # Display dataset context
        st.markdown("---")
        st.subheader("Understanding Your Prediction")
        
        # Create columns for context display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction_context:
                st.markdown(f"**What are we predicting?**")
                st.info(prediction_context.get("what_is_predicted", "Unknown"))
                
                st.markdown("**What do the predictions mean?**")
                pred_values = prediction_context.get("prediction_values", {})
                for value, meaning in pred_values.items():
                    st.markdown(f"• **{value}**: {meaning}")
                
                st.markdown("**Business Impact:**")
                st.success(prediction_context.get("business_implication", "Unknown"))
                
                st.markdown("**Real-World Use:**")
                st.info(prediction_context.get("real_world_use", "Unknown"))
        
        with col2:
            # Show dataset info
            st.markdown("**Dataset Info:**")
            st.metric("Total Records", len(df))
            st.metric("Features", len(df.columns) - 1)
            
            # Show target distribution
            if target in df.columns:
                target_dist = df[target].value_counts()
                st.markdown("**Target Distribution:**")
                for val, count in target_dist.items():
                    percentage = (count / len(df)) * 100
                    st.markdown(f"• {val}: {count} ({percentage:.1f}%)")
            
        # Load preprocessor to get original feature list
        preprocessor = load_preprocessor(ds_key, model_key)
        if preprocessor is not None and hasattr(preprocessor, 'feature_names_in_'):
            feature_columns = list(preprocessor.feature_names_in_)
        else:
            # Fallback to raw dataframe columns
            feature_columns = [col for col in df.columns if col != target]

        st.markdown("---")
        st.subheader("Test Your Model")
        
        # Allow user to auto-fill inputs with a random example
        if st.button("Load Random Example"):
            example_row = df.sample(1).iloc[0]
            for feat in feature_columns[:20]:  # Limit to first 20 features
                st.session_state[f"__input_{feat}"] = example_row[feat]
            st.success("Random example loaded! Adjust the values below to test different scenarios.")

        # Create input form with better explanations
        with st.form("prediction_form"):
            st.markdown("**Enter your test scenario:**")
            
            user_inputs = {}
            from config.app_config import APP_CONFIG
            max_feats = APP_CONFIG.get("max_input_features", 20)
            
            # Create columns for better layout
            cols = st.columns(2)
            col_idx = 0
            
            for i, feature in enumerate(feature_columns[:max_feats]):
                with cols[col_idx]:
                    col_type = df[feature].dtype if feature in df.columns else np.object_
                    default_val = st.session_state.get(f"__input_{feature}")
                    
                    # Get feature explanation
                    feature_explanation = features_explanation.get(feature, f"Feature: {feature}")
                    
                    if np.issubdtype(col_type, np.number):
                        min_val = float(df[feature].min()) if feature in df.columns else 0.0
                        max_val = float(df[feature].max()) if feature in df.columns else 1.0
                        median_val = float(df[feature].median()) if feature in df.columns else 0.5
                        
                        st.markdown(f"**{feature}**")
                        st.caption(feature_explanation)
                        user_inputs[feature] = st.slider(
                            label=f"Value for {feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val if isinstance(default_val, (int, float)) else median_val,
                            step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.1,
                            key=f"slider_{feature}"
                        )
                    else:
                        unique_vals = df[feature].unique() if feature in df.columns else []
                        if len(unique_vals) > 0:
                            st.markdown(f"**{feature}**")
                            st.caption(feature_explanation)
                            user_inputs[feature] = st.selectbox(
                                label=f"Select {feature}",
                                options=unique_vals,
                                index=0,
                                key=f"select_{feature}"
                            )
                        else:
                            st.markdown(f"**{feature}**")
                            st.caption(feature_explanation)
                            user_inputs[feature] = st.text_input(
                                label=f"Enter {feature}",
                                value=default_val if default_val else "",
                                key=f"text_{feature}"
                            )
                
                # Switch to next column every 2 features
                col_idx = (col_idx + 1) % 2
            
            # Prediction button
            predict_btn = st.form_submit_button("Make Prediction", use_container_width=True)
        
        # Handle prediction
        if predict_btn:
            try:
                # Make prediction using service
                result = ml_service.make_prediction(model, user_inputs, ds_key, model_key)
                
                if result["success"]:
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    prediction = result["prediction"]
                    probability = result["probability"]
                    confidence = result["confidence"]
                    
                    # Display prediction with simplified labels
                    pred_values = prediction_context.get("prediction_values", {})
                    prediction_meaning = pred_values.get(str(prediction), f"Class {prediction}")
                    
                    # Extract short label from the meaning
                    if ds_key == "stocks":
                        if prediction == 1:
                            short_label = "Above Average"
                        else:
                            short_label = "Not Above Average"
                    elif ds_key == "terrorism":
                        if prediction == 1:
                            short_label = "Successful"
                        else:
                            short_label = "Unsuccessful"
                    elif ds_key == "netflix":
                        if prediction == 1:
                            short_label = "Movie"
                        else:
                            short_label = "TV Show"
                    else:
                        short_label = prediction_meaning
                    
                    # Create columns for results
                    result_col1, result_col2 = st.columns([1, 1])
                    
                    with result_col1:
                        st.markdown("**Your Prediction:**")
                        if prediction == 1:
                            st.success(f"**{short_label}**")
                        else:
                            st.error(f"**{short_label}**")
                        
                        if confidence is not None:
                            st.metric("Confidence Level", f"{confidence * 100:.1f}%")
                    
                    with result_col2:
                        if probability is not None:
                            st.markdown("**Probability Breakdown:**")
                            # Create probability bar chart
                            classes = [f"Class {i}" for i in range(len(probability))]
                            prob_df = pd.DataFrame({
                                'Class': classes,
                                'Probability': probability
                            })
                            
                            fig = px.bar(
                                prob_df, 
                                x='Class', 
                                y='Probability',
                                title="Prediction Probabilities",
                                color='Probability',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(
                                title_font_size=14,
                                xaxis_title="Prediction Class",
                                yaxis_title="Probability"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Show business interpretation
                    st.markdown("**Business Interpretation:**")
                    business_impact = prediction_context.get("business_implication", "")
                    if business_impact:
                        st.info(business_impact)
                    
                else:
                    st.error(f"Prediction failed: {result['error']}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        
        # Navigation
        st.markdown("---")
        # Actions
        cols = st.columns([1,1,1])
        with cols[0]:
            if st.button("Back to Model Analysis"):
                set_session("current_view", "Model Analysis")
                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
        with cols[1]:
            if st.button("Try Different Model"):
                set_session("current_view", "Model Selection")
                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if st.button("Back to Model Selection"):
            set_session("current_view", "Model Selection")
            st.rerun()

# For direct testing
if __name__ == "__main__":
    create_model_playground_view()
