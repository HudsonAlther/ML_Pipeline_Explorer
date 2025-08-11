"""
Model selection view for ML Pipeline Explorer.
"""

import streamlit as st
from core.model_manager import list_models, get_model
from core.session_manager import set_session
from pathlib import Path
import time
import random
import base64

video_map = {
    "logreg": "static/animations/LogisticTraining.mp4",
    "random_forest": "static/animations/RandomForestTraining.mp4",
    "xgboost": "static/animations/XGBoostTraining.mp4",
}

from config.app_config import APP_CONFIG

def create_training_popup(video_b64):
    """Create a popup-style training animation overlay.

    The popup automatically hides after a configurable duration defined in
    `APP_CONFIG['model_selection_sleep']` (seconds)."""
    auto_hide_ms = int(APP_CONFIG.get("model_selection_sleep", 10) * 1000)  # Increased to 10 seconds
    st.markdown(f"""
    <style>
        .popup-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            animation: fadeIn 0.3s ease-in;
            padding: 2vw;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .popup-content {{
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border-radius: 20px;
            padding: clamp(20px, 3vw, 60px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            text-align: center;
            max-width: min(70vw, 800px);
            width: 100%;
            position: relative;
            animation: slideIn 0.4s ease-out;
            border: 1px solid #777777;
        }}

        @media (max-width: 1280px) {{
            .popup-content {{ max-width: min(65vw, 720px); }}
        }}
        @media (max-width: 1024px) {{
            .popup-content {{ max-width: min(80vw, 680px); }}
        }}
        @media (max-width: 768px) {{
            .popup-content {{ max-width: 92vw; }}
        }}
        
        @keyframes slideIn {{
            from {{ transform: translateY(-50px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        
        .training-video {{
            width: 100%;
            max-width: 100%;
            height: auto;
            aspect-ratio: 16 / 9;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
            margin: clamp(12px, 2vw, 20px) 0;
        }}
        
        .training-status {{
            color: #70B73F; /* lightgreen */
            font-size: clamp(16px, 1.5vw + 8px, 22px);
            font-weight: bold;
            margin: clamp(12px, 2vw, 20px) 0;
            padding: clamp(8px, 1.2vw, 15px) clamp(16px, 2vw, 30px);
            background: linear-gradient(90deg, #1a1a1a, #2a2a2a);
            border-radius: 30px;
            box-shadow: 0 4px 12px rgba(112, 183, 63, 0.2);
            animation: pulse 2s infinite;
            border: 1px solid #70B73F;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.05); opacity: 0.8; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        
        .training-title {{
            color: #ffffff;
            font-size: clamp(20px, 2vw + 12px, 32px);
            font-weight: bold;
            margin-bottom: clamp(12px, 2vw, 20px);
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}
    </style>
    
    <div class="popup-overlay" id="trainingPopup">
        <div class="popup-content">
            <div class="training-title">Model Training</div>
            <video class="training-video" autoplay muted loop playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="training-status">Training in Progress...</div>
        </div>
    </div>
    
    <script>
        // Auto-hide popup after {APP_CONFIG.get('model_selection_sleep', 10)} seconds
        setTimeout(function() {{
            var popup = document.getElementById('trainingPopup');
            if (popup) {{
                popup.style.animation = 'fadeOut 0.5s ease-out';
                setTimeout(function() {{
                    popup.style.display = 'none';
                }}, 500);
            }}
        }}, {auto_hide_ms});
        
        // Add fadeOut animation
        var style = document.createElement('style');
        style.textContent = '@keyframes fadeOut {{ from {{ opacity: 1; }} to {{ opacity: 0; }} }}';
        document.head.appendChild(style);
    </script>
    """, unsafe_allow_html=True)

def model_selection_view():
    st.header("Step 3: Choose Your Model")
    st.markdown("**Select the algorithm that will learn from your data**")
    
    # Important: Training happens via CLI, UI shows simulation
    st.info("💡 **Note:** Models must be pre-trained using `trainer_refactored.py`. The UI shows training simulation and loads existing artifacts.")

    # Selection Guide removed per request

    if st.session_state["dataset"] is None:
        st.error("Please select a dataset first.")
        set_session("current_view", "Data Preparation")
        st.rerun()

    # Get dataset context
    ds_key = st.session_state.get("dataset")
    from config.datasets import DATASETS
    dataset_config = DATASETS.get(ds_key, {})
    prediction_context = dataset_config.get("prediction_context", {})
    
    # Show current problem context
    if prediction_context:
        st.markdown("---")
        st.subheader("Your Current Problem")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Dataset:** {ds_key.title()}")
            st.markdown(f"**What we're predicting:** {prediction_context.get('what_is_predicted', 'Unknown')}")
            st.markdown(f"**Business goal:** {prediction_context.get('business_implication', 'Unknown')}")
        
        with col2:
            st.metric("Target Variable", dataset_config.get("target", "Unknown"))
    
    # Model information 
    model_labels = {
        "logreg": "Logistic Regression",
        "random_forest": "Random Forest", 
        "xgboost": "XGBoost"
    }
    
    #  model descriptions 
    model_descriptions = {
        "logreg": (
            "**Simple & Fast:** Like a straight line that separates two groups. "
            "Easy to understand and explain to others. "
            "Good for when you need to know 'why' the model made its decision."
        ),
        "random_forest": (
            "**Smart & Reliable:** Like having many experts vote on the answer. "
            "Each expert (tree) looks at different parts of the data. "
            "Very good at finding patterns and handling messy data."
        ),
        "xgboost": (
            "**High Performance:** Like a student who learns from their mistakes. "
            "Each step improves on the previous one. "
            "Often the most accurate, but harder to explain."
        ),
    }
    
    # Model strengths and weaknesses in business terms
    model_details = {
        "logreg": {
            "strengths": ["Fast training", "Easy to explain", "Good baseline"],
            "weaknesses": [
                "Assumes a linear decision boundary",
                "Sensitive to multicollinearity and outliers",
                "Misses interactions unless features are engineered"
            ],
            "best_for": "When you need to explain decisions to stakeholders",
            "business_use": "Risk assessment, simple classification tasks"
        },
        "random_forest": {
            "strengths": ["Handles complex patterns", "Robust to errors", "Shows feature importance"],
            "weaknesses": [
                "Less interpretable than linear models",
                "Inference can be slower with many trees",
                "Can overfit noisy data without tuning"
            ],
            "best_for": "When you have messy data with many features",
            "business_use": "Customer segmentation, fraud detection"
        },
        "xgboost": {
            "strengths": ["Highest accuracy", "Handles missing data", "Built-in optimization"],
            "weaknesses": [
                "Many hyperparameters to tune",
                "Harder to explain to non-technical audiences",
                "Training can be slow on very large datasets"
            ],
            "best_for": "When accuracy is the top priority",
            "business_use": "Competition winning, high-stakes predictions"
        }
    }
    
    st.markdown("### Choose Your Model")
    st.markdown("**Each model has different strengths. Choose based on your needs:**")
    
    # Create toggle selector at the top
    tab1, tab2, tab3 = st.tabs([
        "Logistic Regression", 
        "Random Forest", 
        "XGBoost"
    ])
    
    # Defensive: ensure no leftover selection triggers auto-skip
    if st.session_state.get("selected_model") is not None:
        set_session("selected_model", None)

    # Tab 1: Logistic Regression
    with tab1:
        model_key = "logreg"
        details = model_details.get(model_key, {})
        
        st.markdown("---")
        st.markdown(f"**{model_labels.get(model_key, model_key.title())}**")
        st.markdown(model_descriptions.get(model_key, ''))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strengths:**")
            for strength in details.get('strengths', []):
                st.markdown(f"• {strength}")
            
            # Create a container with fixed height for alignment
            with st.container():
                st.markdown("**Best for:**")
                st.info(details.get('best_for', ''))
        
        with col2:
            st.markdown("**Limitations:**")
            for weakness in details.get('weaknesses', [])[:3]:
                st.markdown(f"• {weakness}")
            
            # Create a container with fixed height for alignment
            with st.container():
                st.markdown("**Business use:**")
                st.success(details.get('business_use', ''))
        
        col1, = st.columns(1)
        
        with col1:
            if st.button(f"Select This Model", 
                       key=f"btn_{model_key}", use_container_width=True):
                set_session("selected_model", model_key)
                
                # Display popup animation while training
                video_path = video_map.get(model_key)
                
                if video_path and Path(video_path).exists():
                    try:
                        with open(video_path, "rb") as vid_file:
                            video_bytes = vid_file.read()
                            video_b64 = base64.b64encode(video_bytes).decode()
                        
                        # Create popup with video
                        create_training_popup(video_b64)
                        
                    except Exception as e:
                        st.warning(f"Could not load animation: {str(e)}")
                        st.info("Training in progress... Please wait.")
                else:
                    st.warning(f"Animation file not found: {video_path}")
                    st.info("Training in progress... Please wait.")

                try:
                    from services.ml_service import MLService
                    ml_service = MLService()
                    
                    # Check if model artifacts exist
                    model_metadata = ml_service.get_model_metadata(ds_key, model_key)
                    if not model_metadata:
                        st.error(f"No pre-trained {model_key} model found for {ds_key} dataset. Please run trainer_refactored.py first.")
                        return
                    
                    # Simulate training time with video
                    time.sleep(random.uniform(7, 10))
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return

                # Automatically move to step 4
                set_session("current_view", "Model Analysis")
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()

        

        # No post-training manual action in auto-advance design

    # Tab 2: Random Forest
    with tab2:
        model_key = "random_forest"
        details = model_details.get(model_key, {})
        
        st.markdown("---")
        st.markdown(f"**{model_labels.get(model_key, model_key.title())}**")
        st.markdown(model_descriptions.get(model_key, ''))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strengths:**")
            for strength in details.get('strengths', []):
                st.markdown(f"• {strength}")
            
            # Create a container with fixed height for alignment
            with st.container():
                st.markdown("**Best for:**")
                st.info(details.get('best_for', ''))
        
        with col2:
            st.markdown("**Limitations:**")
            for weakness in details.get('weaknesses', [])[:3]:
                st.markdown(f"• {weakness}")
            
            # Create a container with fixed height for alignment
            with st.container():
                st.markdown("**Business use:**")
                st.success(details.get('business_use', ''))
        
        col1, = st.columns(1)
        
        with col1:
            if st.button(f"Select This Model", 
                       key=f"btn_{model_key}", use_container_width=True):
                set_session("selected_model", model_key)
                
                # Display popup animation while training
                video_path = video_map.get(model_key)
                
                if video_path and Path(video_path).exists():
                    try:
                        with open(video_path, "rb") as vid_file:
                            video_bytes = vid_file.read()
                            video_b64 = base64.b64encode(video_bytes).decode()
                        
                        # Create popup with video
                        create_training_popup(video_b64)
                        
                    except Exception as e:
                        st.warning(f"Could not load animation: {str(e)}")
                        st.info("Training in progress... Please wait.")
                else:
                    st.warning(f"Animation file not found: {video_path}")
                    st.info("Training in progress... Please wait.")

                # Check if pre-trained model exists
                try:
                    from services.ml_service import MLService
                    ml_service = MLService()
                    
                    # Check if model artifacts exist
                    model_metadata = ml_service.get_model_metadata(ds_key, model_key)
                    if not model_metadata:
                        st.error(f"No pre-trained {model_key} model found for {ds_key} dataset. Please run trainer_refactored.py first.")
                        return
                    
                    # Simulate training time with video
                    time.sleep(random.uniform(7, 10))
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return

                # Automatically move to step 4
                set_session("current_view", "Model Analysis")
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()

        with col2:
            st.empty()

        # No post-training manual action in auto-advance design
    
    # Tab 3: XGBoost
    with tab3:
        model_key = "xgboost"
        details = model_details.get(model_key, {})
        
        st.markdown("---")
        st.markdown(f"**{model_labels.get(model_key, model_key.title())}**")
        st.markdown(model_descriptions.get(model_key, ''))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strengths:**")
            for strength in details.get('strengths', []):
                st.markdown(f"• {strength}")
            
            # Create a container with fixed height for alignment
            with st.container():
                st.markdown("**Best for:**")
                st.info(details.get('best_for', ''))
        
        with col2:
            st.markdown("**Limitations:**")
            for weakness in details.get('weaknesses', [])[:3]:
                st.markdown(f"• {weakness}")
            
            # Create a container with fixed height for alignment
            with st.container():
                st.markdown("**Business use:**")
                st.success(details.get('business_use', ''))
        
        col1, = st.columns(1)
        
        with col1:
            if st.button(f"Select This Model", 
                       key=f"btn_{model_key}", use_container_width=True):
                set_session("selected_model", model_key)
                
                # Display popup animation while training
                video_path = video_map.get(model_key)
                
                if video_path and Path(video_path).exists():
                    try:
                        with open(video_path, "rb") as vid_file:
                            video_bytes = vid_file.read()
                            video_b64 = base64.b64encode(video_bytes).decode()
                        
                        # Create popup with video
                        create_training_popup(video_b64)
                        
                    except Exception as e:
                        st.warning(f"Could not load animation: {str(e)}")
                        st.info("Training in progress... Please wait.")
                else:
                    st.warning(f"Animation file not found: {video_path}")
                    st.info("Training in progress... Please wait.")

                # Check if pre-trained model exists
                try:
                    from services.ml_service import MLService
                    ml_service = MLService()
                    
                    # Check if model artifacts exist
                    model_metadata = ml_service.get_model_metadata(ds_key, model_key)
                    if not model_metadata:
                        st.error(f"No pre-trained {model_key} model found for {ds_key} dataset. Please run trainer_refactored.py first.")
                        return
                    
                    # Simulate training time with video
                    time.sleep(random.uniform(7, 10))
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return

                # Automatically move to step 4
                set_session("current_view", "Model Analysis")
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()

        with col2:
            st.empty()
        
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        if st.button("Back to Data Preparation"):
            set_session("current_view", "Data Preparation")
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
    