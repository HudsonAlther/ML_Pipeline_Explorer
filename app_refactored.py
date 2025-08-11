"""
Main entry point for the refactored ML Pipeline Explorer Streamlit app.
"""

import streamlit as st
from core.session_manager import initialize_session
from views.dataset_selection import dataset_selection_view
from views.data_preparation import data_preparation_view
from views.model_selection import model_selection_view
from views.model_analysis import model_analysis_view
from views.model_playground import create_model_playground_view
from config.app_config import APP_CONFIG
from config.validator import validate_config
from services.ml_service import MLService

# Make sure VIEW_MAP is defined at the module level
VIEW_MAP = {
    "Dataset Selection": dataset_selection_view,
    "Data Preparation": data_preparation_view,
    "Model Selection": model_selection_view,
    "Model Analysis": model_analysis_view,
    "Model Playground": create_model_playground_view
}

def main():
    """Main application entry point"""
    st.set_page_config(page_title="ML Pipeline Explorer", layout="wide")
    # Global responsive typography
    st.markdown(
        """
        <style>
        :root { --base-font: clamp(14px, 1.1vw + 10px, 18px); }
        html, body, [class*="st-"] { font-size: var(--base-font); }
        h1 { font-size: clamp(26px, 2.2vw + 16px, 38px); line-height: 1.2; }
        h2 { font-size: clamp(22px, 1.8vw + 12px, 32px); line-height: 1.25; }
        h3 { font-size: clamp(18px, 1.3vw + 10px, 26px); line-height: 1.3; }
        p, li { line-height: 1.5; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize session
    initialize_session()
    
    # Validate configuration on startup
    if "config_validated" not in st.session_state:
        with st.spinner("Validating configuration..."):
            is_valid = validate_config()
            st.session_state["config_validated"] = is_valid
        
        if not is_valid:
            st.error("Configuration validation failed. Please check your setup.")
            st.stop()
    
    # Define steps
    step_names = [
        "Dataset Selection",
        "Data Preparation", 
        "Model Selection",
        "Model Analysis",
        "Model Playground"
    ]
    step_keys = [
        "Dataset Selection",
        "Data Preparation",
        "Model Selection",
        "Model Analysis",
        "Model Playground"
    ]
    
    current_view = st.session_state.get("current_view", step_keys[0])
    
    # Timeline progress bar
    st.markdown("""
        <style>
        .timeline-step {
            display: flex; flex-direction: column; align-items: center; width: 100%;
        }
        .timeline-circle {
            width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.1em; margin-bottom: 4px;
        }
        .timeline-label {
            font-size: 0.98em; text-align: center; margin-bottom: 8px;
        }
        .timeline-bar {
            height: 5px; margin: 0 0.5em; border-radius: 2px; flex: 1;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create timeline
    cols = st.columns(len(step_names) * 2 - 1)
    for idx, name in enumerate(step_names):
        with cols[idx * 2]:
            is_active = (current_view == step_keys[idx])
            is_visited = step_keys.index(current_view) > idx
            
            # Determine circle style based on state
            if is_active:
                # Current step - lightgreen
                circle_style = "background: #70B73F; border: 3px solid #fff; color: #fff; box-shadow: 0 2px 8px rgba(112, 183, 63, 0.3);"
                label_style = "color: #70B73F; font-weight: 600;"
            elif is_visited:
                # Visited step - primary blue
                circle_style = "background: #005E7B; border: 2px solid #005E7B; color: #fff;"
                label_style = "color: #005E7B; font-weight: 500;"
            else:
                # Not visited step - Grey
                circle_style = "background: #777777; border: 2px solid #777777; color: #fff;"
                label_style = "color: #777777; font-weight: 400;"
            
            st.markdown(f"<div class='timeline-step'>"
                        f"<div class='timeline-circle' style='{circle_style}'>{idx+1}</div>"
                        f"<div class='timeline-label' style='{label_style}'>{name}</div>"
                        f"</div>", unsafe_allow_html=True)
        
        # Connect bars between steps
        if idx < len(step_names) - 1:
            with cols[idx * 2 + 1]:
                # Determine bar color based on progress
                if step_keys.index(current_view) > idx:
                    # Progress made - primary blue bar
                    bar_style = "background: #005E7B;"
                else:
                    # No progress yet - Grey bar
                    bar_style = "background: #777777;"
                
                st.markdown(f"<div class='timeline-bar' style='{bar_style}'></div>", unsafe_allow_html=True)
    
    # Show only the current step's content below the timeline
    st.markdown("---")
    step_idx = step_keys.index(current_view) if current_view in step_keys else 0
    
    try:
        VIEW_MAP[step_keys[step_idx]]()
    except Exception as e:
        st.error(f"Error in {current_view}: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
