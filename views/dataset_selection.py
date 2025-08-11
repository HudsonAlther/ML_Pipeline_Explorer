"""
Dataset selection view for ML Pipeline Explorer.
"""

import streamlit as st
from services.ml_service import MLService
from core.session_manager import set_session

def dataset_selection_view():
    """Dataset selection view"""
    st.header("Step 1: Choose Your Business Problem")
    st.markdown("**Select a real-world scenario to explore machine learning**")
    
    # Initialize service
    ml_service = MLService()
    datasets = ml_service.get_datasets()

    # Map dataset keys to business domains
    topic_labels = {
        "stocks": "Finance & Trading",
        "terrorism": "Security & Risk Assessment",
        "netflix": "Entertainment & Content",
    }

    # Enhanced problem descriptions with business context
    problem_descriptions = {
        "stocks": (
            "**Trading Decision Support:** Predict whether Apple's stock will close above its historical average. "
            "This helps traders decide when to buy (above average) or sell (below average) based on current market conditions. "
            "Used by retail investors and day traders for quick buy/sell decisions."
        ),
        "terrorism": (
            "**Security Threat Assessment:** Predict whether a reported incident will be a successful terrorist attack. "
            "This helps security analysts prioritize threat assessments and allocate resources more effectively. "
            "Used by security agencies to evaluate threat levels and plan countermeasures."
        ),
        "netflix": (
            "**Content Classification:** Predict whether a Netflix title is a Movie or TV Show based on metadata. "
            "This helps organize the catalog and improve recommendation algorithms. "
            "Used by streaming platforms to categorize content and enhance user experience."
        ),
    }
    
    # Add prediction context for each dataset
    prediction_context = {
        "stocks": {
            "what_we_predict": "Stock price direction (Above/Below average)",
            "business_value": "Trading decisions and portfolio management",
            "real_world_impact": "Helps investors make buy/sell decisions",
            "risk_note": "This is a simple heuristic - not financial advice"
        },
        "terrorism": {
            "what_we_predict": "Attack success probability",
            "business_value": "Security resource allocation and threat assessment",
            "real_world_impact": "Helps prioritize security measures",
            "risk_note": "For security planning purposes only"
        },
        "netflix": {
            "what_we_predict": "Content type (Movie vs TV Show)",
            "business_value": "Catalog organization and recommendations",
            "real_world_impact": "Improves user experience and content discovery",
            "risk_note": "For content organization purposes"
        }
    }
    
    if st.session_state["dataset"] is None:
        # Create toggle selector at the top
        st.markdown("### Choose Your Business Scenario")
        
        # Create tabs for the 3 options
        tab1, tab2, tab3 = st.tabs([
            f"{topic_labels['stocks']}", 
            f"{topic_labels['terrorism']}", 
            f"{topic_labels['netflix']}"
        ])
        
        # Tab 1: Stocks
        with tab1:
            ds_key = "stocks"
            ds_cfg = datasets[ds_key]
            context = prediction_context.get(ds_key, {})
            
            st.markdown(f"**{topic_labels.get(ds_key, ds_key.title())}**")
            st.markdown(problem_descriptions.get(ds_key, ''))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**What We Predict:**")
                st.info(context.get('what_we_predict', 'Unknown'))
                
                st.markdown("**Business Value:**")
                st.success(context.get('business_value', 'Unknown'))
            
            with col2:
                st.markdown("**Real-World Impact:**")
                st.info(context.get('real_world_impact', 'Unknown'))
                
                st.markdown("**Dataset Info:**")
                st.markdown(f"• **Dataset:** {ds_key.title()}")
                st.markdown(f"• **Target:** {ds_cfg.get('target', '').title()}")
            
            st.markdown("**Risk Note:**")
            st.caption(context.get('risk_note', ''))
            
            if st.button(f"Select {topic_labels.get(ds_key, ds_key.title())}", 
                       key=f"btn_{ds_key}", use_container_width=True):
                with st.spinner(f"Validating {ds_key} dataset..."):
                    if ml_service.validate_dataset(ds_key):
                        set_session("dataset", ds_key)
                        set_session("current_view", "Data Preparation")
                        st.success(f"{ds_key.title()} dataset selected! Loading data preparation...")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        st.error(f"Failed to validate dataset: {ds_key}")
        
        # Tab 2: Terrorism
        with tab2:
            ds_key = "terrorism"
            ds_cfg = datasets[ds_key]
            context = prediction_context.get(ds_key, {})
            
            st.markdown("---")
            st.markdown(f"**{topic_labels.get(ds_key, ds_key.title())}**")
            st.markdown(problem_descriptions.get(ds_key, ''))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**What We Predict:**")
                st.info(context.get('what_we_predict', 'Unknown'))
                
                st.markdown("**Business Value:**")
                st.success(context.get('business_value', 'Unknown'))
            
            with col2:
                st.markdown("**Real-World Impact:**")
                st.info(context.get('real_world_impact', 'Unknown'))
                
                st.markdown("**Dataset Info:**")
                st.markdown(f"• **Dataset:** {ds_key.title()}")
                st.markdown(f"• **Target:** {ds_cfg.get('target', '').title()}")
            
            st.markdown("**Risk Note:**")
            st.caption(context.get('risk_note', ''))
            
            if st.button(f"Select {topic_labels.get(ds_key, ds_key.title())}", 
                       key=f"btn_{ds_key}", use_container_width=True):
                with st.spinner(f"Validating {ds_key} dataset..."):
                    if ml_service.validate_dataset(ds_key):
                        set_session("dataset", ds_key)
                        set_session("current_view", "Data Preparation")
                        st.success(f"{ds_key.title()} dataset selected! Loading data preparation...")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        st.error(f"Failed to validate dataset: {ds_key}")
        
        # Tab 3: Netflix
        with tab3:
            ds_key = "netflix"
            ds_cfg = datasets[ds_key]
            context = prediction_context.get(ds_key, {})
            
            st.markdown("---")
            st.markdown(f"**{topic_labels.get(ds_key, ds_key.title())}**")
            st.markdown(problem_descriptions.get(ds_key, ''))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**What We Predict:**")
                st.info(context.get('what_we_predict', 'Unknown'))
                
                st.markdown("**Business Value:**")
                st.success(context.get('business_value', 'Unknown'))
            
            with col2:
                st.markdown("**Real-World Impact:**")
                st.info(context.get('real_world_impact', 'Unknown'))
                
                st.markdown("**Dataset Info:**")
                st.markdown(f"• **Dataset:** {ds_key.title()}")
                st.markdown(f"• **Target:** {ds_cfg.get('target', '').title()}")
            
            st.markdown("**Risk Note:**")
            st.caption(context.get('risk_note', ''))
            
            if st.button(f"Select {topic_labels.get(ds_key, ds_key.title())}", 
                       key=f"btn_{ds_key}", use_container_width=True):
                with st.spinner(f"Validating {ds_key} dataset..."):
                    if ml_service.validate_dataset(ds_key):
                        set_session("dataset", ds_key)
                        set_session("current_view", "Data Preparation")
                        st.success(f"{ds_key.title()} dataset selected! Loading data preparation...")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()
                    else:
                        st.error(f"Failed to validate dataset: {ds_key}")

    # If somehow we are still on this view after selecting a dataset, jump to Data Preparation
    if st.session_state.get("dataset") is not None and st.session_state.get("current_view") == "Dataset Selection":
        set_session("current_view", "Data Preparation")
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
