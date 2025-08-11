"""
Data preparation view for ML Pipeline Explorer.
"""

import streamlit as st
from visualization.components import metric_card
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from services.ml_service import MLService
import time
from core.session_manager import set_session

def data_preparation_view():
    """Data preparation view"""
    # Early navigation handling to avoid rendering heavy content on transition
    if st.session_state.get("btn_goto_model_selection"):
        set_session("selected_model", None)
        set_session("current_view", "Model Selection")
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
        return

    st.header("Step 2: Understand Your Data")
    st.markdown("**Explore and prepare your data for machine learning**")
    
    if st.session_state["dataset"] is None:
        st.error("Please select a dataset first.")
        set_session("current_view", "Dataset Selection")
        st.rerun()
    
    # Initialize service
    ml_service = MLService()
    ds_key = st.session_state["dataset"]
    
    # Get dataset context
    from config.datasets import DATASETS
    dataset_config = DATASETS.get(ds_key, {})
    prediction_context = dataset_config.get("prediction_context", {})
    
    # Show dataset context
    if prediction_context:
        st.markdown("---")
        st.subheader("Your Business Problem")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**What we're predicting:** {prediction_context.get('what_is_predicted', 'Unknown')}")
            st.markdown(f"**Business value:** {prediction_context.get('business_implication', 'Unknown')}")
            
            pred_values = prediction_context.get("prediction_values", {})
            st.markdown("**Prediction outcomes:**")
            for value, meaning in pred_values.items():
                st.markdown(f"• **{value}**: {meaning}")
        
        with col2:
            st.metric("Dataset", ds_key.title())
            st.metric("Target Variable", dataset_config.get("target", "Unknown"))
    
    # (Navigation moved to bottom of "Ready for ML" tab)
    
    # Show loading progress
    with st.spinner(f"Loading {ds_key} dataset and generating visualizations..."):
        try:
            # Use session cache so visuals persist across view changes
            cache = st.session_state.setdefault("data_cache", {})
            if ds_key in cache:
                df = cache[ds_key]["df"]
                target = cache[ds_key]["target"]
                plots = cache[ds_key]["plots"]
                dataset_info = cache[ds_key]["dataset_info"]
            else:
                # Load dataset data
                df, target, plots = ml_service.load_dataset_data(ds_key)
                dataset_info = ml_service.get_dataset_info(ds_key)
                cache[ds_key] = {
                    "df": df,
                    "target": target,
                    "plots": plots,
                    "dataset_info": dataset_info,
                }
            
            st.success(f"**{dataset_info.get('description', '')}** - {dataset_info.get('business_value', '')}")
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.info("This might be due to a large dataset taking time to load. Please try again.")
            if st.button("Back to Dataset Selection"):
                set_session("current_view", "Dataset Selection")
                st.rerun()
            return
    
    # Guidance for progressing through tabs
    st.info("Work through these tabs in order: Raw Data → Data Cleaning → Ready for ML. Use the Continue button at the bottom of Ready for ML to proceed.")

    # Ensure we track encoding action state
    st.session_state.setdefault("data_encoded", False)

    # Data preparation interface
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Data Cleaning", "Ready for ML"])
    
    with tab1:
        st.markdown("### Your Raw Data")
        st.markdown("**This is what your data looks like before we prepare it for machine learning:**")
        
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### This Data At A Glance")
            
            # 1) Missing values
            metric_card("Missing values", df.isnull().sum().sum())

            # 2) Quality score (if available)
            data_info = plots.get('data_info', {}) if isinstance(plots, dict) else {}
            if 'data_quality_score' in data_info:
                quality_score = data_info['data_quality_score']
                metric_card("Quality score", f"{quality_score:.2%}")

            # 3) Data types (unique type count)
            metric_card("Data types", len(df.dtypes.unique()))

            # 4) Data types found (detailed breakdown)
            st.markdown("**Data Types Found:**")
            dtype_counts = {}
            for dtype, count in df.dtypes.value_counts().to_dict().items():
                clean_name = str(dtype)
                match clean_name:
                    case 'int64':
                        clean_name = 'Numbers (whole)'
                    case 'float64':
                        clean_name = 'Numbers (decimal)'
                    case 'object':
                        clean_name = 'Text'
                    case 'bool':
                        clean_name = 'Yes/No'
                    case 'category':
                        clean_name = 'Categories'
                    case 'datetime64[ns]':
                        clean_name = 'Dates'
                    case _:
                        clean_name = clean_name
                dtype_counts[clean_name] = int(count)

            for dtype, count in dtype_counts.items():
                st.markdown(f"• {dtype} → {count} columns")

        with col2:
            # Right side: only the target distribution graph
            if 'target_dist_plot' in plots:
                st.markdown("### Target Distribution")
                st.caption("This shows how many times an outcome happened")
                fig = plots['target_dist_plot']
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    
    with tab2:
        st.markdown("### Data Cleaning Process")
        st.markdown("**We need to fix issues in the data before training our model:**")
        
        # Show missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.markdown("**Step 1: Handle Missing Values**")
            st.markdown("**Missing values are gaps in your data that need to be filled:**")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(df) * 100)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            st.markdown("**Our cleaning strategy:**")
            st.markdown("• **Numbers**: Fill with median (middle value)")
            st.markdown("• **Text**: Fill with 'missing'")
            st.markdown("• **Why?** This preserves data while handling gaps")
        else:
            st.success("No missing values - your data is already clean!")
        
        # Show feature correlations
        if 'correlation_plot' in plots:
            # Lay out heatmap (left) and step description (right)
            left_col, right_col = st.columns([3, 2])
            
            with left_col:
                numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1a1a1a')
                    ax.set_facecolor('#2a2a2a')
                    from matplotlib.colors import LinearSegmentedColormap
                    palette = {
                        'mediumblue': '#005E7B',
                        'darkred': '#D0073A',
                    }
                    heatmap_cmap = LinearSegmentedColormap.from_list(
                        'approved_bwr', [palette['mediumblue'], palette['darkred']]
                    )
                    sns.heatmap(
                        corr,
                        annot=True,
                        annot_kws={"color": "white", "size": 8},
                        cmap=heatmap_cmap,
                        center=0,
                        square=True,
                        ax=ax,
                        cbar_kws={'label': 'Correlation', 'shrink': 0.7}
                    )
                    # Style ticks and labels for dark theme
                    ax.set_title('Feature Correlation Matrix', color='white', fontsize=12, fontweight='bold')
                    ax.tick_params(colors='white')
                    for label in ax.get_xticklabels():
                        label.set_color('white')
                        label.set_rotation(45)
                        label.set_ha('right')
                    for label in ax.get_yticklabels():
                        label.set_color('white')
                    try:
                        cbar = ax.collections[0].colorbar
                        if cbar is not None:
                            cbar.ax.yaxis.set_tick_params(color='white')
                            for tick in cbar.ax.get_yticklabels():
                                tick.set_color('white')
                            cbar.set_label('Correlation', color='white')
                    except Exception:
                        pass
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)

            with right_col:
                st.markdown("**Step 2: Understand Feature Relationships**")
                st.markdown("**This helps us understand which features are important for predictions:**")
                st.caption("Darker reds/blues indicate stronger positive/negative relationships between features.")
                
                # Step 3 content placed under Step 2 on the right side
                st.markdown("---")
                st.markdown("**Step 3: Data Transformation**")
                st.markdown("**We convert your data into a format the model can understand:**")
                st.markdown("• **Text categories** → Numbers")
                st.markdown("• **Scale numbers** to same range")
                st.markdown("• **Split data** into training and testing sets")
                
                # Encode action (simulated)
                if st.session_state.get("data_encoded"):
                    st.success("Your data has been encoded!")
                else:
                    if st.button("Encode my data", key="btn_encode_data"):
                        with st.spinner("Encoding data..."):
                            time.sleep(5)
                        st.session_state["data_encoded"] = True
                        st.success("Your data has been encoded!")
        else:
            # Fallback layout when no correlation plot is available
            st.markdown("**Step 2: Understand Feature Relationships**")
            st.caption("Correlation heatmap is unavailable for this dataset.")
            st.markdown("---")
            st.markdown("**Step 3: Data Transformation**")
            st.markdown("**We convert your data into a format the model can understand:**")
            st.markdown("• **Text categories** → Numbers")
            st.markdown("• **Scale numbers** to same range")
            st.markdown("• **Split data** into training and testing sets")
            if st.session_state.get("data_encoded"):
                st.success("Your data has been encoded!")
            else:
                if st.button("Encode my data", key="btn_encode_data_fallback"):
                    with st.spinner("Encoding data..."):
                        time.sleep(5)
                    st.session_state["data_encoded"] = True
                    st.success("Your data has been encoded!")
    
    with tab3:
        st.markdown("### Your Data is Now Ready!")
        st.markdown("**Your data has been prepared for machine learning:**")
        
        # Show prepared data info
        from sklearn.model_selection import train_test_split
        
        # Prepare data for splitting
        y = df[target].copy()
        X = df.drop(columns=[target]).copy()
        
        # Handle missing values for splitting
        numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
        
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X[categorical_cols] = X[categorical_cols].fillna('missing')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if y.nunique()==2 else None
        )
        
        st.markdown("**Data Split Results:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card(
                "Training examples",
                len(X_train),
                helptext="Number of rows used to fit the model. The model learns patterns from this data."
            )
        with col2:
            metric_card(
                "Testing examples",
                len(X_test),
                helptext="Number of rows held out to evaluate the model on unseen data."
            )
        with col3:
            metric_card(
                "Features",
                X_train.shape[1],
                helptext="Count of input variables the model uses after preprocessing/encoding."
            )
        with col4:
            metric_card(
                "Target classes",
                y.nunique(),
                helptext="Number of distinct outcome categories in the target (e.g., 0/1 for binary)."
            )
        
        st.markdown("**What this means:**")
        st.markdown("• **Training data**: Used to teach the model (like studying for a test)")
        st.markdown("• **Testing data**: Used to evaluate the model (like taking the test)")
        st.markdown("• **Why split?** Prevents overfitting (memorizing instead of learning)")
        
        st.markdown("**Your prepared data sample:**")
        st.dataframe(X_train.head(), use_container_width=True)
        
        # Store prepared data in session state 
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["numeric_cols"] = numeric_cols
        st.session_state["categorical_cols"] = categorical_cols
        
    
        st.markdown("---")
        nav_col_left, nav_col_spacer, nav_col_right = st.columns([1, 4, 1])
        with nav_col_left:
            if st.button("Select A Different Dataset", key="btn_back_ready_ml"):
                set_session("current_view", "Dataset Selection")
                set_session("dataset", None)
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
        with nav_col_right:
            if st.button("Continue to Model Selection", key="btn_continue_ready_ml"):
                set_session("selected_model", None)
                set_session("current_view", "Model Selection")
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
    
    # Navigation moved to Ready for ML tab footer

