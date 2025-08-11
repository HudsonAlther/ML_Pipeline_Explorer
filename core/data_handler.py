"""
Data loading and preprocessing utilities.
"""

import os
import json
import numpy as np
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
from config.datasets import DATASETS
from config.app_config import APP_CONFIG
from core.data_processor import fetch_dataset, create_data_exploration_plots

ART_DIR = APP_CONFIG["artifact_dir"]

def ensure_json_serializable(obj):
    """Convert numpy types to JSON serializable types"""
    if isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(ensure_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype'):  # numpy dtype
        return str(obj)
    else:
        return obj

# Save trained model to disk
def save_trained_model(ds_key, model_key, model):
    p = ART_DIR / ds_key / model_key
    p.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p / "model.pkl")

# Save model metrics/metadata to disk
def save_model_metadata(ds_key, model_key, metadata):
    p = ART_DIR / ds_key / model_key
    p.mkdir(parents=True, exist_ok=True)
    
    # Ensure metadata is JSON serializable
    serializable_metadata = ensure_json_serializable(metadata)
    
    with open(p / "metrics.json", "w") as f:
        json.dump(serializable_metadata, f, indent=2)

@st.cache_data(show_spinner=False)
def load_dataset(ds_key):
    """Load dataset using configuration with caching"""
    cfg = DATASETS[ds_key]
    df = fetch_dataset(ds_key, cfg)
    return df

@st.cache_data(show_spinner=False)
def get_data_exploration(ds_key):
    """Get dataset with exploration plots with caching"""
    df = load_dataset(ds_key)
    target = DATASETS[ds_key]["target"]
    plots = create_data_exploration_plots(df, ds_key, target)
    return df, target, plots

@st.cache_data(show_spinner=False)
def load_model_comparison(ds_key):
    """Load model comparison data for a dataset with caching"""
    p = ART_DIR / ds_key / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_model_metadata(ds_key, model_key):
    """Load model metadata with caching"""
    p = ART_DIR / ds_key / model_key / "metrics.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_trained_model(ds_key, model_key):
    """Load a pre-trained model with caching"""
    p = ART_DIR / ds_key / model_key / "model.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)  
def load_trained_pipeline(ds_key, model_key):
    """Load a complete pipeline (preprocessor + model) for SHAP and predictions"""
    from sklearn.pipeline import Pipeline
    from core.data_processor import load_preprocessor
    
    # Load the core model
    model = load_trained_model(ds_key, model_key)
    if model is None:
        return None
        
    # Load the preprocessor
    preprocessor = load_preprocessor(ds_key, model_key)
    if preprocessor is None:
        # If no preprocessor, return the raw model
        return model
        
    # Create a full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def get_available_datasets():
    """Get list of available datasets"""
    return list(DATASETS.keys())

def get_dataset_info(ds_key):
    """Get dataset configuration"""
    return DATASETS.get(ds_key, {})
