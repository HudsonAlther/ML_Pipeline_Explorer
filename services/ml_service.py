"""
Service layer for ML operations.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from core.data_handler import (
    load_dataset, get_data_exploration, load_model_comparison,
    load_model_metadata, load_trained_model, get_available_datasets,
    get_dataset_info, save_trained_model, save_model_metadata
)
from core.model_manager import get_model
from core.data_processor import create_preprocessing_pipeline, load_preprocessor, load_feature_names, preprocess_new_data
from utils.metrics import compute_metrics, compute_predictions
from config.datasets import DATASETS
from config.models import MODEL_ZOO

class MLService:
    """Service class for ML operations"""
    
    def __init__(self):
        self.data_handler = None  # Will be initialized when needed
    
    def get_datasets(self) -> Dict[str, Dict]:
        """Get available datasets"""
        return DATASETS
    
    def get_models(self) -> Dict[str, Dict]:
        """Get available models"""
        return MODEL_ZOO
    
    def load_dataset_data(self, ds_key: str) -> Tuple[pd.DataFrame, str, Dict]:
        """Load dataset with exploration data"""
        return get_data_exploration(ds_key)
    
    def get_model_comparison(self, ds_key: str) -> Optional[pd.DataFrame]:
        """Get model comparison data for dataset"""
        return load_model_comparison(ds_key)
    
    def get_model_metadata(self, ds_key: str, model_key: str) -> Optional[Dict]:
        """Get model metadata"""
        return load_model_metadata(ds_key, model_key)
    
    def load_model(self, ds_key: str, model_key: str):
        """Load trained model"""
        return load_trained_model(ds_key, model_key)
    
    # Training is now handled by trainer_refactored.py CLI script
    # UI only loads pre-trained artifacts for security and separation of concerns
    
    def make_prediction(self, model, input_data: Dict, ds_key: str, model_key: str) -> Dict[str, Any]:
        """Make prediction with model using consistent preprocessing"""
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Load preprocessor and feature names
            preprocessor = load_preprocessor(ds_key, model_key)
            feature_names = load_feature_names(ds_key, model_key)
            
            if preprocessor is None or feature_names is None:
                return {
                    "success": False,
                    "error": "Preprocessor not found. Please retrain the model."
                }
            
            # Preprocess the input data
            processed_input = preprocess_new_data(input_df, preprocessor, feature_names)
            
            if processed_input is None:
                return {
                    "success": False,
                    "error": "Failed to preprocess input data"
                }
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(processed_input)[0]
            
            return {
                "success": True,
                "prediction": prediction,
                "probability": probability,
                "confidence": max(probability) if probability is not None else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_dataset_info(self, ds_key: str) -> Dict:
        """Get dataset information"""
        return get_dataset_info(ds_key)
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get model information"""
        return MODEL_ZOO.get(model_key, {})
    
    def validate_dataset(self, ds_key: str) -> bool:
        """Validate if dataset exists and is accessible - optimized for speed"""
        try:
            # Just check if the file exists instead of loading the entire dataset
            from config.datasets import DATASETS
            if ds_key not in DATASETS:
                return False
            
            cfg = DATASETS[ds_key]
            file_path = cfg['path']
            
            # Check if file exists
            from pathlib import Path
            if not Path(file_path).exists():
                return False
                
            return True
        except Exception:
            return False
    
    def validate_model(self, model_key: str) -> bool:
        """Validate if model exists"""
        return model_key in MODEL_ZOO 