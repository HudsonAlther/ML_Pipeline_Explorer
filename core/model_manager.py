"""
Model training and management utilities.
"""

from config.models import MODEL_ZOO, ENSEMBLE_CONFIG

# Get a model instance by key
def get_model(model_key):
    return MODEL_ZOO[model_key]['model']

# Get ensemble configuration
def get_ensemble_config():
    return ENSEMBLE_CONFIG

# List available models
def list_models():
    return list(MODEL_ZOO.keys())
