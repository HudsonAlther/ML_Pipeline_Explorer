"""
Custom error handling for ML Pipeline Explorer.
"""

class MLError(Exception):
    """Base exception for ML Pipeline errors"""
    pass

class DatasetError(MLError):
    """Exception raised for dataset-related errors"""
    pass

class ModelError(MLError):
    """Exception raised for model-related errors"""
    pass

class ValidationError(MLError):
    """Exception raised for validation errors"""
    pass

class TrainingError(MLError):
    """Exception raised for training errors"""
    pass

def handle_training_error(func):
    """Decorator to handle training errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}")
    return wrapper

def handle_dataset_error(func):
    """Decorator to handle dataset errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise DatasetError(f"Dataset operation failed: {str(e)}")
    return wrapper

def handle_model_error(func):
    """Decorator to handle model errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise ModelError(f"Model operation failed: {str(e)}")
    return wrapper 