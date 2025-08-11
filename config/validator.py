"""
Configuration validation for ML Pipeline Explorer.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from config.datasets import DATASETS
from config.models import MODEL_ZOO
from config.app_config import APP_CONFIG

class ConfigValidator:
    """Validator for application configuration"""
    
    @staticmethod
    def validate_datasets() -> Tuple[bool, List[str]]:
        """Validate dataset configurations"""
        errors = []
        
        for ds_key, ds_cfg in DATASETS.items():
            # Check required fields
            required_fields = ['path', 'target', 'description', 'business_value']
            for field in required_fields:
                if field not in ds_cfg:
                    errors.append(f"Dataset '{ds_key}' missing required field: {field}")
            
            # Check if dataset file exists
            if 'path' in ds_cfg:
                dataset_path = Path(ds_cfg['path'])
                if not dataset_path.exists():
                    errors.append(f"Dataset file not found: {ds_cfg['path']}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_models() -> Tuple[bool, List[str]]:
        """Validate model configurations"""
        errors = []
        
        for model_key, model_cfg in MODEL_ZOO.items():
            # Check required fields
            required_fields = ['model', 'description', 'pros', 'cons']
            for field in required_fields:
                if field not in model_cfg:
                    errors.append(f"Model '{model_key}' missing required field: {field}")
            
            # Check if model object is valid
            if 'model' in model_cfg:
                model = model_cfg['model']
                if not hasattr(model, 'fit'):
                    errors.append(f"Model '{model_key}' does not have 'fit' method")
                if not hasattr(model, 'predict'):
                    errors.append(f"Model '{model_key}' does not have 'predict' method")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_app_config() -> Tuple[bool, List[str]]:
        """Validate application configuration"""
        errors = []
        
        # Check artifact directory
        artifact_dir = Path(APP_CONFIG.get('artifact_dir', 'artifacts'))
        if not artifact_dir.exists():
            try:
                artifact_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create artifact directory: {e}")
        
        # Check other required config fields
        required_fields = ['artifact_dir']
        for field in required_fields:
            if field not in APP_CONFIG:
                errors.append(f"App config missing required field: {field}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_all() -> Tuple[bool, Dict[str, List[str]]]:
        """Validate all configurations"""
        results = {}
        
        # Validate datasets
        is_valid, errors = ConfigValidator.validate_datasets()
        results['datasets'] = errors
        
        # Validate models
        is_valid, errors = ConfigValidator.validate_models()
        results['models'] = errors
        
        # Validate app config
        is_valid, errors = ConfigValidator.validate_app_config()
        results['app_config'] = errors
        
        # Overall validation
        all_valid = all(len(errors) == 0 for errors in results.values())
        
        return all_valid, results

def validate_config():
    """Main validation function"""
    is_valid, results = ConfigValidator.validate_all()
    
    if not is_valid:
        print("Configuration validation failed:")
        for section, errors in results.items():
            if errors:
                print(f"\n{section.upper()}:")
                for error in errors:
                    print(f"  - {error}")
        return False
    
    print("Configuration validation passed!")
    return True 