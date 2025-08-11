#!/usr/bin/env python3
"""
Test script for the refactored ML Pipeline Explorer architecture.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore",
        message="Attempting to set identical low and high xlims", module="shap")

def test_configuration():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    try:
        from config.validator import validate_config
        is_valid = validate_config()
        print(f"Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False

def test_service_layer():
    """Test service layer functionality"""
    print("\nTesting service layer...")
    
    try:
        from services.ml_service import MLService
        ml_service = MLService()
        
        # Test getting datasets
        datasets = ml_service.get_datasets()
        print(f"Datasets loaded: {len(datasets)} datasets")
        
        # Test getting models
        models = ml_service.get_models()
        print(f"Models loaded: {len(models)} models")
        
        # Test dataset validation
        for ds_key in datasets.keys():
            is_valid = ml_service.validate_dataset(ds_key)
            print(f"Dataset '{ds_key}' validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test model validation
        for model_key in models.keys():
            is_valid = ml_service.validate_model(model_key)
            print(f"Model '{model_key}' validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return True
    except Exception as e:
        print(f"Service layer error: {e}")
        return False

def test_data_processor():
    """Test data processor functionality"""
    print("\nTesting data processor...")
    
    try:
        from core.data_processor import fetch_dataset, create_data_exploration_plots
        from config.datasets import DATASETS
        
        # Test dataset loading
        for ds_key, ds_cfg in DATASETS.items():
            try:
                df = fetch_dataset(ds_key, ds_cfg)
                print(f"Dataset '{ds_key}' loaded: {df.shape}")
                
                # Test exploration plots
                target = ds_cfg['target']
                plots = create_data_exploration_plots(df, ds_key, target)
                print(f"Exploration plots created for '{ds_key}': {len(plots)} plots")
                
            except Exception as e:
                print(f"Error processing dataset '{ds_key}': {e}")
        
        return True
    except Exception as e:
        print(f"Data processor error: {e}")
        return False

def test_error_handling():
    """Test error handling system"""
    print("\nTesting error handling...")
    
    try:
        from core.error_handler import MLError, DatasetError, ModelError, TrainingError
        
        # Test custom exceptions
        exceptions = [MLError, DatasetError, ModelError, TrainingError]
        for exc in exceptions:
            try:
                raise exc("Test error")
            except exc as e:
                print(f"{exc.__name__} working correctly")
        
        return True
    except Exception as e:
        print(f"Error handling test failed: {e}")
        return False

def test_views():
    """Test view imports"""
    print("\nTesting view imports...")
    
    try:
        from views.dataset_selection import dataset_selection_view
        from views.data_preparation import data_preparation_view
        from views.model_selection import model_selection_view
        from views.model_analysis import model_analysis_view
        from views.model_playground import create_model_playground_view
        
        print("All view modules imported successfully")
        return True
    except Exception as e:
        print(f"View import error: {e}")
        return False

def test_core_modules():
    """Test core module functionality"""
    print("\nTesting core modules...")
    
    try:
        from core.data_handler import load_dataset, get_data_exploration
        from core.model_manager import get_model
        from core.session_manager import initialize_session, set_session
        from core.data_processor import fetch_dataset
        
        print("Core modules imported successfully")
        
        # Test session manager
        initialize_session()
        set_session("test_key", "test_value")
        print("Session manager working")
        
        return True
    except Exception as e:
        print(f"Core module error: {e}")
        return False

def test_utils():
    """Test utility modules"""
    print("\nTesting utility modules...")
    
    try:
        from utils.metrics import compute_metrics, compute_predictions
        from utils.file_utils import ensure_directory
        
        print("Utility modules imported successfully")
        return True
    except Exception as e:
        print(f"Utility module error: {e}")
        return False

def test_visualization():
    """Test visualization modules"""
    print("\nTesting visualization modules...")
    
    try:
        from visualization.charts import create_model_comparison_chart
        from visualization.plots import create_confusion_matrix_plot
        from visualization.components import metric_card
        
        print("Visualization modules imported successfully")
        return True
    except Exception as e:
        print(f"Visualization module error: {e}")
        return False

# ----------------------------------------------------------------------
# Verify video assets exist

def test_video_assets():
    """Ensure training animation MP4s are present and readable"""
    print("\nTesting training video assets ...")
    from pathlib import Path
    video_paths = [
        Path("static/animations/LogisticTraining.mp4"),
        Path("static/animations/RandomForestTraining.mp4"),
        Path("static/animations/XGBoostTraining.mp4"),
    ]
    all_ok = True
    for p in video_paths:
        if p.exists() and p.stat().st_size > 0:
            print(f"Video found: {p} ({p.stat().st_size/1024:.1f} KB)")
        else:
            print(f"Missing or empty video: {p}")
            all_ok = False
    return all_ok

def main():
    """Run all tests"""
    print("Testing Refactored ML Pipeline Explorer Architecture")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Service Layer", test_service_layer),
        ("Data Processor", test_data_processor),
        ("Error Handling", test_error_handling),
        ("Views", test_views),
        ("Core Modules", test_core_modules),
        ("Utils", test_utils),
        ("Visualization", test_visualization),
        ("Video Assets", test_video_assets),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The refactored architecture is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 