"""
Main entry point for the refactored ML training pipeline (CLI/script usage).
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from services.ml_service import MLService
from core.error_handler import handle_training_error, TrainingError
from config.validator import validate_config

@handle_training_error
def train_and_evaluate(ds_key, model_key, test_size=0.2, random_state=42):
    """Train and evaluate a model using the service layer"""
    print(f"\n=== Training: Dataset '{ds_key}' | Model '{model_key}' ===")
    
    # Initialize service
    ml_service = MLService()
    
    # Validate inputs
    if not ml_service.validate_dataset(ds_key):
        raise TrainingError(f"Dataset '{ds_key}' is not valid or accessible")
    
    if not ml_service.validate_model(model_key):
        raise TrainingError(f"Model '{model_key}' is not valid")
    
    # Train model using service
    result = ml_service.train_model(ds_key, model_key, test_size, random_state)
    
    if not result["success"]:
        raise TrainingError(f"Training failed: {result['error']}")
    
    # Print results
    print(f"\nTraining completed successfully!")
    print(f"Test Set Metrics:")
    for metric, value in result["metrics"].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"Model '{model_key}' trained and saved for dataset '{ds_key}'.")
    return result

def main():
    """Main training function with CLI interface"""
    parser = argparse.ArgumentParser(description="Train and evaluate ML model.")
    parser.add_argument("--dataset", help="Dataset key (as in config.datasets)")
    parser.add_argument("--model", help="Model key (as in config.models)")
    parser.add_argument("--test-size", type=float, default=0.2, 
                       help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate configuration before training")
    
    args = parser.parse_args()
    
    # Validate configuration if requested
    if args.validate:
        print("Validating configuration...")
        if not validate_config():
            print("Configuration validation failed. Please fix errors before training.")
            return
    
    # Initialize service to get available options
    ml_service = MLService()
    datasets = list(ml_service.get_datasets().keys())
    models = list(ml_service.get_models().keys())
    
    # Determine what to train
    datasets_to_train = [args.dataset] if args.dataset else datasets
    models_to_train = [args.model] if args.model else models
    
    print(f"Training {len(models_to_train)} model(s) on {len(datasets_to_train)} dataset(s)")
    
    # Train all combinations
    for ds_key in datasets_to_train:
        for model_key in models_to_train:
            try:
                train_and_evaluate(
                    ds_key, 
                    model_key, 
                    test_size=args.test_size,
                    random_state=args.random_state
                )
            except TrainingError as e:
                print(f"Training error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

if __name__ == "__main__":
    main()
