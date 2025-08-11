#!/usr/bin/env python3
"""
Quick script to check which models have been trained for each dataset.
Run this before using the UI to ensure artifacts exist.
"""

import os
from pathlib import Path

def check_trained_models():
    """Check which models have been trained for each dataset."""
    artifacts_dir = Path("artifacts")
    
    if not artifacts_dir.exists():
        print("❌ No artifacts directory found. Run trainer_refactored.py first.")
        return
    
    datasets = ["netflix", "stocks", "terrorism"]
    models = ["logreg", "random_forest", "xgboost"]
    
    print("🔍 Checking trained models...\n")
    
    all_ready = True
    for dataset in datasets:
        dataset_dir = artifacts_dir / dataset
        if not dataset_dir.exists():
            print(f"❌ {dataset}: No artifacts found")
            all_ready = False
            continue
            
        print(f"📊 {dataset.upper()}:")
        for model in models:
            model_dir = dataset_dir / model
            model_file = model_dir / "model.pkl"
            metrics_file = model_dir / "metrics.json"
            
            if model_file.exists() and metrics_file.exists():
                print(f"  ✅ {model}")
            else:
                print(f"  ❌ {model}")
                all_ready = False
        print()
    
    if all_ready:
        print("🎉 All models are trained and ready for the UI!")
    else:
        print("⚠️  Some models are missing. UI will show errors for missing models.")
    
    print("💡 To train missing models, run:")
    print("   python trainer_refactored.py")

if __name__ == "__main__":
    check_trained_models()
