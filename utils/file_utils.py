"""
File utility functions for ML Pipeline Explorer.
"""

import json
import joblib
import os
from pathlib import Path

def ensure_directory(path):
    """Ensure a directory exists, create it if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

# Example: save/load helpers
def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
