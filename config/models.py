"""
Model configurations for ML Pipeline Explorer.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

MODEL_ZOO = {
    "logreg": {
        "model": LogisticRegression(
            max_iter=500, solver="liblinear", random_state=42
        ),
        "description": "Logistic regression baseline",
        "pros": ["Interpretable", "Fast", "Good baseline"],
        "cons": ["Linear decision boundary", "May underfit"],
        "best_for": "Linearly separable patterns",
    },
    "random_forest": {
        "model": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "description": "Random Forest (bagging trees)",
        "pros": ["Handles nonlinearity", "Robust", "Importance"],
        "cons": ["Less interpretable", "Bigger models"],
        "best_for": "Mixed tabular data with interactions",
    },
    "xgboost": {
        "model": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
        "description": "Gradient boosting (XGBoost)",
        "pros": ["Strong performance", "Handles missing", "Feature importance"],
        "cons": ["More complex", "Hyperparameter tuning"],
        "best_for": "Structured data competitions",
    },
}

ENSEMBLE_CONFIG = {
    "voting": "soft",
    "models": ["logreg", "random_forest", "xgboost"],
    "description": "Soft voting ensemble of all models",
    "pros": ["Combines strengths", "Reduces overfitting", "Often best performance"],
    "cons": ["Slower inference", "Less interpretable", "More complex"],
    "best_for": "Maximum predictive performance",
}
