import joblib
import json
import logging
import os
from typing import Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss
)

# Define model update rule
PRIMARY_METRIC = "f1_macro" # Good metric when classes are balanced
DELTA_THRESHOLD = 0.001  # Only update if candidate is meaningfully better
ALWAYS_UPDATE_MODEL = True  # Use if we want to replace the model no matter what

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, test_features, test_labels) -> Tuple[dict, dict]:
    """Load model and compute metrics on test set."""
    model = joblib.load(model_path)
    predictions = model.predict(test_features)

    # For log_loss we need probabilities
    y_proba = model.predict_proba(test_features) if hasattr(model, "predict_proba") else None

    metrics = {
        "f1_macro": f1_score(test_labels, predictions, average="macro"),
        "accuracy": accuracy_score(test_labels, predictions),
        "precision_macro": precision_score(test_labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(test_labels, predictions, average="macro", zero_division=0),
        "log_loss": log_loss(test_labels, y_proba) if y_proba is not None else None
    }
    rounded_metrics = {k: round(v, 2) if v is not None else None for k, v in metrics.items()}
    return metrics, json.dumps(rounded_metrics, indent=4)


def assess_model_update(
        models_dir, 
        current_model_file, 
        candidate_model_file,
        test_features, 
        test_labels
    ) -> dict:
    """Decide if we should update the model based on how much better we need a metric to be."""
    # Evaluating the current model
    current_metrics, current_metrics_rounded = evaluate_model(
        os.path.join(models_dir, current_model_file), test_features, test_labels
    )
    logger.info(f"\nCurrent model {candidate_model_file} metrics:\n{current_metrics_rounded}\n")
    
    # Evaluating the randomly chosen candidate model
    candidate_metrics, candidate_metrics_rounded = evaluate_model(
        os.path.join(models_dir, candidate_model_file), test_features, test_labels
    )
    logger.info(f"\nCandidate model {candidate_model_file} metrics:\n{ candidate_metrics_rounded }\n")

    # Assessing if we should replace the served model
    update_model = True if ALWAYS_UPDATE_MODEL else (
        candidate_metrics[PRIMARY_METRIC] > current_metrics[PRIMARY_METRIC] + DELTA_THRESHOLD
    )
    return update_model
