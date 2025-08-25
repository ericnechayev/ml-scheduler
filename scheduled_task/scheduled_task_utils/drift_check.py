from scipy.stats import ks_2samp


def monitor_label_drift(test_labels, predictions, alpha=0.05) -> bool:
    """
    Check for feature drift and label/prediction drift.
    
    Parameters:
    - test_labels: reference labels
    - predictions: model predictions for production observations
    - alpha: significance level for KS test
    """
    _, p_value = ks_2samp(test_labels, predictions)
    drift_detected = (p_value < alpha)

    return drift_detected
