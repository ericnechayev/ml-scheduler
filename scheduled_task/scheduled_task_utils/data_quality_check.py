import logging

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_test_set(X_test: np.ndarray, y_test: np.ndarray) -> bool:
    """
    Perform data quality and integrity checks for a multi-class test set.
    Parameters:
    - X_test: 2D numpy array of float features
    - y_test: 1D numpy array of integer labels (0, 1, 2)
    Returns: True if the test set is valid, meaning no issues, and otherwise False.
    """
    issues_found = []

    # Shape checks
    if X_test.ndim != 2:
        issues_found.append(f"X_test should be 2D, found shape: {X_test.shape}")
    if y_test.ndim != 1:
        issues_found.append(f"y_test should be 1D, found shape: {y_test.shape}")
    if X_test.shape[0] != y_test.shape[0]:
        issues_found.append(f"Number of samples mismatch: X_test={X_test.shape[0]}, y_test={y_test.shape[0]}")

    n_samples, n_features = X_test.shape

    # Missing values
    if np.isnan(X_test).any():
        issues_found.append("X_test contains NaN values")
    if np.isnan(y_test).any():
        issues_found.append("y_test contains NaN values")

    # Label validity
    unique_labels = np.unique(y_test)
    if not np.all(np.isin(unique_labels, [0, 1, 2])):
        issues_found.append(f"y_test contains invalid labels: {unique_labels}")
    
    # Check class balance
    class_counts = {label: int(np.sum(y_test == label)) for label in [0, 1, 2]}
    
    # Feature ranges
    X_min = np.min(X_test, axis=0)
    X_max = np.max(X_test, axis=0)

    # Warn if any feature has zero variance
    zero_variance_features = np.where(X_min == X_max)[0]
    if len(zero_variance_features) > 0:
        issues_found.append(f"Features with zero variance (columns): {zero_variance_features.tolist()}")

    # Summary report
    logger.info("\n===== Test Set Data Quality Report =====")
    logger.info(f"Number of samples: {n_samples}, Number of features: {n_features}")
    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Unique labels: {unique_labels}")
    logger.info(f"Feature ranges (min-max) per column:")
    for idx in range(n_features):
        logger.info(f"  Feature {idx}: {X_min[idx]:.4f} - {X_max[idx]:.4f}")

    if issues_found:
        logger.info("\nIssues found:")
        for issue in issues_found:
            logger.info(f" - {issue}")
    else:
        logger.info("\nNo issues detected. Test set passed integrity checks.\n")

    test_set_valid = True if not issues_found else False
    return test_set_valid