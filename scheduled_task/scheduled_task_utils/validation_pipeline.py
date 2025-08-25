import logging

from scheduled_task_utils.health_check import ping_api_health
from scheduled_task_utils.data_quality_check import check_test_set
from scheduled_task_utils.latency_check import sample_predict_requests, measure_prediction_latency
from scheduled_task_utils.drift_check import monitor_label_drift


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def perform_routine_checks(api_url, test_features, test_labels, test_set):
    """Basic checks for dataset quality, API performance and model drift"""
    # Ping health check with retries
    if not ping_api_health(f"{api_url}/", retries=3, delay=2):
        logger.warning("API health check failed after retries. Skipping this job run.")
        return

    # Run data quality checks on test set
    test_set_valid = check_test_set(test_features, test_labels)
    if not test_set_valid:
        logger.warning("Test set issues detected. Skipping model evaluation.")
        return
  
    predictions, latencies = sample_predict_requests(test_set=test_set, api_url=api_url)

    # Test predict requests for P50 and P95 latency
    passing_latency_check = measure_prediction_latency(latencies)
    if not passing_latency_check:
      logger.warning("Latency is too high. Skipping model evaluation.")
      return
    
    # Check current model for label drift
    label_drift_detected = monitor_label_drift(test_labels=test_labels, predictions=predictions, alpha=0.05)
    if label_drift_detected:
        logger.warning("Label drift detected - consider monitoring before updating model.")

