import time
import httpx
import logging
from typing import Tuple

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sample_predict_requests(test_set, api_url) -> Tuple[list, list]:
    """Sends prediction requests for an entire test dataset 
    to gather a sample of predictions and latencies."""
    predictions, latencies = [], []

    with httpx.Client(timeout=5) as client:
        for payload in test_set:
            start_time = time.time()
            response = client.post(f"{api_url}/predict", json=payload)
            end_time = time.time()

            latency = end_time - start_time
            prediction = response.json()["species"]
            latencies.append(latency)
            predictions.append(prediction)

            if response.status_code != 200:
                logger.warning(f"Prediction failed for sample {payload}: {response.status_code}, {response.text}")

    latencies = [lat * 1000 for lat in latencies] # Convert to milliseconds

    return predictions, latencies


def measure_prediction_latency(latencies, max_p50=50, max_p95=100) -> bool:
    """
    Send each sample individually to the /predict endpoint and measure latency.
    Return True if latency is acceptable, False otherwise.
    - max_p50: maximum median latency in milliseconds
    - max_p95: maximum 95th percentile latency in milliseconds
    """
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)

    logger.info(f"\nMedian (P50) latency: {median_latency:.4f} ms")
    logger.info(f"P95 latency: {p95_latency:.4f} ms\n")

    if median_latency > max_p50:
        logger.warning(f"Median (P50) latency too high: {median_latency:.4f}s (threshold {max_p50} ms)")
        return False
    if p95_latency > max_p95:
        logger.warning(f"P95 latency too high: {p95_latency:.4f}s (threshold {max_p95} ms)")
        return False
    return True