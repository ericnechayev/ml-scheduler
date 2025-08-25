import os
import random
import httpx
import logging

import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler

from api.iris_config import FEATURE_NAMES
from scheduled_task_utils.evaluation_utils import assess_model_update
from scheduled_task_utils.model_update_utils import get_current_model, update_model_served
from scheduled_task_utils.validation_pipeline import perform_routine_checks


API_URL = "http://api:8000"  # Service name "api" from docker-compose

BASE_DIR = "/app/scheduled_task"
MODELS_DIR, DATA_PATH = f"{BASE_DIR}/retrained_models", f"{BASE_DIR}/test_dataset"

TEST_FEATURES = np.load(f'{DATA_PATH}/X_test.npy')
TEST_LABELS = np.load(f'{DATA_PATH}/y_test.npy')
TEST_SET = [dict(zip(FEATURE_NAMES, row)) for row in TEST_FEATURES]

RETRAINING_SEC_INTERVAL = 10  # Used During Development
RETRAINING_HOURS_INTERVAL = 24   # Example Re-training Frequency in Production


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def scheduled_retraining():
    """
    A simulated re-training task that runs regularly, conducts tests, 
    and deploys a model to replace the model served in the API when needed.
    """
    logging.info("Starting scheduled re-training job...")
    # Checking API health, data quality, P95 & P50 latency, and label drift
    # Separated into validation_pipeline.py for readability, see file
    perform_routine_checks(API_URL, TEST_FEATURES, TEST_LABELS, TEST_SET)

    with httpx.Client(timeout=5) as client:

        # Get current model from API
        current_model_file, current_model_version = get_current_model(client, API_URL)
        logger.info(f"\nCurrent model name: {current_model_file}\nCurrent model version: {current_model_version}\n")
        
        # Pick a random candidate model
        candidate_models = [file for file in os.listdir(MODELS_DIR) if file != current_model_file]
        candidate_model_file = random.choice(candidate_models)
        logger.info(f"\nCandidate model chosen: {candidate_model_file}")

        # Examine the randomly chosen model's performance
        update_model = assess_model_update(
            MODELS_DIR, 
            current_model_file, 
            candidate_model_file,
            TEST_FEATURES, 
            TEST_LABELS
        )
        # Perform a model update if it qualifies
        if update_model:
            logger.info(f"\nCandidate {candidate_model_file} exceeds primary metric threshold. Updating API.")
            # Sending the request to update the model being served
            updated_model_name, updated_model_version = update_model_served(
                client, 
                API_URL, 
                MODELS_DIR, 
                candidate_model_file
            )
            logger.info(f"\nUpdated model name: {updated_model_name}\nUpdated model version: {updated_model_version}\n")
        else:
            # Keep the model the same, since the metrics were not good enough
            logger.info(f"\nCandidate {candidate_model_file} did not surpass primary metric. Keeping current model.")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # scheduler.add_job(scheduled_retraining, "interval", hours=RETRAINING_HOURS_INTERVAL)
    scheduler.add_job(scheduled_retraining, "interval", seconds=RETRAINING_SEC_INTERVAL)

    logging.info("Scheduler started. Waiting for jobs...")
    scheduler.start()
