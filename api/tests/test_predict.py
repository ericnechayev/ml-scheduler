import os
import pytest

from fastapi.testclient import TestClient
import numpy as np

from api.main import app
from api.schema_config import FEATURE_NAMES


MODELS_DIR = "api/prod_models"
AVAILABLE_MODELS = [model.split(".")[0] for model in os.listdir(MODELS_DIR)]

DATA_PATH = "api/tests/data"
TEST_FEATURES = np.load(f'{DATA_PATH}/X_test.npy')
TEST_SET = [dict(zip(FEATURE_NAMES, row)) for row in TEST_FEATURES]


@pytest.mark.parametrize("feature_values", TEST_SET)
def test_predict(feature_values):
    with TestClient(app) as client:
        response = client.post("/predict", headers={"Content-Type": "application/json"}, json=feature_values)

    prediction, model_used = response.json()["species"], response.json()["model"]

    assert (
        (response.status_code == 200) and \
        (type(prediction) == int) and (type(model_used) == str) and \
        (prediction in [0, 1, 2]) and \
        (model_used in AVAILABLE_MODELS)
    )
