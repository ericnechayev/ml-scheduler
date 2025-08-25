import os
import pytest
from fastapi.testclient import TestClient

from api.main import app
from scheduled_task.scheduled_task_utils.model_update_utils import encode_model_file_to_b64


MODELS_DIR = "scheduled_task/retrained_models"
AVAILABLE_MODELS = os.listdir(MODELS_DIR)


@pytest.mark.parametrize("test_model", AVAILABLE_MODELS)
def test_update_model(test_model, tmp_path):
    """
    Tests if the API is able to change the model served by sending a new model.
    The model is encoded into Base64, for transport purposes as JSON plain text.
    """
    
    model_b64 = encode_model_file_to_b64(MODELS_DIR, test_model)

    with TestClient(app) as client:
        response = client.post(
            "/update_model", 
            headers={"Content-Type": "application/json"}, 
            json={"modelFilename": test_model, "modelObject": model_b64}
        )
    expected_model_version = "1.1"
    expected_response = {
        "status": "success", 
        "updatedModelName": test_model, 
        "updatedModelVersion": expected_model_version
    }
    assert (response.status_code == 200) and (response.json() == expected_response)

