import os
from fastapi.testclient import TestClient

from api.main import app, DEFAULT_MODEL_FILE


MODELS_DIR = "api/prod_models"
AVAILABLE_MODELS = os.listdir(MODELS_DIR)

def test_default_model_initialized():
  """Ensure there is a default model initialized after application startup"""
  with TestClient(app) as client:
      response = client.get("/current_model", headers={"Content-Type": "application/json"}).json()
  
  assert (
    (response["currentModelName"] == DEFAULT_MODEL_FILE) and \
    (response["currentModelVersion"] == "1.0")
  )
  