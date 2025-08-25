import base64
import io
from typing import Any
import joblib

from fastapi import HTTPException


def load_model(models_dir, model_filename: str) -> Any:
  """Load a model using joblib from the container model directory."""
  try:
    return joblib.load(f"{models_dir}/{model_filename}")
  except FileNotFoundError:
    raise HTTPException(status_code=404, detail=f"Model file '{model_filename}' not found")
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


def save_retrained_model(model_object, models_dir, model_filename):
    """When a re-trained model is sent to the API, 
    it should be saved as a backup copy for re-loading."""
    # Decode base64 back into bytes
    model_bytes = base64.b64decode(model_object)
    # Load into sklearn using BytesIO
    new_model = joblib.load(io.BytesIO(model_bytes))
    joblib.dump(new_model, f"{models_dir}/{model_filename}")


def increment_model_version(version):
    major, minor = map(int, version.split("."))
    minor += 1
    return f"{major}.{minor}"