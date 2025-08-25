from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schema_config import Iris, FEATURE_NAMES
from api.serving_utils import load_model, save_retrained_model, increment_model_version


MODELS_DIR = "api/prod_models"
DEFAULT_MODEL_FILE = "rf-12-base.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class UpdateModelRequest(BaseModel):
    modelFilename: str
    modelObject: str


def set_served_model(model_filename, default=False):
    """Set model state during app initialization and scheduled model updates."""
    app.state.model = load_model(MODELS_DIR, model_filename)
    app.state.model_file = model_filename
    version = increment_model_version(app.state.model_version) if not default else "1.0"
    app.state.model_version = version
    logger.info(f"\nAPI model set to {model_filename} with model version {version}\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Allows app state to last through the lifespan of the application."""
    def load_default_model():
        try:
          # Load a default model at application startup
          set_served_model(DEFAULT_MODEL_FILE, default=True)
          logger.info(f"\nDefault model loaded from: {app.state.model_file}")
        except Exception as e:
          logger.info("Failed to load default model:", e)
          app.state.model, app.state.model_file, app.state.model_version = None, None, None
    load_default_model()
    yield
    # Clean up the last loaded model
    app.state.model, app.state.model_file, app.state.model_version = None, None, None


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health_check():
    """Ensures the API is able to receive requests well."""
    return JSONResponse(status_code=200, content={"health_check": "success"})


@app.get("/current_model")
def get_current_model() -> dict:
    """Returns name of current model set in the application state."""
    if not hasattr(app.state, "model") or app.state.model is None:
        return {"currentModel": None}
    return JSONResponse(
        status_code=200, 
        content={
            "currentModelName": app.state.model_file,
            "currentModelVersion": app.state.model_version
        }
    )


@app.post("/update_model")
async def update_model(request: UpdateModelRequest) -> dict:
    """Allowed scheduled re-training tasks to change the model this API serves."""
    try:
        # Save a copy in case it needs to be re-loaded
        save_retrained_model(request.modelObject, MODELS_DIR, request.modelFilename)
        # Change the API state to serve this new model
        set_served_model(request.modelFilename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    status = "success" if app.state.model_file == request.modelFilename else "failure"

    return JSONResponse(
        status_code=200,
        content={
            "status": status,
            "updatedModelName": app.state.model_file,
            "updatedModelVersion": app.state.model_version
        }
    )

@app.exception_handler(RequestValidationError)
def input_error_response(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Notify the user that the input format was unable to be processed"""
    model_name = app.state.model_file.split(".")[0]
    
    error_msg = "Please ensure all feature values provided are positive numbers."
    formatted_errors = [
        {"field": ".".join(str(loc) for loc in e["loc"] if loc != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={
            "species": None,
            "model": model_name,
            "errorMessage": error_msg,
            "errorDetails": formatted_errors
        }
    )


@app.post("/predict")
async def predict(features: Iris) -> dict:
    """
    Predicts Iris species.
    Checks if JSON input follows Iris schema defined by Pydantic.
    If so, it preprocesses and predicts the observation.
    Otherwise, it sends a error specifying what to correct.
    """
    # Check if any model was loaded
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    # Preprocess observation
    observation = np.array([[getattr(features, i) for i in FEATURE_NAMES]])

    # Perform prediction
    model_name = app.state.model_file.split(".")[0]
    prediction = int(app.state.model.predict(observation)[0])

    return JSONResponse(
        status_code=200,
        content={
            "species": prediction,
            "model": model_name
        }
    )
