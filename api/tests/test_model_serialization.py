import joblib
import os
import pytest
import numpy as np


MODELS_DIR, DATA_PATH = "api/prod_models", "api/tests/data"
AVAILABLE_MODELS = os.listdir(MODELS_DIR)
TEST_FEATURES = np.load(f'{DATA_PATH}/X_test.npy')


@pytest.mark.parametrize("model_filename", AVAILABLE_MODELS)
def test_model_serialization(model_filename, tmp_path):
    """Ensure model serialization does not alter prediction"""
    model = joblib.load(f"{MODELS_DIR}/{model_filename}")

    preds_before = model.predict_proba(TEST_FEATURES)
    
    filepath = f"{tmp_path}/{model_filename}"
    joblib.dump(model, filepath)
    
    loaded_model = joblib.load(filepath)
    
    preds_after = loaded_model.predict_proba(TEST_FEATURES)
    
    # Assert predictions are within a certain tolerance
    assert np.allclose(preds_before, preds_after, atol=1e-6)
