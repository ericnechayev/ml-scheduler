from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD_EXAMPLE = {
    "sepallength": 5.1,
    "sepalwidth": 3.5,
    "petallength": 1.4,
    "petalwidth": 0.00000001
}

INVALID_PAYLOAD_EXAMPLE = {
    "sepallength": 5.1,
    "sepalwidth": 3.5,
    "petallength": 1.4,
    "petalwidth": -1
}

EXPECTED_VALID_RESPONSE = {
    "species": 0,
    "model": "rf-12-base"
}

EXPECTED_INVALID_RESPONSE = {
    'species': None, 
    'model': 'rf-12-base',
    'errorMessage': 'Please ensure all feature values provided are positive numbers.', 
    'errorDetails': [{'field': 'petalwidth', 'message': 'Input should be greater than 0'}]
}


def test_valid_response():
    test_example = VALID_PAYLOAD_EXAMPLE
    with TestClient(app) as client:
        response = client.post("/predict", headers={"Content-Type": "application/json"}, json=test_example)
    assert (response.status_code == 200) and (response.json() == EXPECTED_VALID_RESPONSE)


def test_invalid_response():
    test_example = INVALID_PAYLOAD_EXAMPLE
    with TestClient(app) as client:
        response = client.post("/predict", headers={"Content-Type": "application/json"}, json=test_example)
    assert (response.status_code == 400) and (response.json() == EXPECTED_INVALID_RESPONSE)
