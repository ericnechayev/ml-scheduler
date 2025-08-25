## Project File Structure

```
├── api/ # FastAPI App (Main API Service)
│ |
│ ├── main.py # Entry point for API having all routes
│ ├── iris_config.py # Schema of incoming prediction requests
│ ├── serving_utils.py # Utility functions for serving models
│ ├── prod_models/ # Folder to save & load production models
│ └── tests/ # PyTest Unit Tests testing API functionality
│
|
├── scheduled_task/ # Scheduled Task Service
| |
│ ├── scheduler_service.py # Main scheduled task runner
│ └── scheduled_task_utils/ # Re-training validation checks
│ └── retrained_models/ # Candidate models for updating API
│ └── test_data/ # Test dataset to validate models
|
|
├── requirements.txt # Python 3.10 dependencies
├── Dockerfile # Container build definition
├── docker-compose.yml # Multi-service orchestration
```

## App Components

- **API Service**

  - Runs a FastAPI application exposing ML prediction endpoints.
  - Uses Pydantic models for request validation and schema docs.
  - Includes unit tests under api/tests/, including for:
    - Model Initialization
    - Model Serialization
    - Model Updating
    - Prediction Requests
    - API Responses

- **Scheduler Service**:
  - Runs independently of the API as a separate container.
  - Simulates re-trainings that consider updates to the API model being served.
  - Checks data quality, API health & response latency, and ML evaluation metrics before updating models.
  - If metrics are satisfactory, requests an update to the model served by sending a new model object.
