import base64
import joblib
import pickle

def encode_model_file_to_b64(models_dir, model_file, tmp_path=None, is_pickle=False):
    """Encodes model file to a transportable format"""
    # Load an existing trained model

    # Convert the pickle file to a joblib file
    if is_pickle:  
        with open(f"{models_dir}/{model_file}", "rb") as f:
            model = pickle.load(f)
        # tmp_path is used for PyTest to dump pickle files
        file_path = f"{tmp_path}/{model_file}"
        joblib.dump(model, file_path)
    else: # Otherwise read joblib from the re-trained models directory
        file_path = f"{models_dir}/{model_file}"            

    # Read joblib file and encode the model into a bytes object
    with open(file_path, "rb") as f:
        model_bytes = f.read()
    
    # Encode into Base64
    return base64.b64encode(model_bytes).decode("utf-8")


def get_current_model(client, api_url):
    """Send a GET request to API to learn what model it's currently using"""
    current_model_resp = client.get(f"{api_url}/current_model").json()
    current_model_file, current_model_version = (
        current_model_resp.get("currentModelName"),
        current_model_resp.get("currentModelVersion")
    )
    return current_model_file, current_model_version


def update_model_served(client, api_url, models_dir, model_file):
    """Request a change to the model being served by the API"""
    # Model is encoded into Base64, for transport purposes as JSON plain text
    model_b64 = encode_model_file_to_b64(models_dir, model_file)

    # Requesting a model update to the API
    response = client.post(
        f"{api_url}/update_model", 
        json={
            "modelFilename": model_file, 
            "modelObject": model_b64
        }
    )
    update_model_resp = response.json()
    updated_model_name, updated_model_version = (
        update_model_resp.get("updatedModelName"),
        update_model_resp.get("updatedModelVersion")
    )
    return updated_model_name, updated_model_version
