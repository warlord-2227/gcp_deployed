import pickle
import sklearn
from google.oauth2 import service_account
from google.cloud import storage, pubsub_v1
import google.cloud.logging
import logging
from io import BytesIO
import utilities

# Define function to load pre-trained models/scalers/transformers etc.
def load_model(object_type,config):
    bucket_name = config['load_model']['bucket_name']
    if object_type == "model":
        blob_name = "pkl/merlin-prod-ad_scoring-meta-brkfst-1.pkl"
    elif object_type == "scaler":
        blob_name = "scaler/scaler.pkl"
    elif object_type == "transformer":
        blob_name = "transformer/yeo_johnson_transformer.pkl"
    else:
        raise ValueError("Invalid object type. Must be 'model', 'scaler', or 'transformer'.")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # Download and load the object/pkl file
    object_file_contents = blob.download_as_string()
    loaded_object = pickle.load(BytesIO(object_file_contents))

    return loaded_object

# Define function to make predictions
def predict(config,data):
    model = load_model("model",config)
    # Make predictions using the loaded model
    probabilities = model.predict_proba(data.drop(columns='ad_name'))
    predictions = model.predict(data.drop(columns='ad_name'))
    # Calculate scores based on probabilities
    scores = utilities.calculate_scores(probabilities, config)
    scores = {'ad_name':data['ad_name'],'ad_performance_score':scores}

    return predictions, probabilities, scores