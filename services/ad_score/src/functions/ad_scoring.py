import lib_main as training
import predict as testing
import pre_process
import utilities
from google.cloud import storage, pubsub_v1
import google.cloud.logging
import logging
import json
import base64

class ad_scoring_model():
    def __init__(self):
        storage_client = storage.Client()
        bucket = storage_client.bucket("sd_storage-ad_score-meta-dev")
        blob = bucket.blob("config/config.json")
        json_data = (blob.download_as_string()).decode('utf-8')
        self.config = json.loads(json_data)
        self.version = self.config['version']
        print("Running Version..... ",self.version)
    
    def training_model(self,dataset):
        self.model_performance = training.pre_process_build_model(self.config,dataset)

        return self.model_performance

    def predicting_model(self,event):
        """to get the predictions, probabilities and scores by ad_scoring_model.dict_metrics"""
        logging.getLogger().setLevel(logging.INFO)
        pubsub_message = base64.b64decode(event["data"]).decode('utf-8')
        input_data = json.loads(pubsub_message)
        logging.info(f"Keys in it {input_data.keys()}")
        self.raw_input_json = input_data
        print('Saving Input Data to Bucket')
        utilities.upload_to_storage(input_data,'input',self.config)
        logging.info(f"Received msg:{type(input_data)}")
        print('Pub-Sub Format.......')
        dataset = pre_process.pre_process_json(input_data)
        predictions, probabilities, scores = testing.predict(self.config,dataset)
        self.dict_metrics = {'predictions':predictions,'probabilities':probabilities,'scores':scores}
        json_output = json.loads(utilities.post_process_results(self.raw_input_json,scores))
        
        # Pushing to Pub-Sub and Uploading to Bucket
        logging.info(f"output is :{json_output}")
        logging.info("into the main function")
        utilities.upload_to_storage(json_output,'output',self.config)
        logging.info("Output json pushed to cloud storage bucket")
        # Pushing to Pub-Sub
        publisher = pubsub_v1.PublisherClient()
        result_topic = "projects/within-merlin/topics/pubsub-ad_score-meta-dev"
        future = publisher.publish(result_topic, data=json.dumps(json_output).encode('utf-8'))
        logging.info("Output json pushed to Pub/Sub")
        return json_output