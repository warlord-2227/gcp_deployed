import numpy as np
import pandas as pd
import json
import pickle
import sklearn
import datetime as dt
from google.cloud import storage, pubsub_v1
import google.cloud.logging
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
import pre_process

def split_data(data, config):
    y = data['Label']
    X = data.drop(['Label'], axis=1).replace(np.nan, 0)
    test_size = config['split_data']['test_size']
    random_state = config['split_data']['random_state']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, config):
    n_estimators = config['train_model']['n_estimators']
    random_state = config['train_model']['random_state']
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, cm, cr

def save_model(model, config):
    file_path = config['save_model']['model_file_path']
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def build_model(file_path,config):
    data = pre_process.preprocess_raw_data(file_path)
    data_1 = data.copy()

    columns_to_remove = []
    columns_to_remove += [col for col in data.columns if 'Variance' in col]
    columns_to_remove += [col for col in data.columns if 'HP' in col or 'Hook' in col]
    data = data.drop(columns=columns_to_remove)

    data = pre_process.transform_campaign_objective(data)
    data = pre_process.tag_audience(data)
    data = pre_process.encode_target_labels(data)

    X_train, X_test, y_train, y_test = split_data(data,config)

    model = train_model(X_train, y_train,config)
    accuracy, cm, cr = evaluate_model(model, X_test, y_test)

    save_model(model,config)

    return accuracy, cm

def scale(value, original_min, original_max, new_min, new_max):
    return new_min + (new_max - new_min) * (value - original_min) / (original_max - original_min)

# Define function to scale probabilities and calculate scores
def calculate_scores(probabilities, config):
    conf_good_okay = config['calculate_scores']['conf_good_okay']
    conf_good = config['calculate_scores']['conf_good']
    case_1_lb = config['calculate_scores']['case_1_lb']
    case_1_ub = config['calculate_scores']['case_1_ub']
    case_2_lb = config['calculate_scores']['case_2_lb']
    case_2_ub = config['calculate_scores']['case_2_ub']
    case_3_lb = config['calculate_scores']['case_3_lb']
    case_3_ub = config['calculate_scores']['case_3_ub']

    final_scores = []
    for prob in probabilities:
        prob_good, prob_okay, prob_bad = prob
        sum_prob_good_okay = prob_good + prob_okay
        
        if sum_prob_good_okay > conf_good_okay and prob_good > conf_good:
            score = scale(prob_good, conf_good, 1, case_1_lb, case_1_ub)
        elif sum_prob_good_okay > conf_good_okay and prob_good <= conf_good:
            score = scale(prob_good + prob_okay, conf_good_okay, 1, case_2_lb, case_2_ub)
        elif sum_prob_good_okay >= 0.55 and sum_prob_good_okay < conf_good_okay:
            score = scale(prob_good + prob_okay, 0.55, conf_good_okay, case_3_lb, case_3_ub)
        else:
            score = sum_prob_good_okay

        # Convert the score to a percentage integer
        score = int(round(score * 100))
        final_scores.append(score)

    return final_scores

# Define function to pre-process the json and make it prediction ready
def post_process_results(input_data,output_data_dict):
    project_name = input_data['project']
    model_name = input_data['model']
    environment_name = input_data['environment']
    data_state = input_data['data_state']
    platform_name = input_data['platform']
    account_data = input_data['account_details']
    model_name_output_str = input_data['model']
    converted_data = {
        "project": project_name,
        "account_details": account_data,
        "model": model_name_output_str,
        "environment": environment_name,
        "data_state": "scored",
        "platform": platform_name,
        "payload": {
            "data": [
                {
                    "ads": []
                }
            ]
        }
    }

    # Appending ads data to the payload

    for i in range(len(output_data_dict['ad_name'])):
        converted_data["payload"]["data"][0]["ads"].append({
            "ad_id": output_data_dict['ad_name'][i],
            "ad_performance_score": output_data_dict['ad_performance_score'][i]
        })

    return json.dumps(converted_data)

# Define function to upload JSON data to Cloud Storage
def upload_to_storage(json_data, type, config):
    account_from = list(map(str, json_data['account_details'].keys()))[0]
    account_name = json_data['account_details'][account_from]['account_name']
    account_id = json_data['account_details'][account_from]['account_id']
    account_platform_name = json_data['account_details'][account_from]['account_platform_name']
    account_platform_id = json_data['account_details'][account_from]['account_platform_id']
    file_name_str = str(account_from+"--"+account_name+"--account_id_"+str(account_id)+"--"+account_platform_name+"--account_platform_id_"+str(account_platform_id))
    if type == 'input':
        bucket_name = config['upload_to_storage_input']['bucket_name']
        folder_name = config['upload_to_storage_input']['folder_name']
        output_file_name = config['upload_to_storage_input']['output_file_name']
        output_file_name = 'input_'+file_name_str+'__'+dt.datetime.now().strftime('%Y-%m-%d - %H:%M:%S:%f %p %Z')+'.json'
    elif type == 'output':
        bucket_name = config['upload_to_storage_output']['bucket_name']
        folder_name = config['upload_to_storage_output']['folder_name']
        output_file_name = config['upload_to_storage_output']['output_file_name']
        output_file_name = 'output_'+file_name_str+'__'+dt.datetime.now().strftime('%Y-%m-%d - %H:%M:%S:%f %p %Z')+'.json'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{output_file_name}")
    blob.upload_from_string(json.dumps(json_data))