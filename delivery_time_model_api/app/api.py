import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from delivery_time_model import __version__ as ml_version
from delivery_time_model.predict import make_prediction

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from prometheus_client import Counter, Histogram, generate_latest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app import __version__, schemas
from app.config import settings
from prometheus_client import Counter, Histogram, generate_latest
import prometheus_client as prom
curr_path = str(Path(__file__).parent)
from sklearn.metrics import r2_score,root_mean_squared_error
from delivery_time_model.processing.data_manager import load_dataset_test1
import psutil
import boto3
import os
from io import StringIO

def append_to_csv_in_s3(bucket_name, object_key,new_data, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
    """
    Appends a new record to an existing CSV file in S3.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        object_key (str): The key (path/filename) of the CSV file in the S3 bucket.
        new_data (dict): The new record to append as a dictionary.
        aws_access_key_id (str, optional): AWS access key ID.
        aws_secret_access_key (str, optional): AWS secret access key.
        region_name (str, optional): AWS region.

    Returns:
        str: The S3 object URL after the update.
    """
    # Initialize the S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    try:
        # Download the existing file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        existing_data = response['Body'].read().decode('utf-8')
        
        # Load the existing CSV into a DataFrame
        existing_df = pd.read_csv(StringIO(existing_data))
    except s3_client.exceptions.NoSuchKey:
        # If the file doesn't exist, create an empty DataFrame
        existing_df = pd.DataFrame()
        
    # Convert the new data to a DataFrame
    new_df = new_data

    # Combine old and new DataFrames, and drop duplicates
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    deduped_df = combined_df.drop_duplicates()

    # Convert the updated DataFrame to CSV
    csv_buffer = StringIO()
    deduped_df.to_csv(csv_buffer, index=False)

    # Upload the updated CSV back to S3
    s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())

    # Return the S3 object URL
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
    return s3_url

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        # name=settings.PROJECT_NAME, api_version=__version__, model_version=ml_version
        name=settings.PROJECT_NAME, api_version=__version__, ml_version=ml_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs_api) -> Any:
    """
    Driver Demand prediction with the demand_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))
    
    predictions_value = ' (min)' + str(results['predictions'][0])
      
    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    bucket_name = "pk-capstone-bucket-01"
    object_key = "inference_data/new_data.csv"
    
    input_df['Time_taken(min)'] = [predictions_value]
    # Provide AWS credentials and region if required
    s3_url = append_to_csv_in_s3(
        bucket_name=bucket_name,
        object_key=object_key,
        new_data=input_df,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="ap-south-1"
    )
    print(f"CSV uploaded successfully to {s3_url}")
    
    return results

rmse_metric = prom.Gauge('delivery_time_rmse', 'Root mean square error for few random test samples')
r2_metric = prom.Gauge('delivery_time_r2_score', 'R2 score for random test samples')
cpu_usage_gauge = prom.Gauge("app_cpu_usage_percent", "CPU usage of the app")
memory_usage_gauge = prom.Gauge("app_memory_usage_bytes", "Memory usage of the app")

# Function for updating metrics
def update_metrics():
    
    # LOAD TEST DATA
    test_data = load_dataset_test1(file_name = "train.csv")
    # test = test_data.sample(10).astype(str)
    test = test_data.astype(str)
    # test_feat = test.drop('Time_taken', axis=1)
    test_actual = test['Time_taken'].astype(float).values
    
    result = make_prediction(input_data=test)
    
    predictions = result.get("predictions")
    print(predictions)
  
    _predictions = list(predictions)
    
    r2 = r2_score(test_actual, _predictions)   
    r2_metric.set(r2)
    
    rmse = root_mean_squared_error(test_actual, _predictions)
    rmse_metric.set(rmse)
    
    # Capture CPU and memory usage
    cpu_usage_gauge.set(psutil.cpu_percent())
    memory_info = psutil.virtual_memory()
    memory_usage_gauge.set(memory_info.used)
  
@api_router.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())
