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

    return results

rmse_metric = prom.Gauge('delivery_time_rmse', 'Root mean square error for few random test samples')
r2_metric = prom.Gauge('delivery_time_r2_score', 'R2 score for random test samples')


# Function for updating metrics
def update_metrics():
    
    # LOAD TEST DATA
    test_data = load_dataset_test1(file_name = "train.csv")
    test = test_data.sample(10).astype(str)
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
  
@api_router.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())
