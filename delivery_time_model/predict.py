import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from delivery_time_model import __version__ as _version
from delivery_time_model.config.core import config
from delivery_time_model.processing.data_manager import load_pipeline
from delivery_time_model.processing.data_manager import pre_pipeline_preparation
from delivery_time_model.processing.validation import validate_inputs

#################### MLflow CODE START to load 'production' model #############################
import mlflow 
import mlflow.pyfunc
# mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
mlflow.set_tracking_uri("http://192.168.1.19:5000")
# Create MLflow client
client = mlflow.tracking.MlflowClient()

# Load model via 'models'
model_name = config.app_config.registered_model_name              #"sklearn-titanic-rf-model"
model_info = client.get_model_version_by_alias(name=model_name, alias="production")
print(f'Model version fetched: {model_info.version}')

demand_pipe = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@production")
#################### MLflow CODE END ##########################################################

# pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
# demand_pipe = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    # print("In make prediction function")
    
    # Drop records where lat and long are zero
    # input_df = pd.DataFrame(input_data)
    # input_df = input_df[-((input_df["Restaurant_latitude"]==0.0) & (input_df["Restaurant_longitude"]==0.0)) ]
    # print(input_df.shape)
    print("Before validate_inputs ")
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    print("After validate_inputs ")
    # print("after calling validate_inputs function")
    # print(validated_data.columns)
    # print(validated_data.head(1))
    # print(errors)
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.ml_config.features)
    # print("After reindex")
    # print(validated_data.iloc[0].to_dict())
    
    results = {"predictions": None, "version": _version, "errors": errors}
    print("The predictions results are :")  
    print(results)
    if not errors:
        # print("inside if statement")
        predictions = demand_pipe.predict(validated_data)
        # print("after prediction statement")
        # print(type(predictions))
        # print(predictions)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    # data_in = {'ID': ['0x4607'], 'Delivery_person_ID': ['INDORES13DEL02'], 'Delivery_person_Age': ['37'], 'Delivery_person_Ratings': ['4.9'], 'Restaurant_latitude': ['22.745049'],
    #            'Restaurant_longitude': ['75.892471'], 'Delivery_location_latitude': ['22.765049'], 'Delivery_location_longitude': ['75.912471'], 'Order_Date': ['19-03-2022'], 'Time_Orderd': ['11:30:00'], 'Time_Order_picked': ['11:45:00'],'Weatherconditions' :['conditions Sunny'],'Road_traffic_density' :['High'],'Vehicle_condition' :['2'],'Type_of_order' :['Snack'],'Type_of_vehicle':['motorcycle'],'multiple_deliveries':['0'],'Festival' :['No'],'City' :['Urban'],'Time_taken(min)' :['(min) 24']}
    # data_in = {'ID': ['0x4607'], 'Delivery_person_ID': ['INDORES13DEL02'], 'Delivery_person_Age': ['37'], 'Delivery_person_Ratings': ['4.9'], 'Restaurant_latitude': ['22.745049'],
    #    'Restaurant_longitude': ['75.892471'], 'Delivery_location_latitude': ['22.765049'], 'Delivery_location_longitude': ['75.912471'], 'Order_Date': ['19-03-2022'], 'Time_Orderd': ['11:30:00'], 'Time_Order_picked': ['11:45:00'],'Weatherconditions' :['conditions Sunny'],'Road_traffic_density' :['High'],'Vehicle_condition' :['2'],'Type_of_order' :['Snack'],'Type_of_vehicle':['motorcycle'],'multiple_deliveries':['0'],'Festival' :['No'],'City' :['Urban']}
    data_in = {'Delivery_person_ID': ['INDORES13DEL02'], 'Delivery_person_Age': ['37'], 'Delivery_person_Ratings': ['4.9'], 'Restaurant_latitude': ['22.745049'],
               'Restaurant_longitude': ['75.892471'], 'Delivery_location_latitude': ['22.765049'], 'Delivery_location_longitude': ['75.912471'], 'Order_Date': ['19-03-2022'], 'Time_Orderd': ['11:30:00'], 'Time_Order_picked': ['11:45:00'],'Weatherconditions' :['conditions Sunny'],'Road_traffic_density' :['High'],'Vehicle_condition' :['2'],'Type_of_order' :['Snack'],'Type_of_vehicle':['motorcycle'],'multiple_deliveries':['0'],'Festival' :['No'],'City' :['Urban']}
    make_prediction(input_data = data_in)																			
