import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,root_mean_squared_error,mean_absolute_percentage_error
import mlflow
import mlflow.sklearn

from delivery_time_model.config.core import config
from delivery_time_model.pipeline import demand_pipe
from delivery_time_model.processing.data_manager import load_dataset, save_pipeline
from xgboost import XGBRegressor

def run_training() -> None:
    
    """
    Train the model.
    """
    import os
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000/"
    
    # read training data
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    def drop_zero_lat_long(data = data):
        dataframe = data[-((data["Restaurant_latitude"]==0.0) & (data["Restaurant_longitude"]==0.0)) ]
        return dataframe
    # R2 score reduced from .76 to .747
    data = drop_zero_lat_long(data)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.ml_config.features],     # predictors
        data[config.ml_config.target],       # target
        test_size = config.ml_config.test_size,
        random_state=config.ml_config.random_state,   # set the random seed here for reproducibility
    )
    
    # Define an input example (sample data)
    input_example = X_train.iloc[:1]
    
    # Start an MLflow run
    with mlflow.start_run():
        # Pipeline fitting
        demand_pipe.fit(X_train, y_train)
        y_pred = demand_pipe.predict(X_test)
        model=XGBRegressor(n_estimators = config.ml_config.n_estimators, 
                                       max_depth = config.ml_config.max_depth,
                                      random_state = config.ml_config.random_state)
        # Log parameters and metrics

        mlflow.log_param("n_estimators", config.ml_config.n_estimators)
        mlflow.log_param("max_depth", config.ml_config.max_depth)
        mlflow.log_metric("r2_score", r2_score(y_test, y_pred))
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test,y_pred)))
        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("mape", np.sqrt(mean_absolute_percentage_error(y_test,y_pred)))
        # Log the model
        mlflow.sklearn.log_model(sk_model=demand_pipe,artifact_path="model",input_example=input_example,registered_model_name="delivery_time_model")

        print(f"Logged model with max_depth: ",r2_score(y_test, y_pred))

    # Calculate the score/error
    print("R2 score:", r2_score(y_test, y_pred))
    print("Mean squared error:", mean_squared_error(y_test, y_pred))
    print(f"Root mean squared error:{np.sqrt(root_mean_squared_error(y_test,y_pred))}")
    print("Mean Absolute Pecentage error:",mean_absolute_percentage_error(y_test,y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist = demand_pipe)
    
if __name__ == "__main__":
    run_training()