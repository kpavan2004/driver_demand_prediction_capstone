import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,mean_absolute_percentage_error
import mlflow.sklearn
import os
from delivery_time_model.config.core import config
from delivery_time_model.pipeline import demand_pipe
from delivery_time_model.processing.data_manager import load_dataset, save_pipeline
from xgboost import XGBRegressor

def convert_to_numeric(value):
    """
    Convert a string to a numeric type (int or float).
    If the value cannot be converted, raises a ValueError.
    """
    try:
        # Try converting to integer
        return int(value)
    except ValueError:
        try:
            # If it fails, try converting to float
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot convert {value} to a numeric type.")


def run_training() -> None:
    
    """
    Train the model.
    """
    # import os
    # os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000/"
    import mlflow
    
    # Set the tracking URI to the server
    mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
    
    # Set an experiment name, unique and case-sensitive
    # It will create a new experiment if the experiment with given doesn't exist
    exp = mlflow.set_experiment(experiment_name = "Driver-Delivery-Time-Prediction-New")
    
    # read training data
    data = load_dataset(file_name = config.app_config.training_data_file)

    row_count = len(data)
    print(f"Number of rows in the training/retraining dataset: {row_count}")
    
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
    # input_example = X_train.iloc[:1]
    
    # Start an MLflow run
    mlflow.start_run(experiment_id= exp.experiment_id)
    # Pipeline fitting
    demand_pipe.fit(X_train, y_train)
    y_pred = demand_pipe.predict(X_test)
    # model=XGBRegressor(n_estimators = config.ml_config.n_estimators, 
    #                                    max_depth = config.ml_config.max_depth,
    #                                   random_state = config.ml_config.random_state)
    test_r2_score = r2_score(y_test, y_pred)
    # Log parameters and metrics
    mlflow.log_metric("r2_score", r2_score(y_test, y_pred))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test,y_pred)))
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("mape", np.sqrt(mean_absolute_percentage_error(y_test,y_pred)))

    # Calculate the score/error
    print("R2 score:", r2_score(y_test, y_pred))
    print("Mean squared error:", mean_squared_error(y_test, y_pred))
    print(f"Root mean squared error:{np.sqrt(mean_squared_error(y_test,y_pred))}")
    print("Mean Absolute Pecentage error:",mean_absolute_percentage_error(y_test,y_pred))
   
    # Load current 'production' model via 'models'
    import mlflow.pyfunc
    model_name = config.app_config.registered_model_name         #"sklearn-titanic-rf-model"
    client = mlflow.tracking.MlflowClient()
  
    try:
        # Capture the test-r2-score of the existing prod-model
        prod_model_info = client.get_model_version_by_alias(name=model_name, alias="production")         # fetch prod-model info
        prod_model_run_id = prod_model_info.run_id                   # run_id of the run associated with prod-model
        prod_run = client.get_run(run_id=prod_model_run_id)          # get run info using run_id
        prod_r2_score = prod_run.data.metrics['r2_score']    # get metrics values

        # Capture the version of the last-trained model
        latest_model_info = client.get_model_version_by_alias(name=model_name, alias="last-trained")           # fetch latest-model info
        latest_model_version = int(latest_model_info.version)              # latest-model version
        new_version = latest_model_version + 1                      # new model version

    except Exception as e:
        print(e)
        new_version = 1


    # Criterion to Log trained model
    if new_version > 1:
        if prod_r2_score < test_r2_score:
            print("Trained model is better than the existing model in production, will use this model in production!")
            better_model = True
        else:
            print("Trained model is not better than the existing model in production!")
            better_model = False
        first_model = False
    else:
        print("No existing model in production, registering a new model!")
        first_model = True
   
    # Register new model/version of model
    mlflow.sklearn.log_model(sk_model = demand_pipe, 
                            artifact_path="trained_model",
                            registered_model_name=model_name
                            )
    # Add 'last-trained' alias to this new model version
    client.set_registered_model_alias(name=model_name, alias="last-trained", version=str(new_version))


    if first_model or better_model:
        # Promote the model to production by adding 'production' alias to this new model version
        client.set_registered_model_alias(name=model_name, alias="production", version=str(new_version))
    else:
        # Don't promote this new model version
        pass

    # Log params
    # fetch latest-model info
    latest_model_info = client.get_model_version_by_alias(name=model_name, alias="experiment-best-model")  
    run_id = latest_model_info.run_id
    run = client.get_run(run_id)
    hyperparams = run.data.params

    # Print the hyperparameters   
    best_params = {key: convert_to_numeric(value) for key, value in hyperparams.items()}
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)
        
    # import ast
    # parsed_params = ast.literal_eval(best_params['best_param'])
    # mlflow.log_param('n_estimators', parsed_params['n_estimators'])
    # mlflow.log_param('max_depth', parsed_params['max_depth'])
    # mlflow.log_param('learning_rate', float(parsed_params['learning_rate']))
    # mlflow.log_param('subsample', float(parsed_params['subsample']))
    # mlflow.log_param('colsample_bytree', float(parsed_params['colsample_bytree']))
        
        

    # End an active MLflow run
    mlflow.end_run()
       
    # persist trained model
    save_pipeline(pipeline_to_persist = demand_pipe)
    
if __name__ == "__main__":
       
    print("Re-Training:", os.environ['RE_TRAIN'])
    if os.environ['RE_TRAIN']=='Yes':
        run_training()