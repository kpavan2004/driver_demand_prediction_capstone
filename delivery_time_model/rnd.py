import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from delivery_time_model.config.core import config
from delivery_time_model.processing.features import WeekdayImputer, WeathersitImputer
from delivery_time_model.processing.features import Mapper
from delivery_time_model.processing.features import OutlierHandler, WeekdayOneHotEncoder


# Retrieve the best model from MLflow
def get_best_model_from_mlflow():
    # Set MLflow tracking URI
    import mlflow
    mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
    
    # Get the experiment
    exp = mlflow.set_experiment(experiment_name = "Driver-Delivery-Time-Prediction-New")
    if exp is None:
        raise ValueError(f"Experiment '{exp}' does not exist in MLflow.")
    
    # Create MLflow client
    client = mlflow.tracking.MlflowClient()

    # Load model via 'models'
    model_name = config.app_config.registered_model_name              #"sklearn-titanic-rf-model"
    model_info = client.get_model_version_by_alias(name=model_name, alias="production")
    print(f'Model version fetched: {model_info.version}')
    best_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@production")
    print(model_info)
    
    # Retrieve and print the hyperparameters of the best model
    # Retrieve the run and its parameters
    run_id = model_info.run_id
    run = client.get_run(run_id)
    hyperparams = run.data.params
    print(hyperparams)
    print(type(hyperparams))
    # Print the hyperparameters
    print("Best Model Hyperparameters:")
    print(hyperparams.items())
    for param, value in hyperparams.items():
        print(f"{param}: {value}")
    
    best_params = {key: value for key, value in hyperparams.items()}
    print(best_params)
    
    print("""""""""""""""""""""""""""""""""""")
    

    # Log params
    latest_model_info = client.get_model_version_by_alias(name=model_name, alias="production")  
    print(latest_model_info)
    # fetch latest-model info
    run_id = latest_model_info.run_id
    print(run_id)
    run = client.get_run(run_id)
    hyperparams = run.data.params

    # Print the hyperparameters
    print("Best Model Hyperparameters:")
    print(hyperparams.items())
    for param, value in hyperparams.items():
        print(f"{param}: {value}")
    
    best_params = {key: value for key, value in hyperparams.items()}
    # print(best_params)
    # Convert the string to a Python dictionary
    # import ast
    # parsed_params = ast.literal_eval(best_params['best_param'])
    # print(parsed_params['n_estimators'])
    # return best_model

get_best_model_from_mlflow()




