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

# Retrieve the best model from MLflow
def get_best_model_from_mlflow():
    # Set MLflow tracking URI
    import mlflow
    mlflow.set_tracking_uri(config.app_config.mlflow_tracking_uri)
    
    # Get the experiment
    exp = mlflow.set_experiment(experiment_name = "Driver-Delivery-Time-Prediction")
    if exp is None:
        raise ValueError(f"Experiment '{exp}' does not exist in MLflow.")
    
    # Create MLflow client
    client = mlflow.tracking.MlflowClient()

    # Load model via 'models'
    model_name = config.app_config.registered_model_name              #"sklearn-titanic-rf-model"
    model_info = client.get_model_version_by_alias(name=model_name, alias="production")
    print(f'Model version fetched: {model_info.version}')
    best_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@production")
    
    # Retrieve and print the hyperparameters of the best model
    # Retrieve the run and its parameters
    run_id = model_info.run_id
    run = client.get_run(run_id)
    hyperparams = run.data.params

    # Print the hyperparameters
    print("Best Model Hyperparameters:")   
    best_params = {key: convert_to_numeric(value) for key, value in hyperparams.items()}
    print(best_params)

    return best_model, best_params

best_model, best_params = get_best_model_from_mlflow()

demand_pipe = Pipeline([
    
    ######### Mapper ###########
    ('map_Weatherconditions', Mapper(variable = config.ml_config.Weatherconditions_var, mappings = config.ml_config.weather_mappings)),
    
    ('map_Road_traffic_density', Mapper(variable = config.ml_config.Road_traffic_density_var, mappings = config.ml_config.traff_den_mappings)),
    
    ('map_Type_of_order', Mapper(variable = config.ml_config.Type_of_order_var, mappings = config.ml_config.order_type_mappings)),
    
    ('map_Type_of_vehicle', Mapper(variable = config.ml_config.Type_of_vehicle_var, mappings = config.ml_config.vehicle_mappings)),
    
    ('map_City_area', Mapper(variable = config.ml_config.City_area_var, mappings = config.ml_config.city_area_mappings)),
      
    ('map_Festival_var', Mapper(variable = config.ml_config.Festival_var, mappings = config.ml_config.festival_mappings)),
        
    ('map_mnth', Mapper(variable = config.ml_config.mnth_var, mappings = config.ml_config.mnth_mappings)),
       
    ######## One-hot encoding ########
    ('encode_weekday', WeekdayOneHotEncoder(variable = config.ml_config.day_of_week_var)),
    ('encode_city', WeekdayOneHotEncoder(variable = config.ml_config.City_var)),

    # Scale features
    ('scaler', StandardScaler()),
    
    ('model_xgb', XGBRegressor(**best_params))
    ])
