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

demand_pipe = Pipeline([

    ######### Imputation ###########

    #('weathersit_imputation', WeathersitImputer(variable = config.ml_config.weathersit_var)),
    
    ######### Mapper ###########
    ('map_Weatherconditions', Mapper(variable = config.ml_config.Weatherconditions_var, mappings = config.ml_config.weather_mappings)),
    
    ('map_Road_traffic_density', Mapper(variable = config.ml_config.Road_traffic_density_var, mappings = config.ml_config.traff_den_mappings)),
    
    ('map_Type_of_order', Mapper(variable = config.ml_config.Type_of_order_var, mappings = config.ml_config.order_type_mappings)),
    
    ('map_Type_of_vehicle', Mapper(variable = config.ml_config.Type_of_vehicle_var, mappings = config.ml_config.vehicle_mappings)),
    
    ('map_City_area', Mapper(variable = config.ml_config.City_area_var, mappings = config.ml_config.city_area_mappings)),
    
    # ('map_City', Mapper(variable = config.ml_config.City_var, mappings = config.ml_config.city_mappings)),
    
    ('map_Festival_var', Mapper(variable = config.ml_config.Festival_var, mappings = config.ml_config.festival_mappings)),
    
    # ('map_yr', Mapper(variable = config.model_config.yr_var, mappings = config.model_config.yr_mappings)),
    
    ('map_mnth', Mapper(variable = config.ml_config.mnth_var, mappings = config.ml_config.mnth_mappings)),
       
    ######## Handle outliers ########
    #('handle_outliers_temp', OutlierHandler(variable = config.ml_config.temp_var)),

    ######## One-hot encoding ########
    ('encode_weekday', WeekdayOneHotEncoder(variable = config.ml_config.day_of_week_var)),
    ('encode_city', WeekdayOneHotEncoder(variable = config.ml_config.City_var)),

    # Scale features
    ('scaler', StandardScaler()),
    
    # Regressor
    # ('model_rf', RandomForestRegressor(n_estimators = config.ml_config.n_estimators, 
    #                                    max_depth = config.ml_config.max_depth,
    #                                   random_state = config.ml_config.random_state))
    ('model_xgb', XGBRegressor(n_estimators = config.ml_config.n_estimators, 
                                       max_depth = config.ml_config.max_depth,
                                      random_state = config.ml_config.random_state))
    
    ])
