import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split

from delivery_time_model.config.core import config
from delivery_time_model.processing.data_manager import load_dataset_test1


@pytest.fixture
def sample_input_data():
    # print("inside sample input data")
    data = load_dataset_test1(file_name = config.app_config.training_data_file)
    # print("Output of load_dataset_test columns:")
    # print(data.columns)
    
    # data = data.head(1)
    
    
    # data = data[-((data["Restaurant_latitude"]==0.0) & (data["Restaurant_longitude"]==0.0)) ]
    # data = data.head(10)
    
    all_except_target = ['ID', 'Delivery_person_ID', 'Delivery_person_Age',
       'Delivery_person_Ratings', 'Restaurant_latitude',
       'Restaurant_longitude', 'Delivery_location_latitude',
       'Delivery_location_longitude', 'Order_Date', 'Time_Orderd',
       'Time_Order_picked', 'Weatherconditions', 'Road_traffic_density',
       'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle',
       'multiple_deliveries', 'Festival','City']
    
    # X_test = data[all_except_target]
    # X_test = data.loc[:, data.columns != config.ml_config.target]
    # y_test = data[config.ml_config.target]
    
    # features = config.ml_config.features.copy()
    # print(features)
    # features.append("Delivery_person_ID")
    
    # Exclude the 'Label' column
    # df_without_label = data.drop(columns=[config.ml_config.target],axis=1)
    features = config.ml_config.features.copy()
    # print(features)
    add_cols = ["Delivery_person_ID","Order_Date","Time_Orderd","Time_Order_picked"]
    for value in add_cols:
        features.append(value)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[features],     # predictors
        data[config.ml_config.target],       # target
        test_size = config.ml_config.test_size,
        random_state=config.ml_config.random_state,   # set the random seed here for reproducibility
    )

    return X_test, y_test