"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from delivery_time_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_num_of_predictions = 9119
    # print(sample_input_data[0].columns)
    print("##############################")
    print(sample_input_data[0].iloc[0].to_dict())
    print("##############################")
    
    print(sample_input_data[0])
    # Convert all columns in sample_input_data[0] to strings
    sample_input_data_new = sample_input_data[0].astype(str)

    # data_in = {'Delivery_person_Age': '30', 'Delivery_person_Ratings': '5', 'Restaurant_latitude': 22.744648, 'Restaurant_longitude': 75.894377, 'Delivery_location_latitude': 22.824648, 'Delivery_location_longitude': 75.974377, 'Weatherconditions': 'conditions Fog', 'Road_traffic_density': 'Medium', 'Vehicle_condition': 0, 'Type_of_order': 'Snack', 'Type_of_vehicle': 'motorcycle', 'multiple_deliveries': '0', 'Festival': 'Yes', 'City_area': 'Metropolitian', 'City': 'INDO', 'Delivery_person_ID': 'INDORES16DEL02'} 
    # data_in = {'ID': ['0x4607'], 'Delivery_person_ID': ['INDORES13DEL02'], 'Delivery_person_Age': ['37'], 'Delivery_person_Ratings': ['4.9'], 'Restaurant_latitude': ['22.745049'],                'Restaurant_longitude': ['75.892471'], 'Delivery_location_latitude': ['22.765049'], 'Delivery_location_longitude': ['75.912471'], 'Order_Date': ['19-03-2022'], 'Time_Orderd': ['11:30:00'], 'Time_Order_picked': ['11:45:00'],'Weatherconditions' :['conditions Sunny'],'Road_traffic_density' :['High'],'Vehicle_condition' :['2'],'Type_of_order' :['Snack'],'Type_of_vehicle':['motorcycle'],'multiple_deliveries':['0'],'Festival' :['No'],'City' :['Urban']}
    result = make_prediction(input_data = sample_input_data_new)


    # # Then
    predictions = result.get("predictions")
    print(predictions)
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.float32)
    assert result.get("errors") is None
    assert len(predictions) == expected_num_of_predictions
    
    _predictions = list(predictions)
    y_true = sample_input_data[1]

    r2 = r2_score(y_true, _predictions)
    mse = mean_squared_error(y_true, _predictions)
    rmse = root_mean_squared_error(y_true, _predictions)
    print("r2 score:", r2)
    print("mse :", mse)
    print("rmse :", rmse)
    # assert r2 > 0.8
    # assert mse < 3000.0