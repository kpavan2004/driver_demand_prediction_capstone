"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error

from delivery_time_model.predict import make_prediction

print("Before calling test_make_prediction")
def test_make_prediction(sample_input_data):
    # Given
    expected_num_of_predictions = 9119
    # print(sample_input_data[0].columns)
    print("##############################")
    print(sample_input_data[0].iloc[0].to_dict())
    print("##############################")
    
    # print(sample_input_data[0])
    # Convert all columns in sample_input_data[0] to strings
    sample_input_data_new = sample_input_data[0].astype(str)
    result = make_prediction(input_data = sample_input_data_new)

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
    rmse = np.sqrt(mean_squared_error(y_true,_predictions))
    print("r2 score:", r2)
    print("mse :", mse)
    print("rmse :", rmse)
    assert r2 > 0.7
    # assert mse < 3000.0