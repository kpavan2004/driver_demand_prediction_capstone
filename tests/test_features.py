
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from delivery_time_model.config.core import config
from delivery_time_model.processing.features import Mapper, OutlierHandler


# def test_weekday_variable_imputer(sample_input_data):
#     # Given
    
#     imputer = WeekdayImputer(variable = config.ml_config.weekday_var, date_var = config.ml_config.date_var)
#     assert np.isnan(sample_input_data[0].loc[7046, 'weekday'])

#     # When
#     subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[7046, 'weekday'] == 'Wed'


