
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

def test_traffic_density_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.Road_traffic_density_var, 
                    mappings = config.ml_config.traff_den_mappings)
    assert sample_input_data[0].loc[1, 'Road_traffic_density'] == 'Jam'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'Road_traffic_density'] == 1
    
def test_weather_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.Weatherconditions_var, 
                    mappings = config.ml_config.weather_mappings)
    assert sample_input_data[0].loc[1, 'Weatherconditions'] == 'Stormy'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'Weatherconditions'] == 1

def test_order_type_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.Type_of_order_var, 
                    mappings = config.ml_config.order_type_mappings)
    assert sample_input_data[0].loc[1, 'Type_of_order'] == 'Snack'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'Type_of_order'] == 0
    
def test_vehicle_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.Type_of_vehicle_var, 
                    mappings = config.ml_config.vehicle_mappings)
    assert sample_input_data[0].loc[1, 'Type_of_vehicle'] == 'scooter'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'Type_of_vehicle'] == 1
    
def test_festival_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.Festival_var, 
                    mappings = config.ml_config.festival_mappings)
    assert sample_input_data[0].loc[1, 'Festival'] == 'No'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'Festival'] == 0
    
def test_festival_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.Festival_var, 
                    mappings = config.ml_config.festival_mappings)
    assert sample_input_data[0].loc[1, 'Festival'] == 'No'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'Festival'] == 0
    
def test_city_area_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.City_area_var, 
                    mappings = config.ml_config.city_area_mappings)
    assert sample_input_data[0].loc[1, 'City_area'] == 'Metropolitian'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'City_area'] == 1

def test_month_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.ml_config.mnth_var, 
                    mappings = config.ml_config.mnth_mappings)
    assert sample_input_data[0].loc[1, 'mnth'] == 'March'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1, 'mnth'] == 3