import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from delivery_time_model.config.core import config
from delivery_time_model.processing.data_manager import pre_pipeline_preparation,pre_pipeline_trans


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    pre_processed = pre_pipeline_trans(data_frame = input_df)  
    validated_data = pre_processed[config.ml_config.features]
    errors = None
    duplicates = validated_data.columns[validated_data.columns.duplicated()].tolist()

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class DataInputSchema(BaseModel):
    Delivery_person_Age: Optional[str]
    Delivery_person_Ratings : Optional[str]
    Restaurant_latitude: Optional[str]
    Restaurant_longitude: Optional[str]
    Delivery_location_latitude: Optional[str]
    Delivery_location_longitude: Optional[str]
    Weatherconditions: Optional[str]
    Road_traffic_density: Optional[str]
    Vehicle_condition: Optional[str]
    Type_of_order: Optional[str]
    Type_of_vehicle: Optional[str]
    multiple_deliveries : Optional[str]
    Festival: Optional[str]
    City_area: Optional[str]
    City: Optional[str]
    day_of_week : Optional[int]
    is_weekend : Optional[int]
    quarter : Optional[int]
    yr: Optional[int]
    mnth: Optional[str]
    Distance: Optional[float]
    order_prepare_time: Optional[float]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
    
class DataInputSchema_api(BaseModel):
    ID: Optional[str]
    Delivery_person_ID: Optional[str]
    Delivery_person_Age: Optional[str]
    Delivery_person_Ratings : Optional[str]
    Restaurant_latitude: Optional[str]
    Restaurant_longitude: Optional[str]
    Delivery_location_latitude: Optional[str]
    Delivery_location_longitude: Optional[str]
    Order_Date: Optional[str]
    Time_Orderd: Optional[str]
    Time_Order_picked: Optional[str]
    Weatherconditions: Optional[str]
    Road_traffic_density: Optional[str]
    Vehicle_condition: Optional[str]
    Type_of_order: Optional[str]
    Type_of_vehicle: Optional[str]
    multiple_deliveries : Optional[str]
    Festival: Optional[str]
    # City_area: Optional[str]
    City: Optional[str]
    # day_of_week : Optional[int]
    # is_weekend : Optional[int]
    # quarter : Optional[int]
    # yr: Optional[int]
    # mnth: Optional[str]
    # Distance: Optional[float]
    # order_prepare_time: Optional[float]
    
class MultipleDataInputs_api(BaseModel):
    inputs: List[DataInputSchema_api]