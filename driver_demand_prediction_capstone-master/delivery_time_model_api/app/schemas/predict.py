from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from delivery_time_model.processing.validation import DataInputSchema_api, DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs_api(BaseModel):
    inputs: List[DataInputSchema_api]

    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    {
                "ID": "0x4607",
                "Delivery_person_ID": "INDORES13DEL02",
                "Delivery_person_Age": "37",
                "Delivery_person_Ratings": "4.9",
                "Restaurant_latitude": "22.745049",
                "Restaurant_longitude": "75.892471",
                "Delivery_location_latitude": "22.765049",
                "Delivery_location_longitude": "75.912471",
                "Order_Date": "19-03-2022",
                "Time_Orderd": "11:30:00",
                "Time_Order_picked": "11:45:00",
                "Weatherconditions": "conditions Sunny",
                "Road_traffic_density": "High",
                "Vehicle_condition": "2",
                "Type_of_order": "Snack",
                "Type_of_vehicle": "motorcycle",
                "multiple_deliveries": "0",
                "Festival": "No",
                "City": "Urban",
                
                # "City_area": "Urban",
                    }
                ]
            }
        }
