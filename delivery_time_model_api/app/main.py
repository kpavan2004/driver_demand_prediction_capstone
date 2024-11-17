import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
#print(sys.path)
from typing import Any
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import APIRouter, FastAPI, Request,Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api import api_router
from app.config import settings

from prometheus_client import Counter, Histogram, generate_latest
import prometheus_client as prom
curr_path = str(Path(__file__).parent)
from sklearn.metrics import r2_score,accuracy_score, f1_score, precision_score, recall_score
import prometheus_client as prom
from delivery_time_model import __version__ as ml_version
from delivery_time_model.predict import make_prediction


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()
curr_path =str(Path(__file__).parent)

# FastAPI object

@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1 style='background-color:LightGray;'><center>Driver Demand Application</center></h1>"
        "<h2>Welcome to the API</h2>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

# def predict_delivery_time(ID,Delivery_person_ID,Delivery_person_Age,Delivery_person_Ratings,Restaurant_latitude,Restaurant_longitude,Delivery_location_latitude,Delivery_location_longitude,Order_Date,Time_Orderd,Time_Order_picked,Weatherconditions,Road_traffic_density,Vehicle_condition,Type_of_order,Type_of_vehicle,multiple_deliveries,Festival,City):
#     ID=str(ID)
#     Delivery_person_ID=str(Delivery_person_ID)
#     Delivery_person_Age=str(Delivery_person_Age)
#     Delivery_person_Ratings=str(Delivery_person_Ratings)
#     Restaurant_latitude=str(Restaurant_latitude)
#     Restaurant_longitude=str(Restaurant_longitude)
#     Delivery_location_latitude=str(Delivery_location_latitude)
#     Delivery_location_longitude=str(Delivery_location_longitude)
#     Order_Date=str(Order_Date)
#     Time_Orderd=str(Time_Orderd)
#     Time_Order_picked=str(Time_Order_picked)
#     Weatherconditions=str(Weatherconditions)
#     Road_traffic_density=str(Road_traffic_density)
#     Vehicle_condition=str(Vehicle_condition)
#     Type_of_order=str(Type_of_order)
#     Type_of_vehicle=str(Type_of_vehicle)
#     multiple_deliveries=str(multiple_deliveries)
#     Festival=str(Festival)
#     City=str(City)
    
#     input = [ID,Delivery_person_ID,Delivery_person_Age,Delivery_person_Ratings,Restaurant_latitude,Restaurant_longitude,Delivery_location_latitude,Delivery_location_longitude,Order_Date,Time_Orderd,Time_Order_picked,Weatherconditions,Road_traffic_density,Vehicle_condition,Type_of_order,Type_of_vehicle,multiple_deliveries,Festival,City]
#     input_df = pd.DataFrame([input],columns = ['ID','Delivery_person_ID','Delivery_person_Age','Delivery_person_Ratings','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked','Weatherconditions','Road_traffic_density','Vehicle_condition','Type_of_order','Type_of_vehicle','multiple_deliveries','Festival','City'])
#     pred_results = make_prediction(input_data=input_df.replace({np.nan: None}))
#     return pred_results[0]

#     ID = gr.Slider(0, 100, label="ID", value=70, info="Choose age between 0 and 100")
#     Delivery_person_ID = gr.Radio(["0", "1"], label="Delivery_person_ID",value="0", info="Does patient have anaemia ->  0-False, 1-True")
#     Delivery_person_Age = gr.Slider(0, 1000, label="Delivery_person_Age",value=161, info="Choose creatinine_phosphokinase between 0 and 1000")
#     Delivery_person_Ratings = gr.Radio(["0", "1"], label="Delivery_person_Ratings", value="0",info="Does patient have diabetes ->  0-False, 1-True")
#     Restaurant_latitude = gr.Slider(0, 100, label="Restaurant_latitude",value=25, info="Choose ejection_fraction between 0 and 100")
#     Restaurant_longitude = gr.Radio(["0", "1"], label="Restaurant_longitude",value="0", info="Does patient have High BP ->  0-False, 1-True")
#     Delivery_location_latitude = gr.Slider(25000, 850000, label="Delivery_location_latitude",value=244000, info="Choose platelets between 25000 and 850000")
#     Delivery_location_longitude = gr.Slider(0, 10, label="Delivery_location_longitude",value=1.2, info="Choose serum_creatinine between 0 and 10")
#     Order_Date = gr.Slider(100, 150, label="Order_Date",value=142, info="Choose serum_sodium between 100 and 150")
#     Time_Orderd =  gr.Radio(["0", "1"], label="Time_Orderd",value="0", info="Sex ->  0-Female, 1-Male")
#     Time_Order_picked = gr.Radio(["0", "1"], label="Time_Order_picked",value="0", info="Does patient smoke ->  0-False, 1-True")
#     Weatherconditions = gr.Slider(0, 365, label="Weatherconditions",value=66, info="Choose time between 0 and 365")
#     Road_traffic_density = gr.Slider(0, 365, label="Road_traffic_density",value=66, info="Choose time between 0 and 365")
#     Vehicle_condition = gr.Slider(0, 365, label="Vehicle_condition",value=66, info="Choose time between 0 and 365")
#     Type_of_order = gr.Slider(0, 365, label="Type_of_order",value=66, info="Choose time between 0 and 365")
#     Type_of_vehicle = gr.Slider(0, 365, label="Type_of_vehicle",value=66, info="Choose time between 0 and 365")
#     multiple_deliveries = gr.Slider(0, 365, label="multiple_deliveries",value=66, info="Choose time between 0 and 365")
#     Festival = gr.Slider(0, 365, label="Festival",value=66, info="Choose time between 0 and 365")
#     City= gr.Slider(0, 365, label="City",value=66, info="Choose time between 0 and 365")

# # Output response
# outputs = gr.Textbox(type="text", label='The patient survival predictions is :')


# # Gradio interface to generate UI link
# title = "Patient Survival Prediction"
# description = "Predict survival of patient with heart failure, given their clinical record"


# iface = gr.Interface(fn = predict_death_event,
#                          inputs = [age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time],
#                          outputs = [outputs],
#                          title = title,
#                          description = description,
#                          allow_flagging='never')

# # iface.launch(share = True,debug=True,server_name="0.0.0.0", server_port = 8001)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface

# # Mount gradio interface object on FastAPI app at endpoint = '/'
# app = gr.mount_gradio_app(app, iface, path="/")



app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
