import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
#print(sys.path)
import gradio as gr
from typing import Any
import pandas as pd
import numpy as np
import yaml
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
from delivery_time_model.config.core import CONFIG_FILE_PATH



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

def predict_delivery_time(ID,Delivery_person_ID,Delivery_person_Age,Delivery_person_Ratings,Restaurant_latitude,Restaurant_longitude,Delivery_location_latitude,Delivery_location_longitude,Order_Date,Time_Orderd,Time_Order_picked,Weatherconditions,Road_traffic_density,Vehicle_condition,Type_of_order,Type_of_vehicle,multiple_deliveries,Festival,City):
    ID=str(ID)
    Delivery_person_ID=str(Delivery_person_ID)
    Delivery_person_Age=str(Delivery_person_Age)
    Delivery_person_Ratings=str(Delivery_person_Ratings)
    Restaurant_latitude=str(Restaurant_latitude)
    Restaurant_longitude=str(Restaurant_longitude)
    Delivery_location_latitude=str(Delivery_location_latitude)
    Delivery_location_longitude=str(Delivery_location_longitude)
    Order_Date=str(Order_Date)
    Time_Orderd=str(Time_Orderd)
    Time_Order_picked=str(Time_Order_picked)
    Weatherconditions=str(Weatherconditions)
    Road_traffic_density=str(Road_traffic_density)
    Vehicle_condition=str(Vehicle_condition)
    Type_of_order=str(Type_of_order)
    Type_of_vehicle=str(Type_of_vehicle)
    multiple_deliveries=str(multiple_deliveries)
    Festival=str(Festival)
    City=str(City)
    
    input = [ID,Delivery_person_ID,Delivery_person_Age,Delivery_person_Ratings,Restaurant_latitude,Restaurant_longitude,Delivery_location_latitude,Delivery_location_longitude,Order_Date,Time_Orderd,Time_Order_picked,Weatherconditions,Road_traffic_density,Vehicle_condition,Type_of_order,Type_of_vehicle,multiple_deliveries,Festival,City]
    input_df = pd.DataFrame([input],columns = ['ID','Delivery_person_ID','Delivery_person_Age','Delivery_person_Ratings','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked','Weatherconditions','Road_traffic_density','Vehicle_condition','Type_of_order','Type_of_vehicle','multiple_deliveries','Festival','City'])
    print(input_df.head())
    pred_results = make_prediction(input_data=input_df.replace({np.nan: None}))
    return pred_results['predictions'][0]

# Load city values from the YAML file
with open(CONFIG_FILE_PATH, "r") as file:
    config = yaml.safe_load(file)
    city_area_values = config["city_area_mappings"]
    Weatherconditions_values = config["weather_mappings"]
    Road_traffic_density_values = config["traff_den_mappings"]
    order_type_values = config["order_type_mappings"]
    vehicle_values = config["vehicle_mappings"]
    festival_values = config["festival_mappings"]
        
ID = gr.Textbox( label="ID", value='0xb379')
Delivery_person_ID = gr.Textbox( label="Delivery_person_ID",value="BANGRES18DEL02")
Delivery_person_Age = gr.Slider(0, 100, label="Delivery_person_Age",value=34)
Delivery_person_Ratings = gr.Textbox( label="Delivery_person_Ratings", value="4.5")
Restaurant_latitude = gr.Textbox(label="Restaurant_latitude",value="12.913041")
Restaurant_longitude = gr.Textbox( label="Restaurant_longitude",value="77.683237")
Delivery_location_latitude = gr.Textbox( label="Delivery_location_latitude",value="13.043041")
Delivery_location_longitude = gr.Textbox(label="Delivery_location_longitude",value="77.813237")
Order_Date = gr.Textbox( label="Order_Date",value="25-03-2022",info="Format: dd-mm-yyyy")
Time_Orderd =  gr.Textbox( label="Time_Orderd",value="19:45:00",info="Format: hh:mm:ss")
Time_Order_picked = gr.Textbox(label="Time_Order_picked",value="19:50:00",info="Format: hh:mm:ss")
Weatherconditions = gr.Dropdown(choices=Weatherconditions_values,label="Weatherconditions",value="Stormy")
Road_traffic_density = gr.Dropdown( choices=Road_traffic_density_values, label="Road_traffic_density",value="Jam")
Vehicle_condition = gr.Textbox( label="Vehicle_condition",value="2")
Type_of_order = gr.Dropdown(choices=order_type_values,label="Type_of_order",value="Snack")
Type_of_vehicle = gr.Dropdown(choices=vehicle_values,label="Type_of_vehicle",value="scooter")
multiple_deliveries = gr.Textbox(label="multiple_deliveries",value="1")
Festival = gr.Radio(["Yes", "No"], label="Festival",value="No")
City= gr.Dropdown(choices=city_area_values, label="City",value="Metropolitian")  # Dropdown for City
    

# Output response
outputs = gr.Textbox(type="text", label='The Delivery time prediction is :',lines=3)


# Gradio interface to generate UI link
title = "Driver Delivery time Prediction"
description = "Predict Driver Delivery time after provding the required features data"


iface = gr.Interface(fn = predict_delivery_time,
                         inputs = [ID,Delivery_person_ID,Delivery_person_Age,Delivery_person_Ratings,Restaurant_latitude,Restaurant_longitude,Delivery_location_latitude,Delivery_location_longitude,Order_Date,Time_Orderd,Time_Order_picked,Weatherconditions,Road_traffic_density,Vehicle_condition,Type_of_order,Type_of_vehicle,multiple_deliveries,Festival,City],
                         outputs = [outputs],
                         title = title,
                         description = description,
                         allow_flagging='never')

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

# # Define custom CSS for background image in the bottom-right corner
# custom_css = """
# body {
#     position: relative;
#     padding-bottom: 20px; /* Optional: Ensure content doesn't overlap the image */
# }

# background-image {
#     background-image: url('https://image.freepik.com/free-vector/hand-drawn-food-delivery-man_23-2147678391.jpg'); /* Local file reference */
#     position: fixed;
#     bottom: 10px;
#     right: 10px;
#     width: 150px; /* Adjust width as needed */
#     height: auto; /* Maintain aspect ratio */
#     z-index: 1000; /* Ensure it appears above other elements */
#     opacity: 0.8; /* Make it slightly transparent */
# }
# """

# # Wrap the interface in Blocks for custom CSS
# with gr.Blocks(css=custom_css) as app1:
#     with gr.Row():
#         iface.render()  # Render the existing interface
#     # Add the background image using HTML
#     gr.HTML("""
#         <img id="background-image" src="https://image.freepik.com/free-vector/hand-drawn-food-delivery-man_23-2147678391.jpg" alt="Background Image" />
#     """)
    
# Required only for gradio app launch
# iface.launch(share = True,debug=True,server_name="0.0.0.0", server_port = 8001)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
