# driver_demand_prediction_capstone
The    food delivery industry faces significant challenges in predicting driver demand and accurately estimating delivery times. The efficiency of the delivery process directly affects customer satisfaction, operational costs, and overall service reliability. Inaccurate predictions of delivery times and driver allocation often lead to under- or over-supply of drivers, delays in deliveries, and unsatisfactory customer experiences. This project aims to develop a machine learning-based predictive model to optimize delivery charges and predict driver demand based on various parameters, improving both the customer experience and operational efficiency for delivery services.

## DVC setup of train.csv
 Refer to github repo: https://github.com/kpavan2004/dvc-driver-demand-capstone

## Launch the mlflow
Mlflow server should be launched first using : and then change port visibility from private to public. Down the docker image created from git repo: https://github.com/kpavan2004/mlflow_server
```bash
docker pull kpavan2004/mlflow_server

docker run -it -d -p 5000:5000 -u root --env-file "C:\Capstone_Project\MLflow\.env"  -v "C:\Capstone_Project\MLflow\mlflow.db:/mlflow/mlflow.db" --name=mlflow_cont kpavan2004/mlflow_server
```

verify if Mlflow ui is up and running http://localhost:5000

## Train the model

```bash
cd /delivery_time_model
python train_pipeline.py
```
## Inference

```bash
cd /delivery_time_model
python predict.py
```

## Test
```bash
pytest
```

## Build
```bash
python -m build 
```
It creates .whl and .pkl file in /dist folder

## copy the .whl file to api folder
```bash
cp /delivery_time_model/dsit/*.whl /delivery_time_model_api/
```

## Deploy
```bash
cd /delivery_time_model_api/api
python main.py
```

## Dockerize the application:
```bash
docker build . -t kpavan2004/driver_delivery_time_pred-fastapi_noenv:latest
```

## Launch the docker container
```bash
docker run -it -d -p 8001:8001 --name=app_cont kpavan2004/driver_delivery_time_pred-fastapi_noenv:latest
```
Note to change port visibility from private to public incase of github codespace

## Prometheus
```bash
docker run -it -d -p 9090:9090 -u root -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml" -v "$PWD/prometheus-data:/prometheus" --name=prom_cont prom/prometheus
```
Note to change port visibility from private to public incase of github codespace

## Grafana
```bash
docker run -it -d -p 3000:3000 -u root -v "$PWD/grafana-data:/var/lib/grafana" --env-file "$PWD/env.list" --name=grafana_cont grafana/grafana-oss
```
Note to change port visibility from private to public incase of github codespace