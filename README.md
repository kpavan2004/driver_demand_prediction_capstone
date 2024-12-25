# driver_demand_prediction_capstone Local setup

git clone https://github.com/kpavan2004/driver_demand_prediction_capstone.git
cd driver_demand_prediction_capstone
git checkout dev
python -m venv venv
source ./venv/Scripts/activate
pip install -r requirements/test_requirements.txt

## Launch the mlflow
Mlflow server should be launched first using : and then change port visibility from private to public incase of codespace. Down the docker image created from MLflow/Dockerfile
```bash
docker run -it -d -p 5000:5000 -u root -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\dev\driver_demand_prediction_capstone\MLflow\mlflow.db:/mlflow/mlflow.db" -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\dev\driver_demand_prediction_capstone\MLflow\mlartifacts/:/mlflow/mlartifacts/" --name=mlflow_cont kpavan2004/mlflow_server_dev
```
Update your local folder settings accordingly for the mount -v argument

verify if Mlflow ui is up and running http://localhost:5000

## Upload the best model to mlflow

cd notebooks/
python Log_Best_Exp_Model.py


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
