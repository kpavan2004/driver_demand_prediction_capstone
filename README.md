training process:
cd /delivery_time_model
python train_pipeline.py

predict:
python predict.py

tests:
cd /tests/app/
python main.py

create the .whl file
python -m build 

copy the .whl file to _api
cp /delivery_time_model/dsit/*.whl /delivery_time_model_api/

Deploy model on fastapi
cd /delivery_time_model_api/api
python main.py

Dockerize the application:
docker build . -t kpavan2004/driver_delivery_time_pred-fastapi:latest

Run below command to run a container from your image
docker run -it -d -p 8001:8001 --name=app_cont kpavan2004/driver_delivery_time_pred-fastapi

From the Prometheus directory, run the below command to start Prometheus
docker run -it -d -p 9090:9090 -u root -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml" -v "$PWD/prometheus-data:/prometheus" --name=prom_cont prom/prometheus

docker run -it -d -p 9090:9090 -u root -v "/c/IISC\ AI\ and\ MLOps/course_materials/Capstone\ Project/project_template/driver_demand_prediction_capstone/Prometheus/prometheus.yml:/etc/prometheus/prometheus.yml" -v "/c/IISC\ AI\ and\ MLOps/course_materials/Capstone\ Project/project_template/driver_demand_prediction_capstone/Prometheus/prometheus-data:/prometheus" --name=prom_cont prom/prometheus

docker run -it -d -p 9090:9090 -u root -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\project_template\driver_demand_prediction_capstone\Prometheus\prometheus.yml:/etc/prometheus/prometheus.yml" -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\project_template\driver_demand_prediction_capstone\Prometheus\prometheus-data:/prometheus" --name=prom_cont prom/prometheus


docker exec -it prom_cont sh
	wget http://host.docker.internal:8001/api/v1/metrics

docker network ls
docker network inspect <network-name>



Launch Mlflow ui

curl http://host.docker.internal:8001/api-endpoint/api/v1/health

