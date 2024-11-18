1.Train the model
cd /delivery_time_model
python train_pipeline.py

2.predict:
python predict.py

3.tests:
cd /tests/app/
python main.py

4.create the .whl file
python -m build 

5.copy the .whl file to _api
cp /delivery_time_model/dsit/*.whl /delivery_time_model_api/

6.Deploy model on fastapi
cd /delivery_time_model_api/api
python main.py

7.Dockerize the application:
docker build . -t kpavan2004/driver_delivery_time_pred-fastapi:latest

8.Run below command to run a container from your image
docker run -it -d -p 8001:8001 --name=app_cont kpavan2004/driver_delivery_time_pred-fastapi

9.

From the Prometheus directory, run the below command to start Prometheus
docker run -it -d -p 9090:9090 -u root -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml" -v "$PWD/prometheus-data:/prometheus" --name=prom_cont prom/prometheus

docker run -it -d -p 9090:9090 -u root -v "/c/IISC\ AI\ and\ MLOps/course_materials/Capstone\ Project/project_template/driver_demand_prediction_capstone/Prometheus/prometheus.yml:/etc/prometheus/prometheus.yml" -v "/c/IISC\ AI\ and\ MLOps/course_materials/Capstone\ Project/project_template/driver_demand_prediction_capstone/Prometheus/prometheus-data:/prometheus" --name=prom_cont prom/prometheus

docker run -it -d -p 9090:9090 -u root -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\project_template\driver_demand_prediction_capstone\Prometheus\prometheus.yml:/etc/prometheus/prometheus.yml" -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\project_template\driver_demand_prediction_capstone\Prometheus\prometheus-data:/prometheus" --name=prom_cont prom/prometheus

nohup mlflow ui --port 5000 &

mlflow ui --backend-store-uri sqlite:///mlflow.db


docker run --name postgres_cont -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgress -e POSTGRES_DB=mydb -p 5430:5430 -d postgres

docker cp ./pg_hba.conf postgres-container:/var/lib/postgresql/data/pg_hba.conf
docker cp ./pg_hba.conf postgres_cont:/var/lib/postgresql/data/pg_hba.conf
docker restart postgres_cont


docker exec -it prom_cont sh
	wget http://host.docker.internal:8001/api/v1/metrics

docker network ls
docker network inspect <network-name>

docker run -it -d -p 3000:3000 -u root -v "$PWD/grafana-data:/var/lib/grafana" --env-file "$PWD/env.list" --name=grafana_cont grafana/grafana-oss

Launch Mlflow ui

curl http://host.docker.internal:8001/api-endpoint/api/v1/health


docker run -d -p 5000:5000 \
  -e MLFLOW_SERVER_FILE_STORE=/mlflow/mlruns \
  -e MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts \
  -v $(pwd)/mlruns:/mlflow/mlruns \
  -v $(pwd)/artifacts:/mlflow/artifacts \
  --name mlflow_server ghcr.io/mlflow/mlflow


