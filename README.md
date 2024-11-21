1.Train the model
Mlflow server should be launched first using : and then change port visibility from private to public
     a)if running locally then export RE_TRAIN='Yes
     mlflow ui --host 0.0.0.0 --port 5000 :- Runs on localhost

    b)Launch codespace and install boto3 and mlflow, export variables, create s3 bukcet and folder and 
     If artefacts target is s3: then export the varables 
     export AWS_ACCESS_KEY_ID=''
     export AWS_SECRET_ACCESS_KEY=''
     export AWS_DEFAULT_REGION='ap-south-1'
     mlflow ui --port 5000 --artifacts-destination s3://pk-capstone-bucket-01/MLflow-data/
cd /delivery_time_model
python train_pipeline.py

Initialize DVC repository:
Create a new GitHub repo

Start a codespace

Upload data (data/train.csv)

Create and activate venv

Install packages:
pip install dvc==3.55.2 dvc-s3==3.2.0

Initialize dvc repository:
dvc init

Configure Remote Storage:
Add remote storage:
dvc remote add -d myremote s3://s3://pk-capstone-bucket-01/DVC-data/

Provide AWS credentials: (will be stored locally)
dvc remote modify --local myremote access_key_id <>
dvc remote modify --local myremote secret_access_key <>

Commit remote-storage config:
git add .dvc/config
git commit -m "remote storage configured"
git push

Data Versioning:
Add data for dvc tracking:
dvc add data/train.csv

Add metadata files for git tracking:
git add data/.gitignore data/train.csv.dvc

Commit changes in git:
git commit -m "Initial data"

Tag the data:
git tag -a v1.1 -m "Dataset v1.1"
The git tag command in Git is used to create a reference to a specific point in your repository’s history, typically to mark a particular commit as important. Tags are often used to denote releases (e.g., version 1.0, 2.0, etc.).

Push data to remote-rtorage:
dvc push

Push metadata file to git repo:
git push

Push tag:
git push origin v1.1

Whenever new data comes in, use the below commands in sequence:
dvc status
dvc add data/titanic.csv
git add data/titanic.csv.dvc
git commit -m "dataset updated"
git tag -a v1.x -m "Dataset v1.x"
dvc push
git push
git push origin v1.x


2.predict:
python predict.p

3.tests:
cd /tests/
pytest -s test_predictions.py

4.create the .whl file and .pkl file
python -m build 

5.copy the .whl file to _api
cp /delivery_time_model/dsit/*.whl /delivery_time_model_api/

6.Deploy model on fastapi
cd /delivery_time_model_api/api
python main.py

7.Dockerize the application:
docker build . -t kpavan2004/driver_delivery_time_pred-fastapi:latest

8.Run below command to run a container from your image and then change port visibility from private to public
docker run -it -d -p 8001:8001 --name=app_cont kpavan2004/driver_delivery_time_pred-fastapi

9. From the Prometheus directory, run the below command to start Prometheus and then change port visibility from private to public
docker run -it -d -p 9090:9090 -u root -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml" -v "$PWD/prometheus-data:/prometheus" --name=prom_cont prom/prometheus

10.Go to Grafana directory, run the below command to start grafana and then change port visibility from private to public
 docker run -it -d -p 3000:3000 -u root -v "$PWD/grafana-data:/var/lib/grafana" --env-file "$PWD/env.list" --name=grafana_cont grafana/grafana-oss



