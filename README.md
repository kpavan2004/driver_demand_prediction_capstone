# Driver Demand Prediction Capstone

## Project Overview

The food delivery industry faces significant challenges in predicting driver demand and accurately estimating delivery times. These issues affect:

- **Customer satisfaction**
- **Operational costs**
- **Overall service reliability**

Inaccurate predictions lead to under- or over-supply of drivers, delivery delays, and unsatisfactory customer experiences.

This project aims to develop a machine learning-based system to:

1. **Optimize delivery charges**
2. **Predict driver demand**

### Full End-to-End Stack Development Includes:
- **Web application** for Administrator and Customer
- **Chatbot** for Administrator and Customer
- **Trained ML Models**
- **CI/CD Pipelines**
- Integration with **Monitoring Tools**
- Infrastructure tools like **Kubernetes**, **AWS**, and **Docker**

---

## Sources and References
[driver_demand_prediction_capstone](https://github.com/kpavan2004/driver_demand_prediction_capstone)

[food_delivery_usecase](https://github.com/kevalkamani/food_delivery_usecase)

---

## Local Setup Instructions

### Build environment for development and enhancement of this project
- Take the docker image from mahan0227/food-delivery-predict , this has all the required packages and dependencies including latest code. There is github action workflow created to build image every day and push to hub

```bash
docker pull mahan0227/food-delivery-predict
```

### Run pulled docker image on your machine considering docker engine, docker desktop installed in it

```bash
docker run -d -it mahan0227/food-delivery-predict
```

### Confirm this repo code exist in it or copy up to date code to container

```bash
docker exec -it <running container id> ls /app
```

### Python virtual environment has been set at bash by default

---

## MLflow Setup

### Launch the MLflow Server
The MLflow server should be launched first. Change the port visibility from private to public if required.

Pull the Docker image and run the MLflow server:
```bash
docker run -it -d -p 5000:5000 -u root \
  -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\dev\driver_demand_prediction_capstone\MLflow\mlflow.db:/mlflow/mlflow.db" \
  -v "C:\IISC_AI_and_MLOps\course_materials\Capstone_Project\dev\driver_demand_prediction_capstone\MLflow\mlartifacts/:/mlflow/mlartifacts/" \
  --name=mlflow_cont kpavan2004/mlflow_server_dev
```
> **Note:** Update the local folder paths in the `-v` argument based on your setup.

### Verify MLflow UI
Verify if the MLflow UI is running:
[http://localhost:5000](http://localhost:5000)

---

## Workflow

### Upload the Best Model to MLflow
```bash
cd notebooks/
python Log_Best_Exp_Model.py
```

## Train the Model

<span style="color:blue">To train the model, navigate to the directory</span>

```bash
cd /delivery_time_model
python train_pipeline.py


### Run Inference
```bash
cd /delivery_time_model
python predict.py
```

### Testing
```bash
pytest
```

### Build the Project
```bash
python -m build
```
This creates `.whl` and `.pkl` files in the `/dist` folder.

### Copy the `.whl` File to API Folder
```bash
cp /delivery_time_model/dist/*.whl /delivery_time_model_api/
```

### Deploy the API
```bash
cd /delivery_time_model_api/api
python main.py
```

---

## Project Features

- **Predictive Modeling:** Optimize delivery charges and driver demand predictions.
- **Web Application:** Interactive interfaces for administrators and customers.
- **ML Models:** End-to-end ML pipelines and inference systems.
- **DevOps Integration:** CI/CD pipelines, monitoring, and scalable infrastructure.

---

## Tools and Technologies

- **Programming Language:** Python
- **Machine Learning Tools:** MLflow, Scikit-learn, Pandas
- **Infrastructure:** Docker, Kubernetes, AWS
- **Version Control:** Git
- **Testing Frameworks:** Pytest

---

## License
[Specify the license here, e.g., MIT License]

