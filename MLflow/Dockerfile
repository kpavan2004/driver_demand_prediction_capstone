FROM python:3.10

# Install dependencies
RUN pip install --no-cache-dir mlflow boto3

# Set working directory
# WORKDIR /app

# Expose port for MLflow UI
EXPOSE 5000

# Default command to run MLflow server
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow/mlflow.db", "--artifacts-destination", "s3://pk-capstone-bucket-01/MLflow-data/"]

