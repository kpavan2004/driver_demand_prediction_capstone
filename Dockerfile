# Use this image for development work of Food delivery prediction project

# Use the full Python 3.10 base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install vim and other required packages
RUN apt-get update && apt-get install -y vim && apt-get clean

# Create a virtual environment
RUN python -m venv /app/venv

# Activate the virtual environment and upgrade pip
RUN /app/venv/bin/pip install --upgrade pip

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Update pip and install dependencies with retries
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=120 -r requirements.txt \
    || pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Ensure the virtual environment is activated by default for interactive shells
RUN echo "source /app/venv/bin/activate" >> /root/.bashrc

# Keep all ports open (for dynamic port usage)
EXPOSE 0-65535

# Set the default command to keep the container running
CMD ["tail", "-f", "/dev/null"]

