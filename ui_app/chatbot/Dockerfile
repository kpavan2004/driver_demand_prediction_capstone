# Step 1: Use an official Python base image
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only the backend requirements to install dependencies
COPY backend/requirements.txt ./backend/requirements.txt

# Step 4: Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r ./backend/requirements.txt

# Step 5: Copy the entire chatbot directory into the container
COPY . .

# Step 6: Expose the port your application runs on (adjust as needed)
EXPOSE 8000

# Step 7: Define the default command to run your application
# Update the command below based on your backend server (e.g., Flask, FastAPI)
CMD ["tail", "-f", "/dev/null"]
