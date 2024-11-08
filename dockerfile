# Use the official Python image as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the application runs on (if you are running a FastAPI server, for example)
EXPOSE 8000

# Define environment variables (optional, if you need any specific configs)
# ENV ENV_VAR_NAME value

# Run the main script
CMD ["python", "api/main.py"]
