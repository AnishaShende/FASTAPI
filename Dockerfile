# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Update package list and install curl and bash
RUN apt-get update && apt-get install -y curl bash

# Give execute permission to the ollama_server.sh script
RUN chmod +x /app/ollama_server.sh

# Expose port 8000
EXPOSE 8000

# Use a shell form to execute multiple commands with a health check loop for Ollama server
CMD /bin/bash -c "/app/ollama_server.sh && uvicorn fastapi_app:app --host 0.0.0.0 --port 8000"
