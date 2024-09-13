# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Give execute permission to the ollama_server.sh script
RUN chmod +x /app/ollama_server.sh

# Run the ollama server script and then start FastAPI app
CMD ["/app/ollama_server.sh", "&&", "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
