# Use Python 3.12.3 as the base image
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and debugging tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Print Python version and pip list for debugging
RUN python --version && pip list

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application with more verbose logging
CMD ["python", "-u", "main.py"]