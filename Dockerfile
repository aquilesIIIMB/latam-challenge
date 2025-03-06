# syntax=docker/dockerfile:1.2
# Use a lightweight Python base image
FROM python:3.9-slim

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libev-dev \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements (adjust the file name/path if needed)
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Expose port 8080 (Cloud Run expects port 8080 by default)
EXPOSE 8080

# Start the API using uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
