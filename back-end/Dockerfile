# Using a slim version for a smaller base image
FROM python:3.11-slim-bullseye

RUN apt-get update \
    && apt-get install libmagic1 swig wget -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Increase timeout to wait for the new installation
RUN pip install --no-cache-dir -r requirements.txt --timeout 200

# Copy source code (for production, but will be overridden by bind mount in development)
COPY . .

# Expose port
EXPOSE 8000

# Set Python path to include the src directory
ENV PYTHONPATH=/app
