#!/bin/bash

set -ex

# Debug: Print hostname to identify which region we're in
HOSTNAME=$(hostname)
echo "Current hostname: $HOSTNAME"

# Set the host directory for Windows
HOST_CORA_DIR="/c/Users/feher/OneDrive/Dokumentumok/code/cora/"

# Debug: Check if the specified directory exists
echo "Checking if $HOST_CORA_DIR exists: $(if [ -d "$HOST_CORA_DIR" ]; then echo "YES"; else echo "NO"; fi)"

if [ ! -d "$HOST_CORA_DIR" ]; then
  echo "Error: Could not find cora directory at $HOST_CORA_DIR"
  exit 1
fi

echo "Mounting from $HOST_CORA_DIR"

# Build the Docker image
UV_HTTP_TIMEOUT=300s docker image build -t cora:latest -f ./DockerfileGH200 ..

# Run the Docker container with the detected directory, and capture its ID
CONTAINER_ID=$(docker run --gpus all -d --restart unless-stopped \
  -v "$HOST_CORA_DIR:/home/cora/workspace" \
  cora:latest //bin/bash -c "sleep infinity")

# Check if the container started successfully and we have an ID
if [ -z "$CONTAINER_ID" ]; then
  echo "Error: Failed to start the Docker container or capture its ID."
  exit 1
fi

echo "Container started with ID: $CONTAINER_ID"

# Give the container a moment to initialize
sleep 5

# Automatically exec into the container
echo "Attempting to exec into the container..."
docker exec -it "$CONTAINER_ID" //bin/bash 