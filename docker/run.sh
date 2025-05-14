#!/bin/bash

set -ex

# Debug: Print hostname to identify which region we're in
HOSTNAME=$(hostname)
echo "Current hostname: $HOSTNAME"

# Debug: Check if directories exist
echo "Checking if /home/ubuntu/balazs/cora exists: $(if [ -d "/home/ubuntu/balazs/cora" ]; then echo "YES"; else echo "NO"; fi)"
echo "Checking if /home/ubuntu/balazs-texas/cora exists: $(if [ -d "/home/ubuntu/balazs-texas/cora" ]; then echo "YES"; else echo "NO"; fi)"

# For the current instance, prioritize balazs-texas
if [ -d "/home/ubuntu/balazs-texas/cora" ]; then
  HOST_CORA_DIR="/home/ubuntu/balazs-texas/cora"
elif [ -d "/home/ubuntu/balazs/cora" ]; then
  HOST_CORA_DIR="/home/ubuntu/balazs/cora"
elif [ -d "/home/balazs/Documents/code/cora" ]; then
  HOST_CORA_DIR="/home/balazs/Documents/code/cora"
else
  echo "Error: Could not find cora directory in either /home/ubuntu/balazs or /home/ubuntu/balazs-texas"
  exit 1
fi

echo "Mounting from $HOST_CORA_DIR"

# Build the Docker image using sudo
sudo docker image build -t cora:latest -f ./DockerfileGH200 ..

# Run the Docker container with the detected directory using sudo, and capture its ID
CONTAINER_ID=$(sudo docker run --gpus all -d --restart unless-stopped \
  -v "$HOST_CORA_DIR:/home/cora/workspace" \
  cora:latest /bin/bash -c "sleep infinity")

# Check if the container started successfully and we have an ID
if [ -z "$CONTAINER_ID" ]; then
  echo "Error: Failed to start the Docker container or capture its ID."
  exit 1
fi

echo "Container started with ID: $CONTAINER_ID"

# Automatically exec into the container
echo "Attempting to exec into the container..."
sudo docker exec -it "$CONTAINER_ID" /bin/bash
