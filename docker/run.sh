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
else
  echo "Error: Could not find cora directory in either /home/ubuntu/balazs or /home/ubuntu/balazs-texas"
  exit 1
fi

echo "Mounting from $HOST_CORA_DIR"

# Build the Docker image using sudo
sudo docker image build -t cora:latest -f ./DockerfileGH200 ..

# Run the Docker container with the detected directory using sudo
sudo docker run --gpus all -it \
  -v "$HOST_CORA_DIR:/home/cora/workspace" \
  cora:latest /bin/bash
