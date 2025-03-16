#!/bin/bash

# Script to build and run the Cora Docker container with GPU support
# Usage: ./run.sh [test|shell]

set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Cora Docker image with JAX 0.5+ and GPU support...${NC}"

# Build the image
docker image build -t cora:latest -f ./DockerfileGH200 ..

# Function to check if nvidia-smi is available
check_gpu() {
  if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected.${NC}"
    nvidia-smi
    return 0
  else
    echo -e "${RED}No NVIDIA GPU detected. Make sure NVIDIA drivers and CUDA are installed.${NC}"
    return 1
  fi
}

# Handle command line parameters
if [ "$1" == "test" ]; then
  # Run the GPU test script
  check_gpu || { echo -e "${YELLOW}Proceeding without GPU acceleration...${NC}"; }
  
  echo -e "${GREEN}Running Cora GPU test script...${NC}"
  docker run --gpus all -it \
    -v $(pwd)/../:/home/cora/workspace \
    cora:latest python /home/cora/workspace/gpu_test.py
elif [ "$1" == "shell" ] || [ -z "$1" ]; then
  # Run an interactive shell
  check_gpu || { echo -e "${YELLOW}Proceeding without GPU acceleration...${NC}"; }
  
  echo -e "${GREEN}Starting interactive shell in Cora Docker container...${NC}"
  docker run --gpus all -it \
    -v $(pwd)/../:/home/cora/workspace \
    cora:latest /bin/bash
else
  echo -e "${RED}Unknown parameter: $1${NC}"
  echo -e "Usage: ./run.sh [test|shell]"
  exit 1
fi
