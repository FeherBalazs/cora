#!/bin/bash

set -ex

docker image build -t cora:latest -f ./DockerfileGH200 ..
docker run --gpus all -it \
  -v /home/ubuntu/balazs/cora:/home/cora/workspace \
  cora:latest /bin/bash
