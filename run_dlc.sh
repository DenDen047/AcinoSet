#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/deeplabcut"

# AcinoSet
docker build -f "$CURRENT_PATH"/docker/Dockerfile.deeplabcut -t ${IMAGE_NAME} . && \
docker run -it --rm \
    --gpus 1 \
    -v "$CURRENT_PATH"/src:/app \
    -v "$CURRENT_PATH"/configs:/configs \
    -v "$CURRENT_PATH"/data:/data \
    -w /app \
    ${IMAGE_NAME} \
    bash