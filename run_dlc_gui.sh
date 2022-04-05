#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/deeplabcut_gui"

docker build -f "$CURRENT_PATH"/docker/Dockerfile.deeplabcut.gui -t ${IMAGE_NAME} . && \
docker run -it --rm \
    --gpus 1 \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -v "$CURRENT_PATH":/workdir \
    -v "$CURRENT_PATH"/data:/data \
    -e VNC_PASSWORD=mypassword \
    -p 2351:8888 \
    -p 6080:80 \
    ${IMAGE_NAME} \
    /bin/bash