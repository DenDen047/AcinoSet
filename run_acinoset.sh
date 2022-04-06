#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/acinoset:latest"

# AcinoSet
docker build -f "$CURRENT_PATH"/docker/Dockerfile.acinoset -t ${IMAGE_NAME} . && \
docker run -it --rm \
    -v "$CURRENT_PATH"/src:/workdir \
    -v "$CURRENT_PATH"/configs:/configs \
    -v "$CURRENT_PATH"/data:/data \
    -w /workdir \
    --name naoya_test \
    ${IMAGE_NAME} \
    /bin/bash

# python all_optimizations.py --data_dir /data/2017_08_29/bottom/zorro/flick2 && \
# python all_optimizations.py --data_dir /data/2017_08_29/top/jules/run1_1 && \
# python all_optimizations.py --data_dir /data/2017_08_29/top/jules/run1_2 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_08_29/top/phantom/run1_1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_02/bottom/jules/flick2_1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_02/bottom/jules/run2 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_02/top/phantom/run1_1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_02/top/phantom/run1_2 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_02/top/phantom/run1_3 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_03/bottom/zorro/run2_1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_03/bottom/zorro/run2_2 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_03/top/phantom/flick1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_03/top/phantom/run1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_03/top/zorro/flick1_1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_03/top/zorro/run1_1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_09_03/top/zorro/run1_2 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_09/bottom/jules/flick2 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_09/top/jules/run1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_10/top/phantom/run1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_10/top/zorro/flick1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_12/top/cetane/run1_1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_12/top/cetane/run1_2 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_16/top/cetane/run1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2017_12_16/top/phantom/run1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_02_27/kiara/flick && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_02_27/kiara/run && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_02_27/romeo/flick && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_02_27/romeo/run && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_03_05/lily/flick && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_03_05/lily/run && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_03_07/menya/run && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_03_09/jules/flick1 && \
# python all_optimizations.py --dlc dlc_pw --data_dir /data/2019_03_09/lily/run
