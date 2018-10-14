#!/bin/bash

# paths
WORKDIR=/root/work

# image disk
IMAGE=denden047/deeprobotics
TAG=latest

# command
if [ $1 = "train" ]; then
    now=$(date "+%Y%m%d%H%M")
    PORTS="-p 6006:6006"
    # 普通に実行する
    RUN_CMD="python3 train.py \
        --save ${LOG_DIR}/policy \
        --tensorboard ${LOG_DIR}/tensorboard_${now} \
        --monitor ${LOG_DIR}/monitor \
        --save-episodes 100 \
        --timesteps 5000000
    "
elif [ $1 = "eval" ]; then
    # vnc内で実行する
    RUN_CMD="python3 evaluate.py"
    cd src
    export "CUDA_VISIBLE_DEVICES"="-1"
    RUN_CMD="python3 train.py \
        --timesteps 100 \
        --frame-time 0.01 \
        --save ${LOG_DIR}/policy \
        --load ${LOG_DIR}/policy/final \
        --visualize \
        --play
    "
    ${RUN_CMD}
    exit 0
elif [ $1 = "check" ]; then
    # vnc内で実行する
    cd src
    RUN_CMD="python3 train.py \
        --save-episodes 5000 \
        --timesteps 10000000 \
        --frame-time 0.01 \
        --visualize \
        --debug \
        --play
    "
    ${RUN_CMD}
    exit 0
elif [ $1 = "vnc" ]; then
    PORTS="-p 6080:80 -p 6081:443"
    RUN_CMD="/startup.sh"
else
    RUN_CMD=""
fi

# run
nvidia-docker run -it --rm \
    ${PORTS} \
    -v ${PWD}:${WORKDIR}:ro \
    -v /data2/naoya:/data \
    -w ${WORKDIR}/src \
    ${IMAGE}:${TAG} \
    ${RUN_CMD}