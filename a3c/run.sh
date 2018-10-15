#!/bin/bash

# paths
WORKDIR=/root/work

# image disk
IMAGE=denden047/deeprobotics
TAG=latest

# command
if [ $1 = "train" ]; then
    PORTS="-p 6006:6006"
    # 普通に実行する
    RUN_CMD="python3 train.py"
    exit 0
elif [ $1 = "vnc" ]; then
    PORTS="-p 6080:80 -p 6081:443"
    RUN_CMD="/startup.sh"
else
    RUN_CMD=""
fi

# run
docker run -it --rm \
    ${PORTS} \
    -v ${PWD}:${WORKDIR}:ro \
    -v /data2/naoya:/data \
    -w ${WORKDIR}/src \
    ${IMAGE}:${TAG} \
    ${RUN_CMD}