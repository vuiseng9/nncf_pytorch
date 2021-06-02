#!/usr/bin/env bash

container=nncf-paas

docker run \
    -d \
    -it \
    --gpus all \
    --shm-size 8G \
    -v /data:/data \
    -p 15015:5000 \
    -p 16016:6006 \
    $container bash
