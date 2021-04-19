#!/usr/bin/env bash

# Assume following codebase
# git clone --branch autoq_branch https://github.com/vuiseng9/nncf

WORKDIR=/home/vchua/mar25-nncf1.6/nncf/examples/classification

RUN_ROOT=/home/vchua/regression-21WW10/nncf1p7_blogpost/
RUN_DIR=21WW17.1.A_resnet50_nncf1p7_nonpad_autoq_0.13

NNCF_CFG=/home/vchua/mar25-nncf1.6/nncf/blogpost_configs/resnet50_autoq_VPU_0.13.json

cd ${WORKDIR}

nohup python main.py \
    --gpu-id 0 \
    -m train \
    --config ${NNCF_CFG} \
    --data /data/dataset/imagenet/ilsvrc2012/torchvision  \
    --log-dir ${RUN_ROOT}/${RUN_DIR} &
