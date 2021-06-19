#!/usr/bin/env bash

# Assume following codebase
# git clone --branch autoq_branch https://github.com/vuiseng9/nncf

WORKDIR=/home/vchua/nncf-pre1.8/nncf/examples/torch/classification

RUN_ROOT=/home/vchua/regression-21WW25/nncf-pre1p8_blogpost
RUN_DIR=21WW25.6.A_resnet50_nncf-pre1p8_nonpad_autoq_0.13

NNCF_CFG=/home/vchua/nncf-pre1.8/nncf/blogpost_configs/resnet50_autoq_VPU_0.13.json

cd ${WORKDIR}

nohup python main.py \
    --gpu-id 0 \
    -m test \
    --config ${NNCF_CFG} \
    --data /data/dataset/imagenet/ilsvrc2012/torchvision  \
    --log-dir ${RUN_ROOT}/${RUN_DIR} &
