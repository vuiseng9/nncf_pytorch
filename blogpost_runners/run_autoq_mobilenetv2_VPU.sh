#!/usr/bin/env bash

# Assume following codebase
# git clone --branch autoq_branch https://github.com/vuiseng9/nncf

WORKDIR=/home/vchua/nncf-pre1.8/nncf/examples/torch/classification

RUN_ROOT=/home/vchua/regression-21WW25/nncf-pre1p8_blogpost
RUN_DIR=21WW25.6.A_mobilenet_v2_nncf-pre1p8_nonpad_autoq_0.15

NNCF_CFG=/home/vchua/nncf-pre1.8/nncf/blogpost_configs/mobilenet_v2_autoq_VPU_0.15.json

cd ${WORKDIR}

# CUDA_VISIBLE_DEVICES=0,1 
nohup python main.py \
    -m test \
    --gpu-id 1 \
    --config ${NNCF_CFG} \
    --data /data/dataset/imagenet/ilsvrc2012/torchvision  \
    --log-dir ${RUN_ROOT}/${RUN_DIR} &



