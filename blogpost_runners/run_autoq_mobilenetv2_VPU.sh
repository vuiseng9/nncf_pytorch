#!/usr/bin/env bash

# Assume following codebase
# git clone --branch autoq_branch https://github.com/vuiseng9/nncf

WORKDIR=/home/vchua/mar25-nncf1.6/nncf/examples/classification

RUN_ROOT=/home/vchua/regression-21WW10/nncf1p7_blogpost/
RUN_DIR=21WW17.1.A_mobilenet_v2_nncf1p7_nonpad_autoq_0.15

NNCF_CFG=/home/vchua/mar25-nncf1.6/nncf/blogpost_configs/mobilenet_v2_autoq_VPU_0.15.json

cd ${WORKDIR}

# CUDA_VISIBLE_DEVICES=0,1 
nohup python main.py \
    -m train \
    --gpu-id 1 \
    --config ${NNCF_CFG} \
    --data /data/dataset/imagenet/ilsvrc2012/torchvision  \
    --log-dir ${RUN_ROOT}/${RUN_DIR} &


