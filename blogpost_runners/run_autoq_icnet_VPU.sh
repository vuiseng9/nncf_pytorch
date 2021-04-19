#!/usr/bin/env bash

# Assume following codebase
# git clone --branch autoq_branch https://github.com/vuiseng9/nncf

WORKDIR=/home/vchua/mar25-nncf1.6/nncf/examples/semantic_segmentation

RUN_ROOT=/home/vchua/regression-21WW10/nncf1p7_blogpost/
RUN_DIR=21WW17.1.A_icnet_nncf1p7_nonpad_autoq_0.14

NNCF_CFG=/home/vchua/mar25-nncf1.6/nncf/blogpost_configs/icnet_camvid_autoq_VPU_0.14.json

cd ${WORKDIR}

# CUDA_VISIBLE_DEVICES=0,1 
nohup python main.py \
    -m train \
    --gpu-id 2 \
    --config ${NNCF_CFG} \
    --data /data/dataset/CamVid \
    --dataset camvid \
    --weight ~/nncf-zoo/crawler/nncf-zoo/icnet_camvid.pth \
    --workers 6 \
    --log-dir ${RUN_ROOT}/${RUN_DIR} &
