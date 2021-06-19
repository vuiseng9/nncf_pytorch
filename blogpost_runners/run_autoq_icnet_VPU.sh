#!/usr/bin/env bash

# Assume following codebase
# git clone --branch autoq_branch https://github.com/vuiseng9/nncf

WORKDIR=/home/vchua/nncf-pre1.8/nncf/examples/torch/semantic_segmentation

RUN_ROOT=/home/vchua/regression-21WW25/nncf-pre1p8_blogpost
RUN_DIR=21WW25.6.A_icnet_nncf-pre1p8_nonpad_autoq_0.14

NNCF_CFG=/home/vchua/nncf-pre1.8/nncf/blogpost_configs/icnet_camvid_autoq_VPU_0.14.json

cd ${WORKDIR}

# CUDA_VISIBLE_DEVICES=0,1 
nohup python main.py \
    -m test \
    --gpu-id 2 \
    --config ${NNCF_CFG} \
    --data /data/dataset/CamVid \
    --dataset camvid \
    --weight ~/nncf-zoo/crawler/nncf-zoo/icnet_camvid.pth \
    --workers 6 \
    --log-dir ${RUN_ROOT}/${RUN_DIR} &
