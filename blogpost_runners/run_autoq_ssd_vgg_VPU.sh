#!/usr/bin/env bash

# Assume following codebase
# git clone --branch autoq_branch https://github.com/vuiseng9/nncf

WORKDIR=/home/vchua/mar25-nncf1.6/nncf/examples/object_detection

RUN_ROOT=/home/vchua/regression-21WW10/nncf1p7_blogpost/
RUN_DIR=21WW17.1.A_ssd_vgg_nncf1p7_nonpad_autoq_0.15_AutoQ

NNCF_CFG=/home/vchua/mar25-nncf1.6/nncf/blogpost_configs/ssd_vgg_voc_autoq_VPU_0.15.json

cd ${WORKDIR}

# CUDA_VISIBLE_DEVICES=0,1 
nohup python main.py \
    -m train \
    --gpu-id 3 \
    --config ${NNCF_CFG} \
    --data /data/dataset/VOC/VOCdevkit \
    --weight ~/nncf-zoo/crawler/nncf-zoo/ssd300_vgg_voc.pth \
    --workers 6 \
    --log-dir ${RUN_ROOT}/${RUN_DIR} &


