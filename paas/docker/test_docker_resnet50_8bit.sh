#!/usr/bin/env bash
#export PYTHONPATH=/workspace/nncf-nemo-ft

WORKDIR=/workspace/nncf/examples/classification

RUN_ROOT=/tmp/dry-runs/
RUN_DIR_LABEL=nncf-resnet50-I8

NNCF_CFG=${WORKDIR}/configs/quantization/resnet50_imagenet_int8.json

DATASET_ROOT=/data/dataset/imagenet/ilsvrc2012/torchvision
cd ${WORKDIR}

# CUDA_VISIBLE_DEVICES=0,1 
python main.py \
    -m test \
    --gpu-id 0 \
    --config ${NNCF_CFG} \
    --data ${DATASET_ROOT} \
    --log-dir ${RUN_ROOT}/${RUN_DIR_LABEL}
