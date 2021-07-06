#!/usr/bin/env bash

# This is an example of single gpu training

RUNDIR=/workspace/mobilenet_v2-paas-ft
DATA=/data/dataset/imagenet/ilsvrc2012/torchvision
mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0
cd /workspace/nncf/examples/classification
python main.py \
    -m train \
    --gpu-id 0 \
    --log-dir $RUNDIR \
    --config /workspace/nncf/paas/cfg/mobilenet_v2_filter_paas_ft.json \
    --data $DATA
