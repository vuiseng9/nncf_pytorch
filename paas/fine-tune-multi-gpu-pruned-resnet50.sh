#!/usr/bin/env bash

# This is an example of multi-gpu training

RUNDIR=/workspace/resnet50-paas-ft
DATA=/data/dataset/imagenet/ilsvrc2012/torchvision
mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /workspace/nncf/examples/classification
python main.py \
    -m train \
    --multiprocessing-distributed \
    --log-dir $RUNDIR \
    --config /workspace/nncf/paas/cfg/resnet50_filter_paas_ft.json \
    --data $DATA