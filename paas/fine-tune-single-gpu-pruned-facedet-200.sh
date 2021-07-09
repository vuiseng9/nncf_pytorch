#!/usr/bin/env bash

# This is an example of single gpu training

RUNDIR=/workspace/face-detection-0200-paas-ft
FTCFG=/workspace/nncf/paas/cfg/face-detection-0200/face-detection-0200.filter_pruning_ft.yaml

DATA=/workspace/WiderFace
mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0
cd /workspace/nncf/paas/mmdetection_tools
python train.py \
    $FTCFG \
    --update_config \
    data.train.dataset.ann_file=$DATA/instances_train.json \
    data.train.dataset.img_prefix=$DATA \
    data.val.ann_file=$DATA/instances_val.json \
    data.val.img_prefix=$DATA \
    load_from=/workspace/nncf/paas/cfg/face-detection-0200/snapshot.pth \
    --work-dir $RUNDIR \
    --gpu-ids 0
