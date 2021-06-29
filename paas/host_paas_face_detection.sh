# This current settings below correspond to containerized environment.

#!/usr/bin/env bash

export workload=facedet
export config=/workspace/nncf/paas/cfg/face-detection-0200/face-detection-0200.filter_pruning.yaml

# Pls revise path to dataset
export data=/data/dataset/WiderFace

# Pls revise path to pretrained model
# Download from https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0200.pth
export ckpt=/workspace/nncf/paas/cfg/face-detection-0200/snapshot.pth

WORKDIR=/workspace/nncf

cd $WORKDIR
python paas/manage.py \
        run \
        --no-reload \
        --host 0.0.0.0 


