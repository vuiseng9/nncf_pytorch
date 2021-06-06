# This current settings below correspond to containerized environment.

#!/usr/bin/env bash

export workload=imgnet
export config=/workspace/nncf/paas/cfg/vgg11_filter_paas.json

# Pls revise path to dataset
export data=/data/dataset/imagenet/ilsvrc2012/torchvision

WORKDIR=/workspace/nncf

cd $WORKDIR
python paas/manage.py \
        run \
        --no-reload \
        --host 0.0.0.0 


