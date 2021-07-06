#!/usr/bin/env bash

cd ..
#docker build . -f docker/Dockerfile -t nncf-paas
docker build . -f docker/Dockerfile --no-cache -t autopaas:v0.5.5-facedet
