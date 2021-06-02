#!/usr/bin/env bash

cd ..
docker build . -f docker/Dockerfile -t nncf-paas
