import json
import numpy as np
import pandas as pd

#rootdir='/data/dataset/WiderFace/'

pth = '/'.join([rootdir, 'instances_train.json'])

with open(pth, 'r') as f:
    d = json.load(f)

img_samples=[]
for img in d['images']:
    if img['id']%4==0:
        img_samples.append(img)

ann_samples=[]
for ann in d['annotations']:
    if ann['id']%4==0:
        ann_samples.append(ann)

quarter_train=dict(
    images=img_samples,
    categories=d['categories'],
    annotations=ann_samples
)

for k,v in quarter_train.items():
    print(len(v), k)

with open('/'.join([rootdir, 'instances_train_quarter.json']), 'w') as f:
    json.dump(quarter_train, f, indent=4)

