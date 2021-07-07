

import torchvision
from copy import deepcopy
import json
import os

def jsonfile2dict(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError('{} does not exist.'.format(path))

class BaseFTCfg():
    def __init__(self, model: str):
        if model in torchvision.models.__dict__:
            self.ftcfg = deepcopy(self.get_base_ft_cfg())
            self.set_model_name(model)
        else:    
            raise ValueError('{} is not part of torchvision architecture list for imagenet'.format(model))
    
    def __repr__(self):
        return json.dumps(self.ftcfg, indent=4)
    
    def set_model_name(self, model_name: str):
        if not isinstance(model_name, str):
            raise TypeError('model_name input is not string')
        self.ftcfg['model'] = model_name

    def set_batch_size(self, batchsize: int):
        if not isinstance(batchsize, int):
            raise TypeError('batchsize input is not integer')
        self.ftcfg['batch_size'] = batchsize
    
    def set_total_epoch(self, n_epoch: int):
        if not isinstance(n_epoch, int):
            raise TypeError('n_epoch input is not integer')
        self.ftcfg['epochs'] = n_epoch

    def set_dist_processing(self, boolean :bool):
        if not isinstance(boolean, bool):
            raise TypeError('input is not boolean')
        self.ftcfg["multiprocessing_distributed"] = boolean

    def set_compression_cfg(self, paas_ft_cfg: dict):
        if not isinstance(paas_ft_cfg, dict):
            raise TypeError('set_compression_cfg only accepts dictionary as input')
        self.ftcfg['compression'] = paas_ft_cfg

    def set_optimizer_cfg(self, optim_cfg: dict):
        if not isinstance(optim_cfg, dict):
            raise TypeError('set_optimizer_cfg only accepts dictionary as input')
        self.ftcfg['optimizer'] = optim_cfg

    def serialize(self, path):
        if len(os.path.basename(path)) <= 0 :
            raise ValueError('{} is an invalid path.'.format(path))
        if self.has_none_value():
            raise ValueError('Null values found in ft cfg, pls check.\n{}'.format(self))

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.ftcfg, f, indent=4)

    def has_none_value(self):
        def _checknone(obj):
            if isinstance(obj, list):
                for e in obj:
                    item = _checknone(e) 
                    if item is True:
                        return True
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    item = _checknone(v)
                    if item is True:
                        return True
            if obj is None:
                return True
            return False
        return _checknone(self.ftcfg)

    def get_base_ft_cfg(self):
        base_ft_cfg = \
        {
            "restful": False,
            "model": None,
            "pretrained": True,
            "input_info": {
            "sample_size": [1, 3, 224, 224]
            },
            "batch_size" : None,
            "workers": 6,
            "epochs": 15,
            "multiprocessing_distributed": True,
            "optimizer": {
                "base_lr": 0.00031,
                "schedule_type": "exponential",
                "gamma": 0.95, 
                "type": "Adam"
            },
        "compression": None
        }
        return base_ft_cfg

class MobileNetV2_FTCfg(BaseFTCfg):
    def __init__(self):
        super().__init__('mobilenet_v2')
        self.set_batch_size(200) # about 8.2GB per card when training on 2 cards
        self.set_total_epoch(15)
        self.set_dist_processing(True)

class ResNet50_FTCfg(BaseFTCfg):
    def __init__(self):
        super().__init__('resnet50')
        self.set_batch_size(256) # about 8.1GB per card when training on 4 cards
        self.set_total_epoch(15)
        self.set_dist_processing(True)

class ResNet101_FTCfg(BaseFTCfg):
    def __init__(self):
        super().__init__('resnet101')
        self.set_batch_size(56)
        self.set_total_epoch(15)
        self.set_dist_processing(True)
