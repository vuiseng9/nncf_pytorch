

import torchvision
from copy import deepcopy
import json
import os
import yaml

def yamlfile2dict(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.load(f)
    else:
        raise ValueError('{} does not exist.'.format(path))

def dict2yamlnfile(d, path):
    if len(os.path.basename(path)) <= 0 :
        raise ValueError('{} is an invalid path.'.format(path))
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(d, f, indent=4)

class FaceDet200FTCfg():
    def __init__(self):
        self.ftcfg = deepcopy(self.get_base_ft_cfg())
    
    def __repr__(self):
        return json.dumps(self.ftcfg, indent=4)
    
    # def set_model_name(self, model_name: str):
    #     if not isinstance(model_name, str):
    #         raise TypeError('model_name input is not string')
    #     self.ftcfg['model'] = model_name

    # def set_batch_size(self, batchsize: int):
    #     if not isinstance(batchsize, int):
    #         raise TypeError('batchsize input is not integer')
    #     self.ftcfg['batch_size'] = batchsize
    
    def set_total_epoch(self, n_epoch: int):
        if not isinstance(n_epoch, int):
            raise TypeError('n_epoch input is not integer')
        self.ftcfg['total_epochs'] = n_epoch

    # def set_dist_processing(self, boolean :bool):
    #     if not isinstance(boolean, bool):
    #         raise TypeError('input is not boolean')
    #     self.ftcfg["multiprocessing_distributed"] = boolean

    def set_compression_params(self, paas_ft_cfg: dict):
        if not isinstance(paas_ft_cfg, dict):
            raise TypeError('set_compression_cfg only accepts dictionary as input')
        self.ftcfg['nncf_config']['compression']['params'] = paas_ft_cfg

    # def set_optimizer_cfg(self, optim_cfg: dict):
    #     if not isinstance(optim_cfg, dict):
    #         raise TypeError('set_optimizer_cfg only accepts dictionary as input')
    #     self.ftcfg['optimizer'] = optim_cfg

    def serialize(self, path):
        if self.has_none_value():
            raise ValueError('Null values found in ft cfg, pls check.\n{}'.format(self))
        dict2yamlnfile(self.ftcfg, path)

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
            "_base_": "model.py",
            "find_unused_parameters": True,
            "lr_config": {
                "policy": "step",
                "step": [
                    40,
                    60
                ],
                "warmup": "linear",
                "warmup_iters": 500,
                "warmup_ratio": 0.1
            },
            "nncf_config": {
                "compression":
                {
                        "algorithm": "filter_pruning",
                        "ignored_scopes": [
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[cls_convs]/Sequential[0]/NNCFConv2d[0]",
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[cls_convs]/Sequential[0]/NNCFConv2d[0]",
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[reg_convs]/Sequential[0]/NNCFConv2d[0]",
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[reg_convs]/Sequential[0]/NNCFConv2d[0]",
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[reg_convs]/Sequential[1]/NNCFConv2d[0]",
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[reg_convs]/Sequential[1]/NNCFConv2d[0]",
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[cls_convs]/Sequential[1]/NNCFConv2d[0]",
                            "SingleStageDetector/SSDHead[bbox_head]/ModuleList[cls_convs]/Sequential[1]/NNCFConv2d[0]"
                        ],
                        "params": None,
                        "pruning_init": 0.00
                },
                "input_info": {
                    "sample_size": [1, 3, 256, 256]
                },
                "log_dir": "."
            },
            "optimizer": {
                "lr": 0.05
            },
            "total_epochs": 80
        }
        return base_ft_cfg
