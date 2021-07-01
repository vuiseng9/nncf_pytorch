import warnings
import numpy as np
import random
import torch
from copy import copy
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner, LoggerHook,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer, load_checkpoint,
                         build_runner)

from mmdet.core import (DistEvalHook, DistEvalPlusBeforeRunHook, EvalHook,
                        EvalPlusBeforeRunHook)
from mmdet.integration.nncf import CompressionHook, CheckpointHookBeforeTraining, compression, wrap_nncf_model
from mmdet.parallel import MMDataCPU
from mmcv.utils import build_from_cfg

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
from mmdet.apis.fake_input import get_fake_input
from functools import partial
from mmcv import ProgressBar
from mmcv.runner.log_buffer import LogBuffer
from torch.utils.data import dataloader

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def add_logging_on_first_and_last_iter(runner):
    def every_n_inner_iters(self, runner, n):
        if runner.inner_iter == 0 or runner.inner_iter == runner.max_iters - 1:
            return True
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    for hook in runner.hooks:
        if isinstance(hook, LoggerHook):
            hook.every_n_inner_iters = every_n_inner_iters.__get__(hook)


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    map_location = 'cuda'
    if not torch.cuda.is_available():
        map_location = 'cpu'

    if cfg.load_from:
        load_checkpoint(model=model, filename=cfg.load_from, map_location=map_location)

    # put model on gpus
    if torch.cuda.is_available():
        model = model.cuda()

    # nncf model wrapper
    nncf_enable_compression = bool(cfg.get('nncf_config'))
    if nncf_enable_compression:
        cfg.nncf_config['log_dir']=cfg.work_dir
        compression_ctrl, model = wrap_nncf_model(model, cfg, data_loaders[0], get_fake_input)
    else:
        compression_ctrl = None

    if torch.cuda.is_available():
        if distributed:
            # put model on gpus
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    else:
        model = MMDataCPU(model)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    add_logging_on_first_and_last_iter(runner)

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        if nncf_enable_compression:
            eval_hook = DistEvalPlusBeforeRunHook if distributed else EvalPlusBeforeRunHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if nncf_enable_compression:
        runner.register_hook(CompressionHook(compression_ctrl=compression_ctrl))
        runner.register_hook(CheckpointHookBeforeTraining())
    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from, map_location=map_location)

    if cfg.get('restful', None) is not None:
        if cfg.restful is True:
            def single_gpu_eval_loss(model, data_loader, compression_ctrl):
                model.eval()
                loss_parser=compression_ctrl._model.get_nncf_wrapped_model()._parse_losses
                log_buffer=LogBuffer()

                dataset = data_loader.dataset
                prog_bar = ProgressBar(len(data_loader))
                for i, data in enumerate(data_loader):
                    with torch.no_grad():
                        losses = model(return_loss=True, **data)
                        loss, log_vars = loss_parser(losses)
                        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
                        log_buffer.update(outputs['log_vars'], outputs['num_samples'])

                        prog_bar.update()
                log_buffer.average()
                return log_buffer.output

            # return compression_ctrl, model, cfg, partial(eval_hook(val_dataloader, **eval_cfg).before_run, runner=runner)
            from mmdet.apis import single_gpu_test
            def single_gpu_test_fn(model, dataloader):
                results = single_gpu_test(model, dataloader, show=False)
                key_score = val_dataset.evaluate(results)
                return key_score
                
            test_fn = partial(single_gpu_test_fn, model=model, dataloader=val_dataloader)

            val_loss_fn = partial(single_gpu_eval_loss, model=model, data_loader=data_loaders[0], compression_ctrl=compression_ctrl)
            return compression_ctrl, model, cfg, val_loss_fn, test_fn

    runner.run(data_loaders, cfg.workflow, compression_ctrl=compression_ctrl)