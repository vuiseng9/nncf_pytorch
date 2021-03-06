import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from functools import partial
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.config.structures import ModelEvaluationArgs
from nncf.common.utils.progress_bar import ProgressBar
from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.torch.structures import AutoQPrecisionInitArgs
from nncf.torch.structures import QuantizationPrecisionInitArgs
from nncf.torch.utils import is_tensor
from nncf.torch.utils import objwalk
from contextlib import contextmanager


class PTInitializingDataLoader(NNCFDataLoader):
    """
    This class wraps the torch.utils.data.DataLoader class,
    and defines methods to parse the general data loader output to
    separate the input to the compressed model and the ground truth target
    for the neural network. This is required for proper initialization of
    certain compression algorithms.
    """

    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    @property
    def batch_size(self):
        return self._data_loader.batch_size

    def __iter__(self):
        return iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader)

    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        """Returns (args, kwargs) for the current model call to be made during the initialization process"""
        raise NotImplementedError

    def get_target(self, dataloader_output: Any) -> Any:
        """
        Parses the generic data loader output and returns a structure to be used as
        ground truth in the loss criterion.

        :param dataloader_output - the (args, kwargs) tuple returned by the __next__ method.
        """

        raise NotImplementedError


class DefaultInitializingDataLoader(PTInitializingDataLoader):

    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        return (dataloader_output[0],), {}

    def get_target(self, dataloader_output: Any):
        return dataloader_output[1]


def wrap_dataloader_for_init(data_loader) -> PTInitializingDataLoader:
    if not isinstance(data_loader, PTInitializingDataLoader):
        loaded_item = next(iter(data_loader))
        if isinstance(loaded_item, (tuple, list)) and len(loaded_item) == 2:
            return DefaultInitializingDataLoader(data_loader)
        raise NotImplementedError("By default it is assumed that the data loader used for initialize "
                                  "produces a tuple/list of (*model_input*, *ground_truth*) and that no special "
                                  "forward arguments have to be set during init. If this is not the case, then instead "
                                  "of your regular data loader you need to pass a specialized version of "
                                  "PTInitializingDataLoader that returns a general (args, kwargs) tuple for your "
                                  "model to be called with at each __next__ call.")
    return data_loader


class PartialDataLoader:
    def __init__(self, regular_data_loader: DataLoader, iter_ratio=1.0):
        if iter_ratio < 0.0 or iter_ratio > 1.0:
            raise ValueError("iter_ratio must be within 0 to 1 range")
        self.data_loader = regular_data_loader
        self.batch_size = regular_data_loader.batch_size
        self._stop_id = math.ceil(len(self.data_loader) * iter_ratio)
        self._batch_id = 0

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self._batch_id = 0
        return self

    def __next__(self) -> Any:
        if self._batch_id < self._stop_id:
            loaded_item = next(self.data_loader_iter)
            self._batch_id += 1
            return loaded_item
        raise StopIteration

    def __len__(self) -> int:
        return self._stop_id


class DataLoaderBaseRunner:
    def __init__(self, model, init_device: Optional[str]):
        self.model = model
        self.init_device = init_device
        self.progressbar_description = 'Algorithm initialization'

    def _run_model_inference(self, data_loader, num_init_steps, device):
        for i, loaded_item in ProgressBar(
                enumerate(data_loader),
                total=num_init_steps,
                desc=self.progressbar_description,
        ):
            if num_init_steps is not None and i >= num_init_steps:
                break
            args_kwargs_tuple = data_loader.get_inputs(loaded_item)
            self._infer_batch(args_kwargs_tuple, device)

    def _infer_batch(self, args_kwargs_tuple, device):
        to_device_fn = partial(torch.Tensor.to, device=device)
        args, kwargs = objwalk(args_kwargs_tuple, is_tensor, to_device_fn)
        self.model(*args, **kwargs)

    def run(self, data_loader, num_init_steps):
        if self.init_device is not None:
            original_device = next(iter(self.model.parameters())).device
            self.model.to(self.init_device)

        self._prepare_initialization()
        device = next(self.model.parameters()).device
        data_loader = wrap_dataloader_for_init(data_loader)

        with torch.no_grad():
            self._run_model_inference(data_loader, num_init_steps, device)
            self._apply_initializers()

        if self.init_device is not None:
            self.model.to(original_device)

    def _prepare_initialization(self):
        raise NotImplementedError

    def _apply_initializers(self):
        raise NotImplementedError


class SimpleDataLoaderRunner(DataLoaderBaseRunner):
    def _prepare_initialization(self):
        pass

    def _apply_initializers(self):
        pass


class DataLoaderBNAdaptationRunner(DataLoaderBaseRunner):
    def __init__(self, model, init_device: str, num_bn_forget_steps):
        super().__init__(model, init_device)
        self.progressbar_description = 'BatchNorm statistics adaptation'
        self.num_bn_forget_steps = num_bn_forget_steps
        self.momentum_bn_forget = 0.9
        self.original_momenta_values = {}
        self.original_training_state = {}

    @staticmethod
    def _apply_to_batchnorms(func):
        def func_apply_to_bns(module):
            if isinstance(module, (torch.nn.modules.batchnorm.BatchNorm1d,
                                   torch.nn.modules.batchnorm.BatchNorm2d,
                                   torch.nn.modules.batchnorm.BatchNorm3d)):
                func(module)

        return func_apply_to_bns

    @contextmanager
    def _bn_training_state_switcher(self) -> None:
        def save_original_bn_training_state(module: torch.nn.Module):
            self.original_training_state[module] = module.training

        def set_bn_training_state(module: torch.nn.Module, state: Dict[str, bool]):
            module.training = state

        def restore_original_bn_training_state(module: torch.nn.Module):
            module.training = self.original_training_state[module]

        self.model.apply(self._apply_to_batchnorms(save_original_bn_training_state))
        self.model.apply(self._apply_to_batchnorms(partial(set_bn_training_state, state=True)))
        try:
            yield
        finally:
            self.model.apply(self._apply_to_batchnorms(restore_original_bn_training_state))

    @contextmanager
    def _bn_momentum_switcher(self) -> None:
        def set_bn_momentum(module, momentum_value):
            module.momentum = momentum_value

        def save_original_bn_momentum(module: torch.nn.Module):
            self.original_momenta_values[module] = module.momentum

        def restore_original_bn_momentum(module: torch.nn.Module):
            module.momentum = self.original_momenta_values[module]

        self.model.apply(self._apply_to_batchnorms(save_original_bn_momentum))
        self.model.apply(self._apply_to_batchnorms(partial(set_bn_momentum,
                                                           momentum_value=self.momentum_bn_forget)))
        try:
            yield
        finally:
            self.model.apply(self._apply_to_batchnorms(restore_original_bn_momentum))

    def _run_model_inference(self, data_loader, num_init_steps, device):
        num_bn_forget_steps = self.num_bn_forget_steps

        with self._bn_training_state_switcher():
            if num_bn_forget_steps is not None and num_bn_forget_steps > 0:
                with self._bn_momentum_switcher():
                    for i, loaded_item in enumerate(data_loader):
                        if i >= num_bn_forget_steps:
                            break
                        args_kwargs_tuple = data_loader.get_inputs(loaded_item)
                        self._infer_batch(args_kwargs_tuple, device)

            for i, loaded_item in ProgressBar(
                    enumerate(data_loader),
                    total=num_init_steps,
                    desc=self.progressbar_description
            ):
                if num_init_steps is not None and i >= num_init_steps:
                    break
                args_kwargs_tuple = data_loader.get_inputs(loaded_item)
                self._infer_batch(args_kwargs_tuple, device)

    def _prepare_initialization(self):
        pass

    def _apply_initializers(self):
        pass


def default_criterion_fn(outputs: Any, target: Any, criterion: Any) -> torch.Tensor:
    return criterion(outputs, target)


def register_default_init_args(nncf_config: 'NNCFConfig',
                               train_loader: torch.utils.data.DataLoader,
                               criterion: _Loss = None,
                               criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor] = None,
                               autoq_eval_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], float] = None,
                               autoq_eval_loader: torch.utils.data.DataLoader = None,
                               model_eval_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], float] = None,
                               device: str = None,
                               ) -> 'NNCFConfig':
    nncf_config.register_extra_structs([QuantizationRangeInitArgs(data_loader=wrap_dataloader_for_init(train_loader),
                                                                  device=device),
                                        BNAdaptationInitArgs(data_loader=wrap_dataloader_for_init(train_loader),
                                                             device=device)])

    if criterion is not None:
        if not criterion_fn:
            criterion_fn = default_criterion_fn
        nncf_config.register_extra_structs([QuantizationPrecisionInitArgs(criterion_fn=criterion_fn,
                                                                          criterion=criterion,
                                                                          data_loader=train_loader,
                                                                          device=device)])

    if autoq_eval_fn is not None:
        if not autoq_eval_loader:
            autoq_eval_loader = train_loader
        nncf_config.register_extra_structs([AutoQPrecisionInitArgs(data_loader=autoq_eval_loader,
                                                                   eval_fn=autoq_eval_fn,
                                                                   nncf_config=nncf_config)])

    if model_eval_fn is not None:
        nncf_config.register_extra_structs([ModelEvaluationArgs(eval_fn=model_eval_fn)])

    return nncf_config
