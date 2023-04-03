from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from typing import Union
import warnings


class DummyScheduler(_LRScheduler):
    """
    Simply returns the learning rate (i.e. no change is made to the given learning rate).
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(DummyScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [group['lr'] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: Union[list, int], last_epoch: int = -1):
    """
    A modified version of https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_constant_schedule_with_warmup,
    that handles mnultiple param_groups. Each param_group is able to have its own num_warmup_steps.
    """

    # def lr_lambda(current_step: int):
    #     if current_step < num_warmup_steps:
    #         return float(current_step) / float(max(1.0, num_warmup_steps))
    #     return 1.0

    class ConstantScheduleWithWarmup:
        def __init__(self, num_warmup_steps):
            self.num_warmup_steps = num_warmup_steps

        def __call__(self, current_step: int):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1.0, self.num_warmup_steps))
            return 1.0

    if isinstance(num_warmup_steps, dict):
        num_warmup_steps = num_warmup_steps.values()

    lr_lambda_list = []
    for i in num_warmup_steps:
        lr_lambda_list.append(ConstantScheduleWithWarmup(num_warmup_steps=i))

    return LambdaLR(optimizer, lr_lambda_list, last_epoch=last_epoch)