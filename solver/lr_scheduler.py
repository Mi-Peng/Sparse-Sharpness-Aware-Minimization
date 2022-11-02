import math
from typing import List

import torch
import torch.optim as optim

from utils.configurable import configurable
from solver.build import LR_SCHEDULER_REGISTRY 



class LRscheduler:
    def __init__(self, optimizer: optim.Optimizer, resume: bool = False):
        self.optimizer = optimizer
        
        if not resume:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch):
        new_lrs = self.get_lr(epoch)
        for group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            group["lr"] = new_lr

@LR_SCHEDULER_REGISTRY.register()
class CosineLRscheduler(LRscheduler):
    @configurable
    def __init__(self, 
        optimizer: optim.Optimizer,
        resume: bool = False,
        *,
        T_max: int,
        eta_min: float,
        warmup_epoch: int,
        warmup_init_lr: float,
    ):
        super().__init__(optimizer, resume)
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epoch = warmup_epoch
        self.warmup_init_lr = warmup_init_lr

    @classmethod
    def from_config(cls, args):
        return {
            "resume": args.resume,
            "T_max": args.epochs,
            "eta_min": args.eta_min,
            "warmup_epoch": args.warmup_epoch,
            "warmup_init_lr": args.warmup_init_lr,
        }

    def get_lr(self, epoch):
        if epoch < self.warmup_epoch:
            lrs = [self.warmup_init_lr + float(epoch)  * (base_lr - self.warmup_init_lr) / self.warmup_epoch for base_lr in self.base_lrs]
        else:
            lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * (epoch - self.warmup_epoch) / (self.T_max - self.warmup_epoch))) / 2 
                   for base_lr in self.base_lrs]
        return lrs

@LR_SCHEDULER_REGISTRY.register()
class MultiStepLRscheduler(LRscheduler):
    @configurable
    def __init__(self, 
        optimizer: optim.Optimizer, 
        resume: bool = False,
        *,
        milestone: List,
        gamma: float,
        warmup_epoch: int,
        warmup_init_lr: float,
    ):
        super().__init__(optimizer, resume)
        self.milestone = milestone
        self.gamma = gamma
        self.warmup_epoch = warmup_epoch
        self.warmup_init_lr = warmup_init_lr

    @classmethod
    def from_config(cls, args):
        return {
            "milestone": args.milestone,
            "gamma": args.gamma,
            "warmup_epoch": args.warmup_epoch,
            "warmup_init_lr": args.warmup_init_lr,
        }

    def get_lr(self, epoch):
        if epoch < self.warmup_epoch:
            lrs = [self.warmup_init_lr + float(epoch)  * (base_lr - self.warmup_init_lr) / self.warmup_epoch for base_lr in self.base_lrs]
        else:
            lrs = []
            for base_lr in self.base_lrs:
                ratio = 1.0
                for mile in self.milestone:
                    if epoch > mile: ratio *= self.gamma
                lrs.append(base_lr * ratio)
        return lrs