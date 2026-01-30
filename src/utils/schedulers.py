# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math


class WSDSchedule(object):
    """ Learning rate scheduler adopted by V-JEPA 2
        warmup-constant-decay (WSD) learning rate schedule
    """
    def __init__(self, optimizer, warmup_steps, anneal_steps, T_max, start_lr, ref_lr, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps - anneal_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        # Warmup phase
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        # Constant phase
        elif self._step < self.T_max + self.warmup_steps:
            new_lr = self.ref_lr
        # Annealing/Decay phase
        else:
            _step = self._step - (self.T_max + self.warmup_steps)
            progress = float(_step) / float(max(1, self.anneal_steps))
            new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if "lr_scale" in group:
                group["lr"] *= group["lr_scale"]

        return new_lr


class WarmupCosineSchedule(object):
    """Cosine annealing learning rate scheduler with a linear warmup phase"""
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, last_epoch=-1, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr  # Starting learning rate for warmup
        self.ref_lr = ref_lr      # Target reference learning rate after warmup
        self.final_lr = final_lr  # Minimum learning rate limit
        self.warmup_steps = warmup_steps   # Number of warmup steps
        self.T_max = T_max - warmup_steps  # Total cosine annealing steps (Total steps minus warmup)
        self._step = 0.0    # Current step counter

    def step(self):
        """Updates the learning rate; called at every training step"""
        self._step += 1
        # Warmup phase
        if self._step < self.warmup_steps:
            # Calculate warmup progress (between 0 and 1)
            progress = float(self._step) / float(max(1, self.warmup_steps))
            # Linearly increase learning rate from start_lr to ref_lr
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            # Calculate learning rate using the cosine annealing formula
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        # Update learning rate for all parameter groups in the optimizer
        # param_groups: a list of dictionaries where each dict represents a parameter group.
        # Commonly used for uniform updates or layer-wise learning rate adjustments.
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr




class CosineWDSchedule(object):
    """ Implements a Cosine Annealing scheduler for Weight Decay (WD).

        Weight decay gradually changes from initial ref_wd to final_wd
        following a cosine curve over a period of T_max steps.

        This helps balance regularization and model performance by using higher
        weight decay (ref_wd) early in training and lowering it (final_wd) later.
    """
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd      # Initial weight decay value
        self.final_wd = final_wd  # Final weight decay value
        self.T_max = T_max        # Total period length (total steps)
        self._step = 0.0          # Current step counter (initialized at 0)

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        # Calculate weight decay value after cosine annealing
        # Formula: final_wd + (ref_wd - final_wd) * 0.5 * (1 + cos(π * progress))
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            # Skip weight decay for specific groups using the WD_exclude flag
            # (e.g., BatchNorm layers usually do not require weight decay)
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd