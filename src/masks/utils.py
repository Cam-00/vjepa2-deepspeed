# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:  # Generally, the passed masks list contains only one element
        # m.unsqueeze(-1) shape: tensor(B, K, 1)
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, K, D)
        all_x += [torch.gather(x, dim=1, index=mask_keep)]  # Shape of torch.gather() result: tensor(B, K, D)
    if not concat:
        return all_x  # a list [tensor(B, K, D)]

    return torch.cat(all_x, dim=0)  # a tensor(M*B, K, D), where M is the length of the masks list, typically M = 1