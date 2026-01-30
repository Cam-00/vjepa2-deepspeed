# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch


class MultiSeqWrapper(nn.Module):
    """ Wrapper for the model to handle multiple inputs with different shapes (different fcp/frame counts).
        Returns: a list, [[tensor1, tensor2], [tensor1, tensor2], ...]
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # model, like encoder / decoder / transformer

    def forward(self, x, masks=None):
        """
        :param x: [list] List of Tensors of different seq lengths
        :param masks: [list[list]] List of Tensors (outer index: masks for given seq length, inner index: multi-masks for that seq len)
        """
        if masks is None:
            return [self.backbone(xi) for xi in x]

        outs = [[] for _ in x]  # [[], [], ...] outer list length is len(dataset_fcps)

        # Iterate through input tensors and masks for each fcp (frames per clip) setting
        # xi : tensor(bs, fpc_i, H, W, C), a batch of video samples
        # mi : list[tensor(B, K1), tensor(B, K2)], where K1 and K2 are counts of visible patches
        for i, (xi, mi) in enumerate(zip(x, masks)):  # (xi, mi) belong to the same fcp setting
            #  mij : tensor(B, K1) or tensor(B, K2),
            #  Corresponding to the two mask sampling strategies in the V-JEPA paper:
            #  short-range masks: take the union of 8 randomly sampled target
            #  blocks covering 15% of each frame.
            #  long-range masks:  take the union of 2 randomly sampled target
            #  blocks covering 70% of each frame.
            for mij in mi:  # Handle two types of mask sampling
                outs[i] += [self.backbone(xi, masks=mij)]  # outs[i] : a list, [tensor1, tensor2]
        return outs  # a list, [[tensor1, tensor2], [tensor1, tensor2], ...]


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, has_cls=False):
        """
        :param x: [list[list]] List of Tensors from encoder output
                (outer index: selected fcp setting in data_fcps list, inner index: encoder output for specific multi-masks)
        :param masks_x: [list[list]] List of Tensors of encoder-masks for different fcp samples
        :param masks_y: [list[list]] List of Tensors of predictor-masks for different fcp samples
        """
        n = 0
        outs = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mxij, myij in zip(xi, mxi, myi):
                outs[i] += [self.backbone(xij, mxij, myij, mask_index=i, has_cls=has_cls)]
                n += 1
        return outs


class MultiModelWrapper(nn.Module):

    def __init__(self, encoder, predictor, target_encoder=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        if target_encoder is not None:
            self.target_encoder = target_encoder
            # Freeze target encoder parameters
            for p in self.target_encoder.parameters():
                p.requires_grad = False

    def forward(self, x, masks_enc, masks_pred, mode='train'):
        if mode == 'train':
            # Forward pass for encoder and predictor
            x = self.encoder(x, masks_enc)
            x = self.predictor(x, masks_enc, masks_pred)
            return x
        elif mode == 'target':
            # Forward pass for the target encoder (EMA model)
            with torch.no_grad():
                h = self.target_encoder(x)
                return h