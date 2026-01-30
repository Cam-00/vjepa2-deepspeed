# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import getLogger
from multiprocessing import Value

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):
    """
    Functionality: Perform batching and masking operations on input batch data.

    Returns: A list where each element is a tuple: (collated_batch, collated_masks_enc, collated_masks_pred)
            The list length depends on the length of dataset_fpcs, which may contain multiple frame counts.
        collated_batch: a tuple: (batch_buffer, batch_label, batch_clip_indices)
            # batch_buffer: Batch sampled frame data, a list of length num_clips, elements are tensor[bs, fpc, H, W, C]
            # batch_label: Batch video labels, a tensor[bs], elements are int numbers or values
            # batch_clip_indices: Batch frame sampling indices, a list of length num_clips, elements are tensor[bs, fpc]
        collated_masks_enc: Patch indices visible to the encoder for the batch, a list: [tensor(B, K1), tensor(B, K2)]
        collated_masks_pred: Patch indices visible to the predictor for the batch, a list: [tensor(B, N - K1), tensor(B, N - K2)]
    """

    def __init__(
        self,
        cfgs_mask,               # Mask configuration list, containing multiple mask generation configs. V-JEPA uses 2 strategies: short-range masks (8 blocks) and long-range masks (2 blocks).
        dataset_fpcs,            # List of frames per clip in the dataset (different clips may have different frame count settings).
        crop_size=(224, 224),    # Video crop size.
        patch_size=(16, 16),     # Pixel size of a single patch.
        tubelet_size=2,          # Tubelet size (frames) in the temporal dimension.
    ):
        super(MaskCollator, self).__init__()

        # Initialize mask generator dictionary: keys are frame counts, values are lists of mask generators.
        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            # Generate mask generators for each frame count setting
            self.mask_generators[fpc] = []
            # Create independent mask generators for each mask configuration
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),     # Spatial mask scale
                    temporal_pred_mask_scale=m.get("temporal_scale"),   # Temporal mask scale
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),       # Number of predicted regions (8 or 2); one region may contain multiple patches.
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),    # Maximum number of visible patches in the encoder.
                    full_complement=m.get("full_complement", False),   # Whether to use the full complement set.
                    pred_full_complement=m.get("pred_full_complement", False),   # Whether to use full complement during prediction.
                    inv_block=m.get("inv_block", False),
                )
                self.mask_generators[fpc].append(mask_generator)

    def step(self):
        """
        Update the state of all mask generators, typically called at the end of an epoch.
        """
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()  # Update the seed of the mask generator.

    def __call__(self, batch):
        # Input batch: Batch of sampled sample data, a list where elements are tuples (buffer, label, clip_indices).
        # buffer: Sampled frame data, a list of length num_clips, elements are numpy.ndarray(fpc, H, W, C).
        # label: Single video label, an int number or one value.
        # clip_indices: Frame sampling indices, a list of length num_clips, elements are 1D ndarray of length fpc.

        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            # sample: content returned by dataset sampling: tuple (buffer, label, clip_indices).
            # Each sample in the batch list is such a tuple.
            fpc = len(sample[-1][-1])
            filtered_batches[fpc] += [sample]

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]  # a list of tuples: (buffer, label, clip_indices), total bs elements.
            batch_size = len(fpc_batch)  # batch size, bs.
            if batch_size == 0:
                continue

            # Batch convert data to Tensor type.
            # clip_indices: [element1, element2, ..., element_num_clips], where each element is a tensor (bs, fpc).
            collated_batch = torch.utils.data.default_collate(fpc_batch)  # a list containing 3 lists, each stacked from bs samples.

            collated_masks_pred, collated_masks_enc = [], []
            for i, mask_generator in enumerate(self.mask_generators[fpc]):
                masks_enc, masks_pred = mask_generator(batch_size)  # tensor(B, K), tensor(B, N-K).
                collated_masks_enc.append(masks_enc)    # a list: [tensor(B, K1), tensor(B, K2)]
                collated_masks_pred.append(masks_pred)  # a list: [tensor(B, N - K1), tensor(B, N - K2)]
            fpc_collations += [(collated_batch, collated_masks_enc, collated_masks_pred)]

        # Returns a list of tuples: (collated_batch, collated_masks_enc, collated_masks_pred).
        return fpc_collations  # list length = number of different frame count settings.


class _MaskGenerator(object):
    """
    Mask Generator:
        Generates visible patch (token) indices for the predictor and encoder for each sample (clip).

        1. Generates visible region indices for encoder and predictor per video clip.
        2. Supports spatio-temporal 3D mask generation (temporal dimension + spatial dimension).
        3. Implements various masking strategies (random block masking, complementary masking, etc.).

    Returns: Visible patch indices for encoder/predictor of a batch:
            Tuple (collated_masks_enc, collated_masks_pred),
            Shapes: tensor(B, K), tensor(B, N-K),
            B = batch_size,
            K = number of non-masked patch indices (visible patch count),
            N = T*H*W ---- total number of patches in blocks.
    """

    def __init__(
        self,
        crop_size=(224, 224),                 # Video frame crop size.
        num_frames=16,                        # Number of frames per clip.
        spatial_patch_size=(16, 16),          # Spatial patch size.
        temporal_patch_size=2,                # Temporal patch depth (number of frames).
        spatial_pred_mask_scale=(0.2, 0.8),   # Range of spatial mask scale.
        temporal_pred_mask_scale=(1.0, 1.0),  # Range of temporal mask scale.
        aspect_ratio=(0.3, 3.0),              # Range of mask region aspect ratio.
        npred=1,                              # Number of masked sampling regions per clip (2 or 8 in V-JEPA).
        max_context_frames_ratio=1.0,         # Max ratio of context frames; 1 means masking targets all frames.
        max_keep=None,                        # Maximum number of visible patches during masking.
        inv_block=False,                      # Whether to invert predictor/encoder roles.
        full_complement=False,                # Whether predictor is exactly the complement of encoder.
        pred_full_complement=False,           # Whether encoder is exactly the complement of predictor.
    ):
        super(_MaskGenerator, self).__init__()
        # Parameter standardization (ensure tuple format).
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2    # (crop_size, crop_size)
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        # Calculate patch count in spatial dimensions (height and width).
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        # Calculate patch count in temporal dimension.
        self.duration = num_frames // temporal_patch_size

        # Masking strategy parameters.
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred

        # Calculate max context duration (number of patches in temporal dimension).
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask

        # Max number of patches to keep for memory considerations.
        self.max_keep = max_keep

        # Create a multiprocessing-shared integer as a thread-safe counter for seeds, starting at -1.
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

        self.inv_block = inv_block

    def step(self):
        """
        Thread-safe counter increment.
        Functionality: Obtain the seed for the mask generator, suitable for multiprocessing.
        """
        i = self._itr_counter
        with i.get_lock():  # Automatically acquire/release lock.
            i.value += 1
            v = i.value
        return v            # Return incremented value as random seed.

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        """
        Sample 3D mask block dimensions (temporal, height, width).
        Functionality: Calculate the spatial shape (t, h, w) of the randomly masked blocks after patchification (3D convolution).
        """
        # -- Sample temporal block mask scale: sampling on temporal axis, number of spatial planes to mask.
        _rand = torch.rand(1, generator=generator).item()  # a pseudo-random number
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))  # If scale = 1, all temporal patches are masked.

        # -- Sample spatial block mask scale: sampling on spatial planes, number of patches to mask (1 patch = 16x16 pixels).
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)  # Number of patches (area) to keep on the spatial plane.

        # -- Sample block aspect-ratio.
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute masked-block height and width (given scale and aspect-ratio).
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))  # Calculate height based on area and aspect ratio.
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)  # Return mask block dimensions.

    def _sample_block_mask(self, b_size):
        """
        Functionality: Generate a 3D binary mask matrix based on input mask block dimensions (t, h, w).

        Input: b_size (t, h, w): masked-block shape, without specified position.

        Returns: A mask matrix of shape (T, H, W) in block space, specifying mask positions.
        """
        t, h, w = b_size
        # Randomly generate starting positions (temporal, height, width) --- true random numbers.
        top = torch.randint(0, self.height - h + 1, (1,))      # Height start
        left = torch.randint(0, self.width - w + 1, (1,))      # Width start
        start = torch.randint(0, self.duration - t + 1, (1,))  # Temporal start

        # Construct a block-shaped mask matrix containing only 0s and 1s.
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames (X=self.max_context_frames).
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0  # Mask all frames outside the context range.

        return mask    # (T, H, W) shape.

    def __call__(self, batch_size):
        """
        Generate mask indices for the batch.
        Create encoder and predictor masks when collating imgs into a batch:
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)

        Returns: Tuple (collated_masks_enc, collated_masks_pred), shapes tensor(B, K) and tensor(B, N-K).
        """
        # 1. Initialize random seed.
        seed = self.step()     # Obtain incremented value as seed.
        g = torch.Generator()  # Create pseudo-random number generator.
        # Set pseudo-random seed to ensure experimental reproducibility.
        g.manual_seed(seed)    # seed must be 32-bit integer.

        # 2. Pseudo-randomly sample 3D mask block dimensions.
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,  # (1.0, 1.0) for V-JEPA.
            spatial_scale=self.spatial_pred_mask_scale,    # (0.15, 0.15) or (0.7, 0.7) for V-JEPA.
            aspect_ratio_scale=self.aspect_ratio,          # (0.75, 1.5) for V-JEPA.
        )   # (t, h, w) -- dimensions of the area to be masked.

        # 3. Store visible patch (token) indices for predictor and encoder for the batch size.
        collated_masks_pred, collated_masks_enc = [], []
        # Initialize minimum patches to keep for encoder/predictor.
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width

        # 4. Generate mask for each sample in the batch.
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                # Initialize mask matrix with all 1s.
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                # Combine multiple sampling regions (via logical AND multiplication).
                for _ in range(self.npred):  # 2 or 8 per V-JEPA.
                    mask_e *= self._sample_block_mask(p_size)
                # Flatten the mask matrix for the clip.
                mask_e = mask_e.flatten()  # (T*H*W, ) 1D tensor with elements 0 or 1.

                ## Flatten mask matrix and get indices (Encoder and predictor are complements here).
                # Predictor visible patch indices.
                mask_p = torch.argwhere(mask_e == 0).squeeze()  # Indices where mask_e is 0.
                # Encoder visible patch indices.
                mask_e = torch.nonzero(mask_e).squeeze()        # Indices where mask_e is non-zero.

                # Validity check.
                empty_context = len(mask_e) == 0
                if not empty_context:
                    # Update min visible patches to unify length across the batch.
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    # Store flattened visible patch indices for each sample.
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        # 5. Check if encoder patches exceed the limit.
        # Enforcing limits on encoder visible patches ensures sufficient masking for effective model training.
        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        # 6. Unify length: ensure every sample in the batch has the same number of visible patches.
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]

        # 7. Complement logic: ensure encoder and predictor masks are mutually exclusive.
        if self.full_complement:  # predictor mask is just complement of encoder mask
            collated_masks_pred = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:  # encoder mask is just complement of predictor mask
            collated_masks_enc = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]
        # 8. Convert list to Tensor.
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)  # tensor (B, K)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)  # tensor (B, N-K)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc  # predict context from block
        else:
            return collated_masks_enc, collated_masks_pred   # tensor(B, K), tensor(B, N-K)