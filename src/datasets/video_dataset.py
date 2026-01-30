# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pathlib
import warnings
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torchvision
# decord library: video reading, frame extraction, data loading
# VideoReader is a class provided by Decord to access frames directly from video files
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(  #### Construct data required for the model by batch sampling from multiple datasets
        data_paths,  # List of video data paths
        batch_size,  # Number of samples per batch
        frames_per_clip=8,  # Number of frames per video clip, default is 8
        dataset_fpcs=None,  # Different datasets may use different frame count settings
        frame_step=4,  # Frame sampling stride, default is 4
        duration=None,  # Video duration (seconds), the length of each video in the dataset for model construction
        fps=None,  # Video frame rate (FPS)
        num_clips=1,  # Number of clips sampled from a single video, default is 1
        random_clip_sampling=True,  # Whether to randomly sample video clips, default is True
        allow_clip_overlap=False,  # Whether to allow overlapping clips, default is False
        filter_short_videos=False,  # Whether to filter out videos that are too short, default is False
        filter_long_videos=int(10 ** 9),  # Threshold to filter out excessively long videos, default is 10^9
        transform=None,  # Data augmentation transformations applied to each video clip
        shared_transform=None,  # Optional, shared transformations (e.g., multi-crop)
        rank=0,  # Rank of the current process, default is 0
        world_size=1,  # Total number of processes, default is 1
        datasets_weights=None,  # Sampling weights for different datasets when using multiple datasets
        collator=None,  # collator = mask_collator, data masking settings
        drop_last=True,
        # If the dataset size is not divisible by batch_size, setting to True drops the last incomplete batch
        num_workers=10,  # Number of worker processes for data loading, default is 10
        pin_mem=True,  # If True, enables memory pinning to accelerate data transfer from host to GPU
        persistent_workers=True,  # Whether to keep worker processes active, default is True
        deterministic=True,  # Whether to use deterministic data loading, default is True
        log_dir=None,  # Optional, path to the log directory
        enable_probe=False,  # Whether to enable model performance validation during the training loop
):
    """ Sample video data required for the model from the raw video dataset
        dataset: a tuple (buffer, label, clip_indices)
        buffer: sampled frame data, a list of length num_clips, elements are numpy.ndarray(C, fpc, H, W) ---> (num_clips, C, fpc, H, W)
        label: single video label, an int or one value
        clip_indices: frame sampling positions, a list of length num_clips, elements are 1D <class 'numpy.ndarray'> of length clip_len, shape (num_clips, clip_len)
    """
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
    )
    # Create log directory object
    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        # Monitor resources during the dataset creation process
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("VideoDataset dataset created")

    # When multiple datasets exist, merge into a new dataset using data weights
    if not enable_probe:  # Stop model validation during training
        if datasets_weights is not None:
            # Returns an iterator pointing to the list of sample indices allocated to a specific GPU rank
            # (equal number of samples per GPU, no duplicates)
            # length of 1D indices list = total dataset length // number of GPUs
            dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                                      drop_last=True)
        else:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )
    else:  # Enable model validation during training
        if datasets_weights is not None:
            dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                                      drop_last=True)
        else:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )

    # Deterministic data loading
    if deterministic:
        # DataLoader: if sampler is used, load samples from dataset based on the 1D indices provided by sampler
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,  # Collator: operates on and outputs a batch of data after sampling
            sampler=dist_sampler,  # Defines strategy for drawing samples; cannot use shuffle if specified
            batch_size=batch_size,  # Number of samples in a batch, e.g., 12 raw video samples
            drop_last=drop_last,  # Drop last incomplete batch if dataset size not divisible by batch_size
            pin_memory=pin_mem,  # Accelerates host-to-GPU transfer
            num_workers=num_workers,  # Uses multi-threading for data loading
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    # Non-deterministic data loading
    # Returns out-of-order results
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """Video classification dataset.

        Features:
            1. Locates a specific video sample from the total dataset, uses decord's VideoReader
               to extract frame samples, applies data augmentation, and returns the sampled frame data.
            2. Filters excessively long or short videos.
            3. Processes individual video samples from the dataset.

        Returns: A tuple (buffer, label, clip_indices)
            buffer: sampled frame data, a list of length num_clips, elements are numpy.ndarray(C, fpc, H, W) ---> (num_clips, C, fpc, H, W)
            label: single video label, an int or one value
            clip_indices: frame sampling positions, a list of length num_clips, elements are 1D <class 'numpy.ndarray'> of length clip_len, shape (num_clips, clip_len)

    """

    def __init__(
            self,
            data_paths,
            datasets_weights=None,  # Dataset weight allocation
            frames_per_clip=16,
            fps=None,
            dataset_fpcs=None,
            frame_step=4,
            # V-JEPA 2 paper uses 4s clips (64 frames total), frame step of 4, final sampled clip is 16 frames
            num_clips=1,
            transform=None,  # Transformations including resizing spatial clips to specific size (e.g., 256*256)
            shared_transform=None,
            random_clip_sampling=True,
            allow_clip_overlap=False,
            filter_short_videos=False,
            filter_long_videos=int(10 ** 9),
            duration=None,  # duration in seconds, duration of each video in the dataset
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips  # Number of clips sampled from a single video
        self.transform = transform  # Data augmentation, including spatial resizing and changing buffer shape to (B, C, T, H, W)
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps

        # Exactly one of fps, duration, or frame_step must be defined
        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

        # Ensure data_paths is a list
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        # If per-dataset frame counts are not defined, unify all datasets to a fixed frame count
        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            # List of frame counts must match the number of data paths
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified for NFS data paths")
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        # samples: stores all videos from all datasets
        # labels: stores labels for all videos, each dataset has labels
        samples, labels = [], []
        self.num_samples_per_dataset = []  # Elements are the number of samples in each dataset

        # Sequentially read data and labels for each dataset
        for data_path in self.data_paths:

            if data_path[-4:] == ".csv":  # e.g., ssv2_train_paths.csv
                try:
                    # pandas.read_csv: returns a DataFrame (2D table structure)
                    # Reads headerless numerical data separated by space
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    # In image captioning datasets where we have space, we use :: as delimiter.
                    data = pd.read_csv(data_path, header=None, delimiter="::")

                # Load all data for one dataset
                samples += list(data.values[:, 0])  # list: [str1, str2, ...]
                # Load labels for one dataset
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == ".npy":
                data = np.load(data_path, allow_pickle=True)  # Load .npy file data

                # map(...): apply lambda to each element, returning an iterator
                # repr(x)[1:-1]: convert x to string and strip surrounding quotes
                data = list(map(lambda x: repr(x)[1:-1], data))  # list: [str1, str2, ...]

                samples += data
                # Fill labels with 0 for this dataset
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

        # Calculate index intervals for each dataset using cumulative sum
        # e.g., [2, 4, 6] -> [2, 6, 12]
        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        # [Optional] Weights for each sample to be used by downstream weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples  # [str, str, ...]
        self.labels = labels  # [label, label, ...]

        # Combined with itertools.islice for fast skipping of trained samples
        # Use 'b' for boolean type, initialized to False
        import multiprocessing
        self._fast_skip_mode_shared = multiprocessing.Value('b', False)

    def set_fast_skip(self, enable=False):
        # Modify shared variable value
        with self._fast_skip_mode_shared.get_lock():
            self._fast_skip_mode_shared.value = enable
        print(f"[Process {os.getpid()}] Set shared fast_skip to: {enable}")

    def __getitem__(self, index):
        """ Feature:
                Retrieves a specific video from the total dataset, splits it into equal-length segments,
                samples frames to form clips, and returns the concatenated sampled frame data.
            Returns: a tuple: (buffer, label, clip_indices)
                    buffer: sampled frame data, a list of shape (num_clips, C, T, H, W)
                    label: label of the specified video, an int
                    clip_indices: frame sampling positions, list of shape (num_clips, clip_len)
        """
        # If in fast skip mode, return dummy data to avoid triggering video/image loading
        if self._fast_skip_mode_shared.value:
            nc = self.num_clips
            fpc = self.dataset_fpcs[0]
            dummy_buffer = [np.zeros((3, fpc, 256, 256), dtype=np.float32) for _ in range(nc)]
            dummy_label = 0
            dummy_indices = [np.zeros(fpc, dtype=np.int64) for _ in range(nc)]
            return dummy_buffer, dummy_label, dummy_indices

        sample = self.samples[index]  # Get data specified by index (string)
        loaded_sample = False
        # Keep trying to load videos until a valid sample is found
        while not loaded_sample:
            if not isinstance(sample, str):
                logger.warning("Invalid sample.")
            else:
                # Load specified image data
                if sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                    loaded_sample = self.get_item_image(index)
                # Load specified video data
                else:
                    # Split long video into segments and sample clips per segment
                    # Returns: tuple (buffer, label, clip_indices)
                    loaded_sample = self.get_item_video(index)

            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        return loaded_sample

    def get_item_video(self, index):
        # Get specified video path
        sample = self.samples[index]
        # Get the dataset index to which the video belongs
        dataset_idx, _ = self.per_dataset_indices[index]
        # Get frame count used by corresponding dataset
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        # buffer: numpy.ndarray, shape (T, H, W, C)
        # clip_indices: list of length num_clips, elements are 1D ndarray of length clip_len
        buffer, clip_indices = self.loadvideo_decord(sample, frames_per_clip)
        loaded_video = len(buffer) > 0
        if not loaded_video:
            return

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc: (i + 1) * fpc] for i in range(nc)]

        # Apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)

        # Split sampled video into clips list
        buffer = split_into_clips(buffer)  # (nc, fpc, H, W, C)

        # Apply transforms (including resizing and adjusting shape to (nc, C, T, H, W))
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def get_item_image(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            # Read image and convert to tensor
            image_tensor = torchvision.io.read_image(path=sample, mode=torchvision.io.ImageReadMode.RGB)
        except Exception:
            return
        label = self.labels[index]
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expanding input image [3, H, W] ==> [T, 3, H, W]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))  # [T, 3, H, W] ==> [T H W 3]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)

        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample, fpc):
        """Load video content using Decord

            Input:
                sample: Individual video file path
                fpc: frames_per_clip

            Returns:
                buffer: Sampled video frames, numpy.ndarray of shape (T, H, W, C)
                clip_indices: Frame sampling positions, list of shape (num_clips, clip_len)
        """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None

        _fsize = os.path.getsize(fname)  # Get file size in bytes
        # Filter long videos
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None

        try:
            # Use CPU for decoding, frames returned as ndarray (T, H, W, C)
            # num_threads=-1: use all available CPU threads
            vr = VideoReader(fname, num_threads=-1)
        except Exception:
            return [], None

        # Calculate frame sampling stride
        fstp = self.frame_step  # Sampling stride, e.g., every 4th frame
        if self.duration is not None or self.fps is not None:
            try:
                # Get average FPS, rounded up
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)

            # Calculate video sampling stride
            if self.duration is not None:  # Specified video duration (seconds)
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                # Specified FPS
                assert self.duration is None
                # stride = avg_fps // specified_fps
                fstp = video_fps // self.fps

        assert fstp is not None and fstp > 0
        # Calculate clip length in frames
        clip_len = int(fpc * fstp)  # e.g., 64 frames = 16 * 4

        # Filter short videos
        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None

        # Seek to start of video before sampling
        vr.seek(0)

        # Partition video into equal sized segments and sample each clip
        partition_len = len(vr) // self.num_clips
        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, sample a random window within the segment
                end_indx = clip_len
                # Random clip sampling within segment
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                # Offset indices by partition index
                indices = indices + i * partition_len
            else:
                # Handle case where partition_len < clip_len and overlap not allowed
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len

                # Handle case where overlap is allowed and partition_len < clip_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - sample_len // fstp) * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        # Bulk retrieve frames by indices
        buffer = vr.get_batch(all_indices).asnumpy()  # returns shape (N, H, W, C)
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)