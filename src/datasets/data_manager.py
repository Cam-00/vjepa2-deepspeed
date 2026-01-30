# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


# 数据加载初始化(包含数据预处理), 返回 data_loader, dist_sampler
def init_data(
    batch_size,  # Number of samples per batch
    transform=None,  # Data augmentation for individual samples, including spatial resizing/cropping of clips (e.g., 224x224 or 256x256)
    shared_transform=None,  # Shared transformations applied across all samples (e.g., consistent processing for multi-frame video)
    data="ImageNet",  # Dataset type (default: ImageNet)
    collator=None,  # Mask collator settings; both V-JEPA and V-JEPA 2 use multi-block masking
    pin_mem=True,  # Whether to use pinned memory (accelerates data transfer to GPU)
    num_workers=8,  # Number of subprocesses for data loading
    world_size=1,  # Total number of processes for distributed training
    rank=0,  # Rank of the current process
    root_path=None,  # Root directory path of the dataset
    image_folder=None,  # Image subdirectory name (specific to classification datasets)
    training=True,  # Whether to load the ImageNet training set; otherwise loads the validation set
    enable_probe=False,  # Whether to enable model performance validation within the training loop
    drop_last=True,  # Whether to discard the last incomplete batch
    subset_file=None,  # Path to the file specifying a data subset
    clip_len=None,  # Number of frames per clip (specific to video datasets)
    dataset_fpcs=None,  # Custom frames per clip (fpcs) for each dataset
    frame_sample_rate=None,  # Sampling interval between video frames
    duration=None,  # Clip duration in seconds
    fps=None,  # Video frames per second (FPS)
    num_clips=1,  # Number of clips to sample from each video
    random_clip_sampling=True,  # Whether to sample clips randomly
    allow_clip_overlap=False,  # Whether to allow overlap between sampled clips
    filter_short_videos=False,  # Whether to filter out videos that are too short
    filter_long_videos=int(1e9),  # Threshold for filtering out excessively long videos
    datasets_weights=None,  # Sampling weights for each dataset in a multi-dataset setup
    persistent_workers=False,  # Whether to keep worker processes alive between epochs
    deterministic=True,  # Whether to enable deterministic mode
    log_dir=None,  # Directory for storing logs
):
    if data.lower() == "imagenet":
        from src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            subset_file=subset_file,
        )

    elif data.lower() == "videodataset":
        from src.datasets.video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
            enable_probe=enable_probe,
        )

    return (data_loader, dist_sampler)
