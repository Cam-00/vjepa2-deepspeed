# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.datasets.utils.video.randerase import RandomErasing


def make_transforms(
    training=True,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    num_views_per_clip=1,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
):

    if not training and num_views_per_clip > 1:
        print("Making EvalVideoTransform, multi-view")
        _frames_augmentation = EvalVideoTransform(
            num_views_per_clip=num_views_per_clip,
            short_side_size=crop_size,
            normalize=normalize,
        )

    else:
        _frames_augmentation = VideoTransform(
            training=training,
            random_horizontal_flip=random_horizontal_flip,
            random_resize_aspect_ratio=random_resize_aspect_ratio,
            random_resize_scale=random_resize_scale,
            reprob=reprob,
            auto_augment=auto_augment,
            motion_shift=motion_shift,
            crop_size=crop_size,
            normalize=normalize,
        )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        training=True,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3 / 4, 4 / 3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ):

        self.training = training

        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(crop_size, crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
            ]
        )

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.normalize = torch.tensor(normalize)

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )

        self.spatial_transform = (
            video_transforms.random_resized_crop_with_shift if motion_shift else video_transforms.random_resized_crop
        )

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode="pixel",
            max_count=1,
            num_splits=1,
            device="cpu",
        )

    def __call__(self, buffer):

        if not self.training:
            return [self.eval_transform(buffer)]

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        if self.auto_augment:
            buffer = self.autoaug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        buffer = tensor_normalize(buffer, self.normalize[0], self.normalize[1])
        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return [buffer]


class EvalVideoTransform(object):

    def __init__(
        self,
        num_views_per_clip=1,
        short_side_size=224,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 通常使用 ImageNet 的经验值
                                                                   # 如果是 0-1 范围的张量，这些值能将数据拉回到正态分布附近
    ):
        self.views_per_clip = num_views_per_clip
        self.short_side_size = short_side_size
        self.spatial_resize = video_transforms.Resize(short_side_size, interpolation="bilinear")
        self.to_tensor = video_transforms.Compose(
            [
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
            ]
        )

    def __call__(self, buffer):

        # Sample several spatial views of each clip
        buffer = np.array(self.spatial_resize(buffer))
        T, H, W, C = buffer.shape

        num_views = self.views_per_clip
        side_len = self.short_side_size
        spatial_step = (max(H, W) - side_len) // (num_views - 1)

        all_views = []
        for i in range(num_views):
            start = i * spatial_step
            if H > W:
                view = buffer[:, start : start + side_len, :, :]
            else:
                view = buffer[:, :, start : start + side_len, :]
            view = self.to_tensor(view)
            all_views.append(view)

        return all_views


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


@torch.no_grad()
def project_query_token(query_param: nn.Parameter, radius: float = 1.0, eps: float = 1e-8):
    """
    将可学习query token投影到半径r的球面  (抑制“范数爆、softmax 变尖锐”)

    通常在 optimizer.step() 之后调用

    Args:
        query_param: 可学习参数，形状支持 [D], [1, D], [B, D], [1, 1, D], [B, 1, D]
        radius: 球面半径，默认1.0 ， 推荐0.5-2.0范围
        eps: 数值稳定性参数，防止除零
    """
    # 参数验证
    if not isinstance(query_param, nn.Parameter):
        raise TypeError("query_param must be a nn.Parameter")

    if radius <= 0:
        raise ValueError("radius must be positive")

    q = query_param.data

    # 计算范数
    norm = q.norm(dim=-1, keepdim=True)

    # 只在需要时进行投影（避免不必要的计算）
    if (norm > radius).any():
        scale = radius / norm.clamp(min=eps)
        scale = torch.where(norm > radius, scale, torch.ones_like(scale))
        query_param.data.mul_(scale)


def load_encoder_from_wrapper(encoder_model, checkpoint_path):
    """
    encoder_model: 你新实例化的独立 Encoder 对象
    checkpoint_path: 保存的权重文件路径 (通常是 model_states.pt)
    """
    # 1. 加载原始 state_dict
    full_state_dict = torch.load(checkpoint_path, map_location='cpu')

    # 如果是 DeepSpeed 保存的，真正的 state_dict 可能在 'module' 或 'model' 键下
    if 'module' in full_state_dict:
        full_state_dict = full_state_dict['module']

    # 2. 过滤并重命名 key
    # 目标：将 "encoder.backbone.xxx" 转换为 "backbone.xxx"
    encoder_state_dict = {}
    prefix = 'encoder.'

    for k, v in full_state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # 移除 "encoder." 前缀
            encoder_state_dict[new_key] = v

    # 3. 加载到独立的 encoder 中
    msg = encoder_model.load_state_dict(encoder_state_dict, strict=True)
    print(f"Encoder loaded with message: {msg}")

    return encoder_model