"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

modelcustom API requirements:

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.

API requirements for Predictor module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip tokens (shape=[batch_size x num_encoder_tokens x feature_dim])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import logging

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 原代码
# def init_module(
#     resolution: int,
#     frames_per_clip: int,
#     checkpoint: str,
#     # --
#     model_kwargs: dict,
#     wrapper_kwargs: dict,
# ):
#     logger.info(f"Loading pretrained model from {checkpoint}")
#     checkpoint = torch.load(checkpoint, map_location="cpu")
#
#     enc_kwargs = model_kwargs["encoder"]
#     enc_ckp_key = enc_kwargs.get("checkpoint_key")
#     enc_model_name = enc_kwargs.get("model_name")
#
#     model = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)
#
#     pretrained_dict = checkpoint[enc_ckp_key]
#     # --
#     pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
#     pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
#     for k, v in model.state_dict().items():
#         if k not in pretrained_dict:
#             logger.info(f'key "{k}" could not be found in loaded state dict')
#         elif pretrained_dict[k].shape != v.shape:
#             logger.info(f'key "{k}" is of different shape in model and loaded state dict')
#             pretrained_dict[k] = v
#     msg = model.load_state_dict(pretrained_dict, strict=False)
#     logger.info(f"loaded pretrained model with msg: {msg}")
#     print(model)
#
#     model = ClipAggregation(
#         model,
#         tubelet_size=model.tubelet_size,
#         **wrapper_kwargs,
#     )
#     del checkpoint
#     return model

# 优化后代码，可以加载deepspeed保存的checkpoint
def init_module(
        resolution: int,
        frames_per_clip: int,
        checkpoint: str,
        model_kwargs: dict,
        wrapper_kwargs: dict,
):
    logger.info(f"Loading checkpoint from {checkpoint}")
    # 1. 加载全量权重文件
    # 注意：如果是 DeepSpeed 保存的目录，请指向目录下的 mp_rank_00_model_states.pt
    # 或者使用 zero_to_fp32.py 合并后的文件
    ckpt_dict = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_model_name = enc_kwargs.get("model_name")

    # 2. 提取真正的 state_dict 核心
    # DeepSpeed 默认将模型包装在 'module' 键下
    if "module" in ckpt_dict:
        full_state_dict = ckpt_dict["module"]
    elif "encoder" in ckpt_dict:  # 兼容原有的旧格式
        full_state_dict = ckpt_dict["encoder"]
    else:
        full_state_dict = ckpt_dict

    # 3. 实例化基础 ViT 模型
    model = vit.__dict__[enc_model_name](
        img_size=resolution,
        num_frames=frames_per_clip,
        **enc_kwargs
    )

    # 4. 关键：针对 MultiModelWrapper 的键名转换
    # 目标：将 "target_encoder.backbone.patch_embed.weight" -> "backbone.patch_embed.weight"
    # 或者直接到具体参数名
    new_state_dict = {}
    prefix = "target_encoder."

    for k, v in full_state_dict.items():
        # 移除 DeepSpeed 可能添加的 module. 前缀
        clean_k = k.replace("module.", "")

        # 识别属于 target_encoder 的部分
        if clean_k.startswith(prefix):
            # 移除 "encoder."，保留后续结构（如 "backbone.xxx" 或 "patch_embed.xxx"）
            inner_k = clean_k[len(prefix):]
            new_state_dict[inner_k] = v

    # 5. 参数校验与对齐
    model_state = model.state_dict()
    for k in list(new_state_dict.keys()):
        if k not in model_state:
            # 尝试进一步剥离 "backbone." 前缀，以匹配基础 ViT 的结构
            if k.startswith("backbone."):
                simple_k = k.replace("backbone.", "")
                if simple_k in model_state:
                    new_state_dict[simple_k] = new_state_dict.pop(k)
                    continue
            logger.info(f'key "{k}" could not be matched to model')
        elif new_state_dict[k].shape != model_state[k].shape:
            logger.info(f'key "{k}" shape mismatch: {new_state_dict[k].shape} vs {model_state[k].shape}')
            new_state_dict.pop(k)

    # 6. 加载权重
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")

    # 7. 再次包装为 ClipAggregations
    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )

    del ckpt_dict
    return model

class ClipAggregation(nn.Module):
    """
    Process each clip indepdnently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        outputs = self.model(x)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size  # num temporal indices
            S = N // T  # num spatial tokens

            # Unroll outputs into a 2D array [spatial_views x temporal_views]
            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outputs in enumerate(all_outputs):
                # Concatenate along temporal dimension
                outputs = [o.reshape(B, T, S, D) for o in outputs]
                outputs = torch.cat(outputs, dim=1).flatten(1, 2)
                # Compute positional embedding
                if (self.pos_embed is not None) and (clip_indices is not None):
                    _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, max_T, D]
                    pos_embed = apply_masks(pos_embed, _indices, concat=False)  # list(Tensor([B, T, D]))
                    pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)  # [B, T*num_clips, S, D]
                    pos_embed = pos_embed.flatten(1, 2)
                    outputs += pos_embed
                all_outputs[i] = outputs

            return all_outputs

        return multiviews_postprocess(outputs)
