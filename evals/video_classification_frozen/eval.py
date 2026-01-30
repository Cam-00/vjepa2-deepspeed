# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import math
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import gc
from torch.utils.tensorboard import SummaryWriter

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger
from src.utils.gradient_monitor import GradientMonitor
from evals.video_classification_frozen.utils import project_query_token


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers")

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    accumulation_steps = args_opt.get("accumulation_steps")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    # opt_kwargs = [
    #     dict(
    #         ref_wd=kwargs.get("weight_decay"),
    #         final_wd=kwargs.get("final_weight_decay"),
    #         start_lr=kwargs.get("start_lr"),
    #         ref_lr=kwargs.get("lr"),
    #         final_lr=kwargs.get("final_lr"),
    #         warmup=kwargs.get("warmup"),
    #     )
    #     for kwargs in args_opt.get("multihead_kwargs")
    # ]

    opt_kwargs = [
        dict(
            base_start_lr=kwargs.get("base_start_lr"),
            base_ref_lr=kwargs.get("base_ref_lr"),
            base_final_lr=kwargs.get("base_final_lr"),
            cross_start_lr=kwargs.get("cross_start_lr"),
            cross_ref_lr=kwargs.get("cross_ref_lr"),
            cross_final_lr=kwargs.get("cross_final_lr"),
            head_start_lr=kwargs.get("head_start_lr"),
            head_ref_lr=kwargs.get("head_ref_lr"),
            head_final_lr=kwargs.get("head_final_lr"),
            base_ref_wd=kwargs.get("base_ref_wd"),
            base_final_wd=kwargs.get("base_final_wd"),
            cross_ref_wd=kwargs.get("cross_ref_wd"),
            cross_final_wd=kwargs.get("cross_final_wd"),
            warmup=kwargs.get("warmup"),
            head_keywords=kwargs.get("head_keywords"),
            cross_keywords=kwargs.get("cross_keywords"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%d", "iter"), ("%.5f", "train_loss1"),
                               ("%.5f", "train_loss2"), ("%.5f", "acc_max"), ("%.5f", "acc_min"), ("%.2e", "mem"))

    # Initialize model

    # -- init models  初始化模型参数，权重冻结
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    # -- init classifier
    classifiers = [
        AttentiveClassifier(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in opt_kwargs
    ]
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    print(classifiers[0])

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    # optimizer, scaler, scheduler, wd_scheduler = init_opt(
    #     classifiers=classifiers,
    #     opt_kwargs=opt_kwargs,
    #     iterations_per_epoch=ipe,
    #     num_epochs=num_epochs,
    #     use_bfloat16=use_bfloat16,
    #     accumulation_steps=accumulation_steps,
    # )

    optimizer, scaler, scheduler, wd_scheduler = init_group_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
        accumulation_steps=accumulation_steps,
    )



    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        if val_only:
            train_acc = -1.0
            train_losses = [-1.0, -1.0]
        else:
            train_acc, train_losses = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                accumulation_steps=accumulation_steps,
                rank=rank,
                epoch=epoch,
                csv_logger=csv_logger,
                folder=folder,
            )

        save_checkpoint(epoch + 1)

        val_acc, _ = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            accumulation_steps=accumulation_steps,
            rank=rank,
            epoch=epoch,
            csv_logger=csv_logger,
            folder=folder,
        )

        logger.info("[%5d] train_loss: [%.3f %.3f] train_acc: %.3f test_acc: %.3f%%"
                    % (epoch + 1, train_losses[0], train_losses[1], train_acc, val_acc))
        # if rank == 0:
        #     csv_logger.log(epoch + 1, train_losses[0][0], train_losses[1][0], train_acc, val_acc)

        if val_only:
            return


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    accumulation_steps,
    rank,
    epoch,
    csv_logger,
    folder,
):
    # 监控梯度统计量
    log_dir = os.path.join(folder, "gradient_monitor")
    writer = [SummaryWriter(log_dir=os.path.join(log_dir, f"classifier_{i}")) for i in range(len(classifiers))] # TensorBoard日志目录
    grad_monitor = [GradientMonitor(c, writer[i]) for i, c in enumerate(classifiers)]

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.05)  # 为了平滑梯度，减少梯度尖峰 而添加
    top1_meters = [AverageMeter() for _ in classifiers]
    ipe = len(data_loader)
    for itr, data in enumerate(data_loader):
        # 如果是eval, 只验证500步
        if training is False and itr > 500:
            break

        # if training:
        #     [s.step() for s in scheduler]
        #     [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)  # list[tensor(B, N, C)], N = T*H*W
                # print("outputs", len(outputs))  # 1
                # print("outputs[0]", outputs[0].size())  # torch.Size([10, 3136, 768])
                if (itr + 1) % 1000 == 0:
                    print(f"encoder outputs norm: {outputs[0].norm(p=2, dim=-1).mean(dim=-1).mean().item():.4f}")
                    print(f"encoder outputs std: {outputs[0].std().item():.4f}")
                    print(f"encoder outputs mean: {outputs[0].mean().item():.4f}")
                if not training:
                    outputs = [[c(o) for o in outputs] for c in classifiers]  # list[list[tensor]] , tensor:(B, num_class)
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]   # list[list[tensor]] , tensor:(B, num_class)
            # print("output[0] len:", len(outputs[0]))  # 1
            # print("output[1] len:", len(outputs[1]))  # 1
            # print("output[0][0].size:", outputs[0][0].size())  #  torch.Size([10, 174])

            # Compute loss
            # losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]
            losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]  # [[1], [1]]
            # print("losses before accumlation : ", losses)
            losses = [[los/accumulation_steps for los in loss] for loss in losses]
            # print("losses after accumlation : ", losses)

        if training:
            if use_bfloat16:
                [[s.scale(lij).backward() for lij in li] for s, li in zip(scaler, losses)]
                # [s.step(o) for s, o in zip(scaler, optimizer)
                # # 梯度自适应剪裁
                # [g.adaptive_gradient_clipping() for g in grad_monitor]

                if (itr + 1) % accumulation_steps == 0:
                    [s.unscale_(o) for s, o in zip(scaler, optimizer)]
                    # [torch.nn.utils.clip_grad_norm_(c.parameters(), max_norm=0.5) for c in classifiers]
                    # # 梯度自适应剪裁
                    # [g.adaptive_gradient_clipping() for g in grad_monitor]
                    # 监控梯度/权重统计量
                    if (itr + 1) % 1000 == 0:
                        [g.print_summary() for g in grad_monitor]
                        [g.log_to_tensorboard(epoch * ipe + itr) for g in grad_monitor]
                    else:
                        [g.log_to_tensorboard(epoch * ipe + itr) for g in grad_monitor]
                    [s.step(o) for s, o in zip(scaler, optimizer)]
                    [s.update() for s in scaler]
            else:
                [[lij.backward() for lij in li] for li in losses]
                # [o.step() for o in optimizer]
                # [torch.nn.utils.clip_grad_norm_(c.parameters(), max_norm=5) for c in classifiers]
                # # 梯度自适应剪裁
                # [g.adaptive_gradient_clipping() for g in grad_monitor]
                if (itr + 1) % accumulation_steps == 0:
                    # [s.unscale_(o) for s, o in zip(scaler, optimizer)]
                    # [torch.nn.utils.clip_grad_norm_(c.parameters(), max_norm=0.5) for c in classifiers]
                    # # 梯度自适应剪裁
                    # [g.adaptive_gradient_clipping() for g in grad_monitor]
                    # 监控梯度/权重统计量
                    if (itr + 1) % 1000 == 0:
                        [g.print_summary() for g in grad_monitor]
                        [g.log_to_tensorboard(epoch * ipe + itr) for g in grad_monitor]
                    else:
                        [g.log_to_tensorboard(epoch * ipe + itr) for g in grad_monitor]
                    # # 梯度自适应剪裁
                    # [g.adaptive_gradient_clipping() for g in grad_monitor]
                    [o.step() for o in optimizer]
            if (itr + 1) % accumulation_steps == 0:
                # 可学习 query token L2范数约束， 抑制“范数爆、softmax 变尖锐”
                [project_query_token(c.module.pooler.query_tokens, 0.5) for c in classifiers]
                [s.step() for s in scheduler]
                [wds.step() for wds in wd_scheduler]
                [o.zero_grad() for o in optimizer]
            # [o.zero_grad() for o in optimizer]

        losses_log = [loss[0].detach().item() * accumulation_steps for loss in losses]
        # print("losses_log : ", losses_log)
        with torch.no_grad():
            outputs = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
            top1_accs = [100.0 * coutputs.max(dim=1).indices.eq(labels).sum() / batch_size for coutputs in outputs]
            top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]
            for t1m, t1a in zip(top1_meters, top1_accs):
                t1m.update(t1a)

        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        _acc_top1 = np.array([t1m.val for t1m in top1_meters])
        # print("losses[0] size :", len(losses[0]))  # 1
        if itr % 10 == 0:
            logger.info(
                "[%5d] [%.3f%% %.3f%%] %.3f%% [%.3f%% %.3f%%] [losses: %.3f %.3f] [mem: %.2e] [%.3f%% %.3f%%]"
                % (
                    itr,
                    _agg_top1[0],
                    _agg_top1[1],
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    losses_log[0],
                    losses_log[1],
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    _acc_top1[0],
                    _acc_top1[1],
                )
            )
            if rank == 0:
                csv_logger.log(epoch, itr, losses_log[0], losses_log[1],
                               _agg_top1.max(), _agg_top1.min(), torch.cuda.max_memory_allocated() / 1024.0**2)

    # 本次epoch结束，如果有剩余梯度，强制更新
    if training:
        if ipe % accumulation_steps != 0:
            # 监控梯度/权重统计量
            [g.print_summary() for g in grad_monitor]
            [g.log_to_tensorboard((epoch + 1) * ipe) for g in grad_monitor]
            if use_bfloat16:
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [o.step() for o in optimizer]
            # 可学习 query token L2范数约束， 抑制“范数爆、softmax 变尖锐”
            [project_query_token(c.module.pooler.query_tokens, 0.5) for c in classifiers]
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]
            [o.zero_grad() for o in optimizer]

    [w.close() for w in writer]
    return _agg_top1.max(), losses_log


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading encoder
    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
    )

    # 防止内存泄漏 (原代码没有)
    # try:
    #     yield data_loader, data_sampler
    # finally:
    #     del data_loader
    #     gc.collect()
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, accumulation_steps, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * (iterations_per_epoch / accumulation_steps)),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * (iterations_per_epoch / accumulation_steps)))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * (iterations_per_epoch / accumulation_steps)))]
        scalers += [torch.amp.GradScaler("cuda") if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


def init_group_opt(classifiers, iterations_per_epoch, accumulation_steps, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):

        param_groups = param_groups_no_wd(
            model=c,
            base_start_lr=kwargs.get("base_start_lr"),
            base_ref_lr=kwargs.get("base_ref_lr"),
            base_final_lr=kwargs.get("base_final_lr"),
            cross_start_lr=kwargs.get("cross_start_lr"),
            cross_ref_lr=kwargs.get("cross_ref_lr"),
            cross_final_lr=kwargs.get("cross_final_lr"),
            head_start_lr=kwargs.get("head_start_lr"),
            head_ref_lr=kwargs.get("head_ref_lr"),
            head_final_lr=kwargs.get("head_final_lr"),
            base_ref_wd=kwargs.get("base_ref_wd"),
            base_final_wd=kwargs.get("base_final_wd"),
            cross_ref_wd=kwargs.get("cross_ref_wd"),
            cross_final_wd=kwargs.get("cross_final_wd"),
            mc_warmup_steps=int(kwargs.get("warmup") * (iterations_per_epoch / accumulation_steps)),
            head_keywords=kwargs.get("head_keywords"),
            cross_keywords=kwargs.get("cross_keywords"),
        )

        logger.info("Using AdamW...")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * (iterations_per_epoch / accumulation_steps)))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * (iterations_per_epoch / accumulation_steps)))]
        scalers += [torch.amp.GradScaler("cuda") if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


def param_groups_no_wd(model,
                       base_start_lr, base_ref_lr, base_final_lr,
                       cross_start_lr, cross_ref_lr, cross_final_lr,
                       head_start_lr, head_ref_lr, head_final_lr,
                       base_ref_wd, base_final_wd,
                       cross_ref_wd, cross_final_wd,
                       head_ref_wd=0.0, head_final_wd=0.0,
                       mc_warmup_steps=0.0,
                       head_keywords=None, cross_keywords=None, ln_keywords=None,
                       verbose=2):
    """
    智能参数分组，支持不同的学习率和权重衰减

    Args:
        model: 神经网络模型
        base_lr: 基础学习率
        head_lr: 分类头学习率
        cross_lr: 交叉注意力学习率
        wd: 权重衰减系数
        head_keywords: 分类头参数关键词
        cross_keywords: 交叉注意力参数关键词
        ln_keywords: LayerNorm参数关键词
        verbose: 是否打印分组信息
    """
    # 默认关键词
    if head_keywords is None:
        head_keywords = ["head", "classifier", "fc", "linear_out", "proj"]
    if cross_keywords is None:
        cross_keywords = ["cross", "cross_attn", "cross_block", "xfm"]
    if ln_keywords is None:
        ln_keywords = ["ln", "layernorm", "norm"]

    groups = {
        'decay': [], 'no_decay': [], 'cross_params': [], 'head_params': []
    }
    group_names = {key: [] for key in groups.keys()}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_bias = n.endswith(".bias")
        is_ln = any(keyword in n.lower() for keyword in ln_keywords)
        is_ln_or_bias = is_bias or is_ln

        is_cross = any(keyword in n.lower() for keyword in cross_keywords)
        is_head = any(keyword in n.lower() for keyword in head_keywords)

        if is_head:
            groups['head_params'].append(p)
            group_names['head_params'].append(n)
        elif is_cross:
            groups['cross_params'].append(p)
            group_names['cross_params'].append(n)
        elif is_ln_or_bias:
            groups['no_decay'].append(p)
            group_names['no_decay'].append(n)
        else:
            groups['decay'].append(p)
            group_names['decay'].append(n)

    # 打印分组信息
    if verbose:
        print("=" * 50)
        print("Parameter Grouping Summary:")
        print("=" * 50)
        for group_name in groups:
            count = len(groups[group_name])
            print(f"{group_name:15s}: {count:4d} parameters")

        if verbose > 1:  # 详细模式
            for group_name, names in group_names.items():
                if names:
                    print(f"\n{group_name}:")
                    for name in sorted(names):
                        print(f"  {name}")

    return [
        {"params": groups['decay'], "group_name": 'base',
         "base_start_lr": base_start_lr, "base_ref_lr": base_ref_lr, "base_final_lr": base_final_lr,
         "base_ref_wd": base_ref_wd, "base_final_wd": base_final_wd, "mc_warmup_steps": mc_warmup_steps},
        {"params": groups['no_decay'], "group_name": 'no_decay',
         "base_start_lr": base_start_lr, "base_ref_lr": base_ref_lr, "base_final_lr": base_final_lr,
         "weight_decay": 0.0, "mc_warmup_steps": mc_warmup_steps},
        {"params": groups['cross_params'], "group_name": 'cross',
         "cross_start_lr": cross_start_lr, "cross_ref_lr": cross_ref_lr, "cross_final_lr": cross_final_lr,
         "cross_ref_wd": cross_ref_wd, "cross_final_wd": cross_final_wd, "mc_warmup_steps": mc_warmup_steps},
        {"params": groups['head_params'], "group_name": 'head',
         "head_start_lr": head_start_lr, "head_ref_lr": head_ref_lr, "head_final_lr": head_final_lr,
         "head_ref_wd": head_ref_wd, "head_final_wd": head_final_wd, "mc_warmup_steps": mc_warmup_steps}
    ]


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            prefix = group.get('group_name')
            if prefix == 'no_decay':
                prefix = 'base'
            ref_lr = group.get(prefix + '_ref_lr')
            final_lr = group.get(prefix + '_final_lr')
            start_lr = group.get(prefix + '_start_lr')
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            prefix = group.get('group_name')
            if prefix == 'no_decay':
                ref_wd = 0
                final_wd = 0
            else:
                ref_wd = group.get(prefix + '_ref_wd')
                final_wd = group.get(prefix + '_final_wd')
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
