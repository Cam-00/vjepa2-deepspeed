# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings

# remove the origin SLURM setting, replaced by CLI params control
# # -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
# try:
#     # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
#     # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
#     # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
#     # --          TO EACH PROCESS
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
# except Exception:
#     pass

import copy
import gc
import random
import time

import deepspeed
import argparse
import numpy as np
import torch
import math
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_opt, init_video_model, load_checkpoint
from src.datasets.data_manager import init_data
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.wrappers import MultiModelWrapper
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.gradient_monitor import GradientMonitor
from src.utils.data_monitor import knn_eval
from src.utils.optimizer_tuning import get_param_groups


# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1    # checkpoint save frequency, epoch
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # --------------------------------------------------------------- #
    #  Passed in params from deepspeed config file (.json)
    # --------------------------------------------------------------- #

    # -- get deepspeed Cconfig
    deepspeed_config = args.get("deepspeed_config", None)
    local_rank = args.get("local_rank", -1)
    deepspeed_enabled = args.get("deepspeed", False)

    if deepspeed_enabled and local_rank == -1:
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        rank = int(os.environ.get('RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    else:
        rank = 0
        world_size = 1

    # ---------------------------------------------------------------- #
    #  Passed in params from model config file (.yaml)
    # ---------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    model_eval = cfgs_meta.get("model_eval")
    eval_freq = cfgs_meta.get("eval_freq")
    monitor_freq = cfgs_meta.get("monitor_freq")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    load_checkpoint_epoch = cfgs_meta.get("load_checkpoint_epoch", 0)
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    save_checkpoint_steps = cfgs_meta.get("save_checkpoint_steps", 100)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    enable_probe = cfgs_meta.get("enable_probe")
    accumulation_steps = cfgs_meta.get("accumulation_steps")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA FOR TRAIN
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), \
            "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA FOR VAL
    probe_cfgs_data = args.get("data_probe")
    probe_dataset_type = probe_cfgs_data.get("dataset_type", "videodataset")
    probe_train_dataset_paths = probe_cfgs_data.get("datasets_train", [])
    probe_val_dataset_paths = probe_cfgs_data.get("datasets_val", [])
    datasets_weights_train = probe_cfgs_data.get("datasets_weights_train")
    datasets_weights_val = probe_cfgs_data.get("datasets_weights_val")
    probe_dataset_fpcs = probe_cfgs_data.get("dataset_fpcs")
    probe_max_num_frames = max(dataset_fpcs)
    if datasets_weights_train is not None:
        assert len(datasets_weights_train) == len(probe_train_dataset_paths), \
            "Must have one probe train sampling weight specified for each dataset"
    if datasets_weights_train is not None:
        assert len(datasets_weights_val) == len(probe_val_dataset_paths), \
            "Must have one prob val sampling weight specified for each dataset"
    batch_size_probe = probe_cfgs_data.get("batch_size")
    fps_probe = probe_cfgs_data.get("fps")
    crop_size_probe = probe_cfgs_data.get("crop_size", 224)
    pin_mem_probe = probe_cfgs_data.get("pin_mem", False)
    num_workers_probe = probe_cfgs_data.get("num_workers", 1)
    persistent_workers_probe = probe_cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    anneal = cfgs_opt.get("anneal")
    start_lr = float(cfgs_opt.get("start_lr"))
    lr = float(cfgs_opt.get("lr"))
    final_lr = float(cfgs_opt.get("final_lr"))
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = float(cfgs_opt.get("eps", 1.0e-8))
    base_lr_encoder = float(cfgs_opt.get("base_lr_encoder"))
    base_lr_predictor = float(cfgs_opt.get("base_lr_predictor"))

    # ----------------------------------------------------------------------- #
    # Passed in params from model config file (.yaml) --- Finished
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    #  Initialize setting for training
    # ----------------------------------------------------------------------- #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    if deepspeed_enabled:
        from datetime import timedelta
        deepspeed.init_distributed(dist_backend="nccl",timeout=timedelta(minutes=60))
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if deepspeed_enabled:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            # origin for environment isolation, each gpu can see each other only in node where they are.
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device='cpu' if deepspeed_enabled else device,  # initialize from cpu in DeepSpeed mode for saving VRAM
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model and not deepspeed_enabled:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()
    elif compile_model and deepspeed_enabled:
        logger.warning("Model compilation is disabled when using DeepSpeed")

    # multi-blocks masks and 2 mask sampling strategies (short-range masks / long-range masks)
    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    # -- data augment for training mode, including:
    #   resizing clips to a uniform spatial scale(crop_size : 224*224 or 256*256)
    #   video data shape from decord(T H W C) ---> encoder input shape(B, C, T, H, W)
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers for training model
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,  # if dataset is ImageNet, then load ImageNet train sets
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights,
        persistent_workers=persistent_workers,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
    )
    try:
        _dlen = len(unsupervised_loader)
        logger.info(f"access dataset length successfully: {_dlen}")
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:   # ipe: iterations_per_epoch, default ipe = 300
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- data augment for val mode
    val_transform = make_transforms(
        auto_augment=False,
        crop_size=crop_size_probe,
        eval_mode=True,
    )

    # -- init data-loaders for val mode
    (probe_train_dataloader, _) = init_data(
        data=probe_dataset_type,
        root_path=probe_train_dataset_paths,
        batch_size=batch_size_probe,
        training=True,    # if dataset is ImageNet, then load ImageNet train sets
        dataset_fpcs=probe_dataset_fpcs,
        fps=fps_probe,
        transform=val_transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights_train,
        persistent_workers=persistent_workers_probe,
        collator=None,
        num_workers=num_workers_probe,
        pin_mem=pin_mem_probe,
        log_dir=None,
        enable_probe=enable_probe,
    )
    try:
        _ptdlen = len(probe_train_dataloader)
        logger.info(f"access probe train dataset length successfully: {_ptdlen}")
    except Exception:  # Different interface for webdataset
        _ptdlen = probe_train_dataloader.num_batches

    (probe_val_dataloader, _) = init_data(
        data=probe_dataset_type,
        root_path=probe_val_dataset_paths,
        batch_size=batch_size_probe,
        training=False,
        dataset_fpcs=probe_dataset_fpcs,
        fps=fps_probe,
        transform=val_transform,
        rank=rank,
        world_size=world_size,
        datasets_weights=datasets_weights_val,
        persistent_workers=persistent_workers_probe,
        collator=None,
        num_workers=num_workers_probe,
        pin_mem=pin_mem_probe,
        log_dir=None,
        enable_probe=enable_probe,
    )
    try:
        _pvdlen = len(probe_val_dataloader)
        logger.info(f"access probe val dataset length successfully: {_pvdlen}")
    except Exception:  # Different interface for webdataset
        _pvdlen = probe_val_dataloader.num_batches

    # -- init optimizer and scheduler
    if not deepspeed_enabled:
        # origin mode
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            encoder=encoder,
            predictor=predictor,
            wd=wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=warmup,
            anneal=anneal,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            mixed_precision=mixed_precision,
            betas=betas,
            eps=eps,
            accumulation_steps=accumulation_steps,
        )
    else:
        # DeepSpeed mode
        optimizer = None
        scaler = None
        scheduler = None
        wd_scheduler = None

    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    from deepspeed.runtime.lr_schedules import WarmupLR, WarmupCosineLR

    deepspeed_model = MultiModelWrapper(encoder, predictor, target_encoder)
    optimizer_params = get_param_groups(deepspeed_model, base_lr_encoder=base_lr_encoder,
                                        base_lr_predictor=base_lr_predictor, base_wd=wd)

    # if use ZeRO-Offload (save states of optimizer on CPU ), then choose DeepSpeedCPUAdam
    # if training on GPU(ZeRO-1/2)，then choose FusedAdam (torch_adam: false)
    if deepspeed_enabled:
        optimizer = FusedAdam(
            optimizer_params,
            lr=base_lr_encoder,     # float
            weight_decay=wd,        # float
            betas=betas,            # float
            eps=eps,                # float
            adam_w_mode=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_params, betas=betas, eps=eps)

    lr_scheduler = WarmupCosineLR(optimizer, total_num_steps=num_epochs * ipe, warmup_num_steps=warmup * ipe,
                                  warmup_min_ratio=0.2, cos_min_ratio=0.002, warmup_type='linear')

    # Initialize model
    if deepspeed_enabled:
        # Initialize deepspeed model
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=deepspeed_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            model_parameters=None,
            config=deepspeed_config,
            dist_init_required=False,
        )
    else:
        # origin mode
        if torch.cuda.device_count() > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            encoder = DDP(encoder.to(device), device_ids=[rank])
            predictor = DDP(predictor.to(device), device_ids=[rank])
            target_encoder = DDP(target_encoder.to(device), device_ids=[rank])
        else:
            encoder = encoder.to(device)
            predictor = predictor.to(device)
            target_encoder = target_encoder.to(device)

        for p in target_encoder.parameters():
            p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe // accumulation_steps * num_epochs)
        for i in range(int(ipe // accumulation_steps * num_epochs) + 1)
    )

    def save_checkpoint(epoch, global_step, path):

        if deepspeed_enabled:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()

            client_state = {
                "epoch": epoch,
                "loss": float(loss_meter.avg),
                "batch_size": batch_size,
                "world_size": world_size,
                "global_step": global_step,
            }
            # save the latest checkpoint
            model_engine.save_checkpoint(path, client_state=client_state, tag=f"latest_ck", save_latest=True)

            if dist.is_initialized():
                dist.barrier()
        else:
            # origin save
            save_dict = {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "epoch": epoch,
                "loss": loss_meter.avg,
                "batch_size": batch_size,
                "world_size": world_size,
                "lr": lr,
            }
            try:
                torch.save(save_dict, path)
            except Exception as e:
                logger.info(f"Encountered exception when saving checkpoint: {e}")

    start_epoch = 0
    start_global_step = 0

    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        if deepspeed_enabled:
            if load_checkpoint_epoch > 0:
                _, model_client_state = model_engine.load_checkpoint(
                    load_dir=load_path,
                    tag=f"latest_ck",
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True
                )
                if model_client_state:
                    start_epoch = model_client_state.get("epoch", 0)
                    start_global_step = model_client_state.get("global_step", 0)
                    logger.info(f"Resuming from epoch {start_epoch}, itr {start_global_step}")

                logger.info(f"DeepSpeed checkpoints loaded successfully")
        else:
            # origin load
            (
                encoder,
                predictor,
                target_encoder,
                optimizer,
                scaler,
                start_epoch,
            ) = load_checkpoint(
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
            )
            for _ in range(start_epoch * (ipe // accumulation_steps)):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    if model_eval:
        if deepspeed_enabled:
            model_engine.eval()
            for p in model_engine.parameters():
                p.requires_grad = False
            # get sub-model from deepspeed model
            encoder_engine = model_engine.encoder
            predictor_engine = model_engine.predictor
            target_encoder_engine = model_engine.target_encoder
        else:
            target_encoder.eval()
    else:
        encoder_engine = model_engine.encoder
        predictor_engine = model_engine.predictor
        target_encoder_engine = model_engine.target_encoder

    # -- Initialize loader
    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    skip_batches = start_epoch * ipe + start_global_step

    # skip already trained batches
    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        # -- update distributed-data-loader epoch
        for itr in range(skip_batches):
            if (itr + 1) % 10 == 0:
                logger.info(f"Skip {itr + 1}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- Monitor gradient statistics
    models_name = ["encoder", "target_encoder", "predictor"]
    models = [encoder_engine, target_encoder_engine, predictor_engine]
    log_dir = os.path.join(folder, "gradient_monitor")
    writer = [SummaryWriter(log_dir=os.path.join(log_dir, f"{nm}")) for nm in models_name]
    grad_monitor = [GradientMonitor(m, writer[i]) for i, m in enumerate(models)]

    knn_log_dir = os.path.join(folder, "knn_accuracy_and_feature_variance_monitor")
    writer_knn = SummaryWriter(log_dir=knn_log_dir)

    # ----------------------------------------------------------------------- #
    #  Initialize setting for training --- Finished
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    #  TRAINING LOOP --- Beginning
    # ----------------------------------------------------------------------- #

    for epoch in range(start_epoch, num_epochs):
        if start_global_step == ipe:
            start_global_step = 0
            continue
        logger.info("Epoch %d" % epoch)

        loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        new_lr = 0.0
        new_wd = 0.0

        for itr in range(start_global_step, ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    # -- get next batch data
                    # sample: a list, its item is tuple: (collated_batch, collated_masks_enc, collated_masks_pred)
                    # collated_batch: a tuple, (batch_buffer, batch_label, batch_clip_indices)
                    # batch_buffer: video data, a list, len is num_clips, its item is tensor[B, C, T, H, W]  --->
                    # (num_clips, B, C, T, H, W)
                    # batch_label: label, a tensor[bs], its item is an int number or one value
                    # batch_clip_indices: a list, len is num_clips, its item is tensor[bs, fpc]  --->
                    # (num_clips, bs, clip_len)
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            for _fpc_sample in sample:
                bs, fpc = _fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            def load_clips():
                all_clips, all_masks_enc, all_masks_pred = [], [], []
                for fpc_sample in sample:
                    udata, masks_enc, masks_pred = fpc_sample
                    if deepspeed_enabled:
                        device_data = udata[0][0].to(device, non_blocking=True)
                    else:
                        device_data = udata[0][0].to(device, non_blocking=True)
                    all_clips += [device_data]
                    all_masks_enc += [[m.to(device, non_blocking=True) for m in masks_enc]]
                    all_masks_pred += [[m.to(device, non_blocking=True) for m in masks_pred]]
                return all_clips, all_masks_enc, all_masks_pred

            clips, masks_enc, masks_pred = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                global new_lr, new_wd

                if not deepspeed_enabled:
                    if itr == 0:
                        new_lr = scheduler.step()
                        new_wd = wd_scheduler.step()
                    if (itr + 1) % accumulation_steps == 0:
                        new_lr = scheduler.step()
                        new_wd = wd_scheduler.step()

                def forward_target(cs):
                    with torch.no_grad():
                        if deepspeed_enabled:
                            h = model_engine(cs, masks_enc, masks_pred, mode="target")
                        else:
                            h = target_encoder(cs)
                        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                        return h

                def forward_context(cs):
                    if deepspeed_enabled:
                        z = model_engine(cs, masks_enc, masks_pred, mode="train")
                    else:
                        z = encoder(cs, masks_enc)
                        z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    # Assumption: predictor will have returned only masked tokens for z
                    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

                    loss, n = 0, 0
                    for zi, hi in zip(z, h):
                        for zij, hij in zip(zi, hi):
                            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                            n += 1
                    if n == 0:
                        print("Warning : loss will be divided by zero")
                    loss /= n

                    if not deepspeed_enabled:
                        loss /= accumulation_steps
                    return loss

                # Forward
                if deepspeed_enabled:
                    if isinstance(clips, list):
                        cs = [c.to(dtype=dtype) for c in clips]
                    else:
                        cs = clips.to(dtype=dtype)
                    h = forward_target(cs)
                    z = forward_context(cs)
                    loss = loss_fn(z, h)

                    # DeepSpeed backward
                    model_engine.backward(loss)

                    # monitor gradients
                    if (itr + 1) % monitor_freq == 0:
                        [g.log_to_tensorboard(epoch * ipe + itr) for g in grad_monitor]

                    # DeepSpeed update
                    model_engine.step()
                else:
                    with torch.amp.autocast('cuda', dtype=dtype, enabled=mixed_precision):
                        h = forward_target(clips)
                        z = forward_context(clips)
                        loss = loss_fn(z, h)  # jepa prediction loss

                    #  Backward & step
                    if mixed_precision:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # -- accumulation steps
                    if (itr + 1) % accumulation_steps == 0:
                        [g.log_to_tensorboard(epoch * ipe + itr) for g in grad_monitor]
                        if mixed_precision:
                            scaler.unscale_(optimizer)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        optimizer.zero_grad()

                # # monitor lr & weight decay
                # if rank == 0 and (itr + 1) % ipe == 0:
                #     print(f"\nStep {itr + 1}:")
                #     for i, group in enumerate(optimizer.param_groups):
                #         group_lr = group['lr']
                #         group_wd = group['weight_decay']
                #         group_name = group.get('name', f'Group {i}')
                #         print(f"  {group_name:40} : lr = {group_lr:.2e}    weight_decay = {group_wd:.2e}")

                # -- momentum update
                if (itr + 1) % accumulation_steps == 0 or deepspeed_enabled:
                    # momentum update of target encoder
                    m = next(momentum_scheduler)
                    with torch.no_grad():
                        params_k = []
                        params_q = []
                        if deepspeed_enabled:
                            for param_q, param_k in zip(encoder_engine.parameters(),
                                                        target_encoder_engine.parameters()):
                                params_k.append(param_k)
                                params_q.append(param_q)
                        else:
                            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                                params_k.append(param_k)
                                params_q.append(param_q)

                        torch._foreach_mul_(params_k, m)
                        torch._foreach_add_(params_k, params_q, alpha=1 - m)

                    if deepspeed_enabled:
                        current_lr = model_engine.get_lr()[0] if isinstance(model_engine.get_lr(),
                                                                            list) else model_engine.get_lr()
                        opt = model_engine.optimizer
                        if hasattr(opt, 'param_groups'):
                            current_wd = opt.param_groups[0].get('weight_decay', 0.0)
                        else:
                            current_wd = wd
                        return float(loss), current_lr, current_wd
                    else:
                        return (float(loss * accumulation_steps), new_lr, new_wd)

            if not model_eval:
                (loss, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(train_step)
                loss_meter.update(loss)
                gpu_time_meter.update(gpu_etime_ms)

            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            iter_time_meter.update(iter_elapsed_time_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if ((itr+1) % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch,
                            itr + 1,
                            loss_meter.avg,
                            "[" + ", ".join([f"{k}: " + "%.1f" % mask_meters[k].avg for k in mask_meters]) + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            if not model_eval:
                log_stats()
                assert not np.isnan(loss), "loss is nan"

            # save the latest checkpoint in fixed step for deepspeed mode
            if (itr + 1) % save_checkpoint_steps == 0:
                save_checkpoint(epoch, itr + 1, latest_path)

            # knn eval
            if (itr + 1) % eval_freq == 0:
                # calculate k-NN Acc / embedding variance
                acc, embed_var, embed_mean = knn_eval(model_engine.module.encoder, probe_train_dataloader,
                                                      probe_val_dataloader, device, k=5)

                # save to TensorBoard
                writer_knn.add_scalar('Monitor/Feature_Variance', embed_var, epoch * ipe + itr + 1)
                writer_knn.add_scalar('Monitor/Feature_Mean', embed_mean, epoch * ipe + itr + 1)
                writer_knn.add_scalar('Monitor/kNN_Accuracy', acc, epoch * ipe + itr + 1)

                print(
                    f"Epoch {epoch + 1} | Variance: {embed_var:.9f} | Mean: {embed_mean:.9f} | k-NN Acc: {acc:.9f} \n")

        # -- Save Checkpoint
        if not model_eval:
            logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch, ipe, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0 or (epoch + 1) == warmup:
                save_every_file = f"e{epoch+1}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch, ipe, save_every_path)

        start_global_step = 0

    for w in writer:
        w.close()
    writer_knn.close()

    logger.info("Training completed successfully!")