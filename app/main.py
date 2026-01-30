# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import pprint
from pathlib import Path
import os
import sys
import yaml
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes."
)

#  -- add DeepSpeed config
parser.add_argument("--deepspeed", action='store_true', help='enable DeepSpeed')
parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
                    help='DeepSpeed config file')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')


def deepspeed_main(fname, devices, deepspeed_config_path):
    """
    DeepSpeed multiprocessing start method
    """
    import logging
    from src.utils.logging import get_logger

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # -- setting CUDA device
    if devices and len(devices) > 0:
        if local_rank < len(devices):
            device_id = devices[local_rank].split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            torch.cuda.set_device(int(device_id))

    # -- setting log
    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logger.info(f"DeepSpeed mode - called-params {fname}")

    # -- load model config(.yaml)
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # -- add deepspeed params to args
    params['deepspeed'] = True
    params['deepspeed_config'] = deepspeed_config_path
    params['local_rank'] = local_rank
    params['world_size'] = world_size
    params['rank'] = rank
    params['devices'] = devices

    # only log config on rank 0
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    logger.info(f"Running with DeepSpeed... (rank: {rank}/{world_size})")

    # Initialize app, transfer all params including model config and deepspeed config
    app_main(params["app"], args=params)


def original_main(rank, fname, world_size, devices):
    """
    Legacy multiprocessing start method
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging
    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config
    app_main(params["app"], args=params)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.deepspeed:
        # DeepSpeed mode
        print(f"Running in DeepSpeed mode with config: {args.deepspeed_config}")
        deepspeed_main(args.fname, args.devices, args.deepspeed_config)

    else:
        # -- legacy mode
        if args.debugmode:
            original_main(rank=0, fname=args.fname, world_size=1, devices=["cuda:0"])
        else:
            num_gpus = len(args.devices)
            mp.set_start_method("spawn")
            for rank in range(num_gpus):
                mp.Process(
                    target=original_main,
                    args=(rank, args.fname, num_gpus, args.devices)
                ).start()