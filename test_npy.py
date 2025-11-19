import os
import sys
import copy

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config
from torch.cuda.amp import GradScaler
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch, eval_one_epoch_1f, npy_eval_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    parser.add_argument('--loss', type=str, default='l2', help='ce, l2')
    parser.add_argument('--inner_loss', type=str, default='none', help='cos, l2')
    parser.add_argument('--loss_clean', type=str, default='none', help='ce, l2')
    parser.add_argument('--is_attack', type=int, default=0, help='Adversarial attack type')
    parser.add_argument('--attack', type=str, default=None, help='Adversarial attack type')
    parser.add_argument('--eps', type=float, default=4, help='Epsilon for adversarial perturbation')
    parser.add_argument('--iterations_adv', type=int, default=5, help='Iterations for adversarial attack')
    parser.add_argument('--stepsize_adv', type=float, default=2., help='Step size for adversarial attack (no effect for apgd)')
    parser.add_argument('--norm', type=str, default='linf', help='Norm for attacks; linf, l2')
    parser.add_argument('--max_sample', type=int, default=None, help='sample number for testing')
    parser.add_argument('--thumos', type=int, default=1, help='wether it is thumos dataset.')
    parser.add_argument('--dataset', type=str, default='')  
    parser.add_argument('--na',type=str, default="na", help='remark.')
    parser.add_argument('--prop', type=float, default=None, help='resume from a checkpoint')
    parser.add_argument('--forwhole', type=int, default=1, help='resume from a checkpoint')
    parser.add_argument('--file_path', type=str, default="/tmp/tad/simple_transfer/", help='the checkpoint path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("gloo", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    if args.rank == 0:
        create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank, args=args)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        max_samples=args.max_sample,
        seed=42,
        thumos=args.thumos,
        dataset_name=args.dataset,
        **cfg.solver.test,
    )

    # build model
    model = build_detector(cfg.model)

    # DDP
    model = model.to(args.local_rank)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model._set_static_graph()
    # for params in model.parameters():
    #     params.requires_grad = False
    # model.requires_grad_(False)
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # load checkpoint: args -> config -> best
    if args.checkpoint != "none":
        checkpoint_path = args.checkpoint
    elif "test_epoch" in cfg.inference.keys():
        checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
    else:
        checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
    logger.info("Loading checkpoint from: {}".format(checkpoint_path))
    device = f"cuda:{args.rank % torch.cuda.device_count()}"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

    # Model EMA
    print(checkpoint.keys())    #dict_keys(['epoch', 'model', 'model_ema', 'optimizer', 'scheduler', 'mAP'])
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        # print(checkpoint["state_dict_ema"].skeys())
        model.load_state_dict(checkpoint["state_dict_ema"], strict=False)
        # model.load_state_dict(checkpoint["model_ema"])
        logger.info("Using Model EMA...")
    else:
        model.load_state_dict(checkpoint["state_dict"])

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # test the detector
    logger.info("Testing Starts...\n")
    # model_orig = copy.deepcopy(model)
    # model_orig.to(device)
    npy_eval_one_epoch(
        test_loader,
        model,
        # model_orig,
        cfg,
        logger,
        args.rank,
        model_ema=None,  # since we have loaded the ema model above
        use_amp=use_amp,
        world_size=args.world_size,
        not_eval=args.not_eval,
        args=args,
        scaler=scaler,
        attack=args.is_attack,
    )
    logger.info("Testing Over...\n")


if __name__ == "__main__":
    main()
