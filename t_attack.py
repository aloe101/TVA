import os
import sys
import copy

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config
from torch.cuda.amp import GradScaler
from opentad.models import build_detector, build_backbone
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.attack import image_attack
from opentad.attack.fgsm import fgsm_one_epoch, fs_one_epoch, fs_pgd_one_epoch, fs_mifgsm_one_epoch, ensemble_mifgsm_one_epoch, multilayer_mifgsm_one_epoch, bicos_mifgsm_one_epoch, taai_mifgsm_one_epoch, fs_difgsm_one_epoch, fs_sim_one_epoch, fs_bsr_one_epoch
from opentad.attack.ftm import fs_ftm_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--sur_checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    parser.add_argument('--loss', type=str, default='l1', help='ce, l2, l1')
    parser.add_argument('--inner_loss', type=str, default='none', help='cos, l2')
    parser.add_argument('--loss_clean', type=str, default='none', help='ce, l2')
    parser.add_argument('--is_attack', type=int, default=0, help='Adversarial attack type')
    parser.add_argument('--attack', type=str, default='none', help='Adversarial attack type')
    parser.add_argument('--eps', type=float, default=8, help='Epsilon for adversarial perturbation')
    parser.add_argument('--iterations_adv', type=int, default=4, help='Iterations for adversarial attack')
    parser.add_argument('--stepsize_adv', type=float, default=4., help='Step size for adversarial attack (no effect for apgd)')
    parser.add_argument('--decay', type=float, default=1., help='decay for momentum')
    parser.add_argument('--norm', type=str, default='linf', help='Norm for attacks; linf, l2')
    parser.add_argument('--max_sample', type=int, default=None, help='sample number for testing')
    parser.add_argument('--thumos', type=int, default=1, help='wether it is thumos dataset.')
    parser.add_argument('--na',type=str, default="na", help='remark.')
    parser.add_argument('--prop', type=float, default=None, help='resume from a checkpoint')
    parser.add_argument('--direction_image_model', type=str, default='resnet', help='resnet, densenet, squeezenet, vgg, alexnet')
    parser.add_argument('--depth', type=int, default=1, help='1,2,3,4')
    parser.add_argument('--attack_method', type=str, default='pgd', help='')
    parser.add_argument('--output_dir', type=str, default="/tmp/tad/simple_transfer/", help='')
    parser.add_argument('--ensemble', type=int, default=0, help='whether to use ensemble model')
    parser.add_argument('--multilayer', type=int, default=0, help='whether to use multilayer')
    parser.add_argument('--bicos', type=int, default=0, help='whether to use bicos')    #nargs='+', default=[]
    parser.add_argument('--smooth', type=int, default=0, help='')
    parser.add_argument('--flow', type=int, default=0, help='')
    parser.add_argument('--diversity', type=float, default=0.0, help='whether to use diversity')
    parser.add_argument('--tdrop', type=float, default=0.0, help='')
    parser.add_argument('--dual_mask', type=float, default=0.0, help='')
    parser.add_argument('--single', type=int, default=0, help='')
    parser.add_argument('--window_size', type=int, default=0, help='')
    parser.add_argument('--temp', type=float, default=0.01, help='')
    parser.add_argument('--dataset', type=str, default='thumos')

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
        seed=args.seed,
        thumos=args.thumos,
        **cfg.solver.test,
    )

    # build model
    dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
    model = build_backbone(cfg.model.backbone)
    model.dummy_param = dummy_param
    model = model.to(args.local_rank)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        logger.info("Using single model...")

    

    # test the detector
    logger.info("Attack Starts...\n")
    if args.attack_method == "pgd":
        fs_pgd_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    elif args.ensemble and args.attack_method == "mifgsm":
        ensemble_mifgsm_one_epoch(
        model_1=model_1,
            model_2=model_2,
            model_3=model_3,
            dataloader=test_loader,
            cfg=cfg,
            args=args,
            logger=logger,
    )
    
    elif (args.smooth or args.flow or args.tdrop or args.dual_mask) and args.attack_method == "mifgsm":
        logger.info("Using TAAI MIFGSM attack...")
        taai_mifgsm_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    elif args.bicos and args.attack_method == "mifgsm":
        logger.info("Using BICOS MIFGSM attack...")
        bicos_mifgsm_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    elif args.multilayer and args.attack_method == "mifgsm":
        multilayer_mifgsm_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    elif args.attack_method == "difgsm":
        fs_difgsm_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    elif args.attack_method == "ftm":
        logger.info(f"attack using ftm... \n bicon{args.bicos}; flow{args.flow}")
        fs_ftm_one_epoch(
        model=model,
            dataloader=test_loader,
            cfg=cfg,
            args=args,
            logger=logger,
    )
    elif args.attack_method == "sim":
        logger.info(f"attack using sim... ")
        fs_sim_one_epoch(
        model=model,
            dataloader=test_loader,
            cfg=cfg,
            args=args,
            logger=logger,
    )
    elif args.attack_method == "bsr":
        logger.info(f"attack using bsr... ")
        fs_bsr_one_epoch(
        model=model,
            dataloader=test_loader,
            cfg=cfg,
            args=args,
            logger=logger,
    )
    elif args.attack_method == "fgsm":
        fgsm_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    elif args.attack_method == "fs":
        fs_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    elif args.attack_method == "mifgsm":
        fs_mifgsm_one_epoch(
            model=model,
                dataloader=test_loader,
                cfg=cfg,
                args=args,
                logger=logger,
        )
    
    
    logger.info("Attack Over...\n")


if __name__ == "__main__":
    main()
