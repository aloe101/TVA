# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import requests
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from typing import Callable
import scipy.stats as st
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import pandas as pd
import glob
import argparse
import json
import logging
import os
from datetime import datetime
import time
import random

warnings.filterwarnings("ignore")

# CUDA_VISIBLE_DEVICES=3 python attack.py --benchmark mvbench --attack_method mifgsm --bicos 1 --flow 1

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Parameters")

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--benchmark", type=str)

    parser.add_argument("--flag_top_p", action='store_true')
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--flag_top_k", action='store_true')
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument('--eps', type=float, default=8, help='Epsilon for adversarial perturbation')
    parser.add_argument('--iterations_adv', type=int, default=20, help='Iterations for adversarial attack')
    parser.add_argument('--stepsize_adv', type=float, default=1., help='Step size for adversarial attack (no effect for apgd)')
    parser.add_argument('--decay', type=float, default=1.0, help='Decay for momentum')
    parser.add_argument('--loss', type=str, default='l1', help='ce, l2, l1')
    parser.add_argument('--temp', type=float, default=0.01, help='')
    parser.add_argument('--bicos', type=int, default=0, help='whether to use bicos')
    parser.add_argument('--single', type=int, default=0, help='whether to use bicos')
    parser.add_argument('--flow', type=int, default=0, help='')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument('--attack_method', type=str, default='pgd', help='')
    parser.add_argument('--output_dir', type=str, default="/projects/hui007/tad/mllm_transfer/", help='')
    parser.add_argument('--diversity_prob', type=float, default=0.5, help='Probability of diversity')
    parser.add_argument('--data_idx', type=int, default=0, help='')
    return parser.parse_args()
args = parse_args()
def setup_logger(name, log_dir=None):
    # 创建logger实例
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 如果指定了日志目录，创建文件处理器
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            # 生成带时间戳的日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"log_{timestamp}.txt")
            
            # 创建文件处理器
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            # 为文件处理器设置格式（包含时间戳）
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(message)s", 
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
        
        # 为控制台处理器设置格式
        console_formatter = logging.Formatter("%(message)s")
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
    
    return logger

def load_video(video_path, max_frames_num, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def load_video_v2(video_path, max_frames_num, force_sample=False, start=None, end=None):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    video_time = total_frame_num / avg_fps

    # Convert start and end (in seconds) to frame indices
    start_frame = int(start * avg_fps) if start is not None else 0
    end_frame = int(end * avg_fps) if end is not None else total_frame_num

    # Clamp values to valid frame range
    start_frame = max(0, min(start_frame, total_frame_num - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frame_num))

    # Default: sample one frame per second
    frame_interval = int(avg_fps)
    frame_idx = list(range(start_frame, end_frame, frame_interval))

    # Force uniform sampling if requested or too many frames
    if len(frame_idx) > max_frames_num or force_sample:
        frame_idx = np.linspace(start_frame, end_frame - 1, max_frames_num, dtype=int).tolist()

    frame_time = [i / avg_fps for i in frame_idx]
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])

    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames

class BiContrastiveLoss(nn.Module):
    def __init__(self, initial_temp=1, final_temp=0.07, total_batches=5, set_temp=0.1):
        super(BiContrastiveLoss, self).__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self.decay_rate = self.calculate_decay_rate()
        self.set_temp = set_temp
 
    def forward(self, clean_feat, adv_feat, bicos, single=False):
        T, D, H = clean_feat.shape
        clean_feat = clean_feat.mean(dim=1)
        adv_feat = adv_feat.mean(dim=1)
        if bicos == 1:
            clean_feat = clean_feat.reshape(H, T)
            adv_feat = adv_feat.reshape(H, T)
            clean_feat = F.normalize(clean_feat, p=2, dim=1)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)
    
            cos_sim = torch.matmul(clean_feat, adv_feat.t()) / self.current_temp
            print(f"self.current_temp: {self.current_temp}")
            print(f"single: {single}")
    
            labels = torch.eye(cos_sim.size(0), device=cos_sim.device)
            loss_adv_2clean = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()

            if single==1:
                loss_clean2adv = 0.0
                loss = loss_adv_2clean
            elif single == 2:
                loss_adv_2clean == 0.0
                loss = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
            else:
                loss_clean2adv = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
                loss = (loss_clean2adv + loss_adv_2clean) / 2
        elif bicos == 2:
            clean_feat = clean_feat.reshape(B, D*T)
            adv_feat = adv_feat.reshape(B, D*T)
            clean_feat = F.normalize(clean_feat, p=2, dim=1)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)
    
            cos_sim = torch.matmul(clean_feat, adv_feat.t()) / self.current_temp
            print(f"self.current_temp: {self.current_temp}")
    
            labels = torch.eye(cos_sim.size(0), device=cos_sim.device)
            loss_clean2adv = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
            loss_adv_2clean = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()
    
            loss = (loss_clean2adv + loss_adv_2clean) / 2
        return loss
    def update_temperature(self, batch_count):
        self.current_temp = self.initial_temp * np.exp(-self.decay_rate * batch_count)
        # if self.current_temp <= 0.02:
        #     self.current_temp = 0.02

    def calculate_decay_rate(self):
        return -np.log(self.final_temp / self.initial_temp) / self.total_batches    #0.532

def norm_grad(grad):
    norm = torch.mean(torch.abs(grad), tuple(range(1, grad.ndim)), keepdim=True)
    grad = grad / (norm + 1e-8)
    return grad


if args.loss == 'l2':
    loss_fn = nn.MSELoss(reduction='mean')
elif args.loss == 'l1':
    loss_fn = nn.L1Loss(reduction='mean')
criterion = BiContrastiveLoss(initial_temp=0.01, final_temp=0.01, total_batches=args.iterations_adv)
OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
def transform_video(video, mode='forward'):
    dtype = video.dtype
    mean = torch.as_tensor(OPENAI_DATASET_MEAN, dtype=dtype, device=model.device)
    std = torch.as_tensor(OPENAI_DATASET_STD, dtype=dtype, device=model.device)
    if mode == 'forward':
            # [-mean/std, mean/std]
        video.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    elif mode == 'back':
            # [0, 1]
        video.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
    return video

def pgd_attack(video, ori_embedding):
    video = video.clone().detach()
    unnorm_video = transform_video(video.permute(1,0,2,3), mode='back').permute(1,0,2,3)
    perturbation = torch.zeros_like(video).uniform_(-args.eps/255, args.eps/255)
    perturbation.requires_grad = True
    for i in range(args.iterations_adv):
        adv_embedding = vision_tower(video+perturbation)
        loss_1 = loss_fn(adv_embedding, ori_embedding.detach())
        loss_2 = 0.0
        if args.bicos:
            loss_2 = criterion(adv_embedding, ori_embedding.detach(), args.bicos, single=args.single)
        if args.flow:
            cos_sim = F.cosine_similarity(adv_embedding[:,:,1:], adv_embedding[:,:,:-1])
            loss_3 = args.flow * (1-cos_sim).mean()
            del cos_sim
        else:
            loss_3 =0.0
        loss = loss_1+loss_2+loss_3
                    # loss = adv_videos.sum()
        start_time = time.time()
        gradient = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
        end_time = time.time()
                
        if i % 5 ==0:
            logger.info(f"Feature space {args.loss} {args.attack_method}: [data_idx: {data_idx}][{i}/{args.iterations_adv}] loss={loss.item()} l1 loss={loss_1.item() if isinstance(loss_1, torch.Tensor) else loss_1} bicon loss={loss_2.item() if isinstance(loss_2, torch.Tensor) else loss_2} flow loss={loss_3.item() if isinstance(loss_3, torch.Tensor) else loss_3}")
        # gradient = norm_grad(gradient)

        perturbation = perturbation + args.stepsize_adv * gradient.sign()
        perturbation = torch.clamp(perturbation, min=-args.eps/255, max=args.eps/255)
        adv_videos = torch.clamp(unnorm_video + perturbation, min=0, max=1)
        perturbation = adv_videos - unnorm_video
        adv_videos = transform_video(adv_videos.clone().detach().permute(1,0,2,3), mode='forward').permute(1,0,2,3)
        del adv_embedding, gradient, loss, loss_1, loss_2, loss_3
        torch.cuda.empty_cache()
    return adv_videos

def mifgsm_attack(video, ori_embedding):
    video = video.clone().detach()
    unnorm_video = transform_video(video.permute(1,0,2,3), mode='back').permute(1,0,2,3)
    
    perturbation = torch.zeros_like(video).uniform_(-args.eps/255, args.eps/255)
    perturbation.requires_grad = True
    momentum = torch.zeros_like(perturbation)

    for i in range(args.iterations_adv):
        adv_embedding = vision_tower(video+perturbation)
        loss_1 = loss_fn(adv_embedding, ori_embedding.detach())
        loss_2 = 0.0
        if args.bicos:
            loss_2 = criterion(adv_embedding, ori_embedding.detach(), args.bicos, single=args.single)
        if args.flow:
            cos_sim = F.cosine_similarity(adv_embedding[:,:,1:], adv_embedding[:,:,:-1])
            loss_3 = args.flow * (1-cos_sim).mean()
            del cos_sim
        else:
            loss_3 =0.0
        loss = loss_1+loss_2+loss_3
                    # loss = adv_videos.sum()
        start_time = time.time()
        gradient = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
        end_time = time.time()
                
        if i% 5 == 0:
            logger.info(f"Feature space {args.loss} {args.attack_method}: [data_idx: {data_idx}][{i}/{args.iterations_adv}] loss={loss.item()} l1 loss={loss_1.item() if isinstance(loss_1, torch.Tensor) else loss_1} bicon loss={loss_2.item() if isinstance(loss_2, torch.Tensor) else loss_2} flow loss={loss_3.item() if isinstance(loss_3, torch.Tensor) else loss_3}")
        gradient = norm_grad(gradient)
        momentum = args.decay * momentum + gradient

        perturbation = perturbation + args.stepsize_adv * momentum.sign()
        perturbation = torch.clamp(perturbation, min=-args.eps/255, max=args.eps/255)
        adv_videos = torch.clamp(unnorm_video + perturbation, min=0, max=1)
        perturbation = adv_videos - unnorm_video
        adv_videos = transform_video(adv_videos.clone().detach().permute(1,0,2,3), mode='forward').permute(1,0,2,3)
        criterion.update_temperature(i)
        del adv_embedding, gradient, loss, loss_1, loss_2, loss_3
        torch.cuda.empty_cache()
    return adv_videos

def difgsm_attack(video, ori_embedding):
    def input_diversity(video):
        if random.random() > args.diversity_prob:
            return video
        v_size = video.shape[-1]
        video = video.unsqueeze(0)
        B,C,T,H,W = video.shape
        # video = video.squeeze(1)
        video = video.view(-1, 3, H, W)
        rnd = random.randint(int(v_size*0.9), v_size)
        rescaled = F.interpolate(video, size=(rnd, rnd), mode='bilinear', align_corners=False)
        rescaled = rescaled.view(B, C, T, rnd, rnd)
        pad_top = random.randint(0, v_size-rnd)
        pad_bottom = v_size - rnd - pad_top
        pad_left = random.randint(0, v_size-rnd)
        pad_right = v_size - rnd - pad_left
        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)
        # padded = padded.unsqueeze(0)
        return padded.squeeze(0)
    video = video.clone().detach()
    perturbation = torch.zeros_like(video).uniform_(-args.eps/255, args.eps/255)
    perturbation.requires_grad = True
    unnorm_video = transform_video(video.permute(1,0,2,3), mode='back').permute(1,0,2,3)

    for i in range(args.iterations_adv):
        adv_embedding = vision_tower(input_diversity(video+perturbation))
        loss_1 = loss_fn(adv_embedding, ori_embedding.detach())
        loss_2 = 0.0
        if args.bicos:
            loss_2 = criterion(adv_embedding, ori_embedding.detach(), args.bicos, single=args.single)
        if args.flow:
            cos_sim = F.cosine_similarity(adv_embedding[:,:,1:], adv_embedding[:,:,:-1])
            loss_3 = args.flow * (1-cos_sim).mean()
            del cos_sim
        else:
            loss_3 =0.0
        loss = loss_1+loss_2+loss_3
                    # loss = adv_videos.sum()
        start_time = time.time()
        gradient = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
        end_time = time.time()
                
        if i % 5 ==0:
            logger.info(f"Feature space {args.loss} {args.attack_method}: [data_idx: {data_idx}][{i}/{args.iterations_adv}] loss={loss.item()} l1 loss={loss_1.item() if isinstance(loss_1, torch.Tensor) else loss_1} bicon loss={loss_2.item() if isinstance(loss_2, torch.Tensor) else loss_2} flow loss={loss_3.item() if isinstance(loss_3, torch.Tensor) else loss_3}")
        gradient = norm_grad(gradient)

        perturbation = perturbation + args.stepsize_adv * gradient.sign()
        perturbation = torch.clamp(perturbation, min=-args.eps/255, max=args.eps/255)
        adv_videos = torch.clamp(unnorm_video + perturbation, min=0, max=1)
        perturbation = adv_videos - unnorm_video
        adv_videos = transform_video(adv_videos.clone().detach().permute(1,0,2,3), mode='forward').permute(1,0,2,3)
        del adv_embedding, gradient, loss, loss_1, loss_2, loss_3
        torch.cuda.empty_cache()
    return adv_videos

def tifgsm_attack(video, ori_embedding):
    sclae_step = 5
    def _initial_kernel(kernlen, nsig):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def _conv2d_frame(grads):
        '''
        grads: N, C, T, H, W
        '''
        # generate start_kernel
        grads = grads.unsqueeze(0)
        grads = grads.permute(0,2,1,3,4).to(dtype=torch.float16)
        kernel = _initial_kernel(15, 3).astype(np.float32) # (15,15)
        stack_kernel = np.stack([kernel, kernel, kernel]) # (3,15,15)
        stack_kernel = torch.from_numpy(np.expand_dims(stack_kernel, 1)).to(model.device, dtype=torch.float16) # 3,1,15,15
    
        frames = grads.shape[2]
        out_grads = torch.zeros_like(grads)
        for i in range(frames):
            this_grads = grads[:,:,i]
            out_grad = nn.functional.conv2d(this_grads, stack_kernel, groups=3, stride=1, padding=7)
            out_grads[:,:,i] = out_grad
        out_grads = out_grads / torch.mean(torch.abs(out_grads), [1,2,3], True)
        out_grads = out_grads.permute(0,2,1,3,4).squeeze(0)
        del grads, kernel, stack_kernel
        return out_grads
    video = video.clone().detach()
    unnorm_video = transform_video(video.permute(1,0,2,3), mode='back').permute(1,0,2,3)
    perturbation = torch.zeros_like(video).uniform_(-args.eps/255, args.eps/255)
    perturbation.requires_grad = True
    for i in range(args.iterations_adv):
        adv_embedding = vision_tower(video+perturbation)
        loss_1 = loss_fn(adv_embedding, ori_embedding.detach())
        loss_2 = 0.0
        if args.bicos:
            loss_2 = criterion(adv_embedding, ori_embedding.detach(), args.bicos, single=args.single)
        if args.flow:
            cos_sim = F.cosine_similarity(adv_embedding[:,:,1:], adv_embedding[:,:,:-1])
            loss_3 = args.flow * (1-cos_sim).mean()
            del cos_sim
        else:
            loss_3 =0.0
        loss = loss_1+loss_2+loss_3
        start_time = time.time()
        gradient = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
        end_time = time.time()
        gradient = _conv2d_frame(gradient)
                
        if i % 5 ==0:
            logger.info(f"Feature space {args.loss} {args.attack_method}: [data_idx: {data_idx}][{i}/{args.iterations_adv}] loss={loss.item()} l1 loss={loss_1.item() if isinstance(loss_1, torch.Tensor) else loss_1} bicon loss={loss_2.item() if isinstance(loss_2, torch.Tensor) else loss_2} flow loss={loss_3.item() if isinstance(loss_3, torch.Tensor) else loss_3}")
        gradient = norm_grad(gradient)

        perturbation = perturbation + args.stepsize_adv * gradient.sign()
        perturbation = torch.clamp(perturbation, min=-args.eps/255, max=args.eps/255)
        adv_videos = torch.clamp(unnorm_video + perturbation, min=0, max=1)
        perturbation = adv_videos - unnorm_video
        adv_videos = transform_video(adv_videos.clone().detach().permute(1,0,2,3), mode='forward').permute(1,0,2,3)
        del adv_embedding, gradient, loss, loss_1, loss_2, loss_3
        torch.cuda.empty_cache()
    return adv_videos
    
def sim_attack(video, ori_embedding):
    sclae_step = 5
    def _multi_scale(model, adv_videos, labels, perturbation):    
        def obtain_grad(vid, labels, ith):
            outputs = vision_tower(vid)
            loss = loss_fn(outputs, ori_embedding.detach())
            grad = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
            if ith ==0:
                logger.info(f"Feature space {args.loss} {args.attack_method}: [data idx: {data_idx}] [{ith}/5] loss={loss.item()}")
            del outputs, loss
            return grad
        
        mean_grad = None
        for ith in range(sclae_step):
            tmp_videos = 1 / 2**i * adv_videos
            grad = obtain_grad(tmp_videos, labels, ith)
            if mean_grad is None:
                mean_grad = grad
            else:
                mean_grad += grad.clone()
        return mean_grad / sclae_step
    video = video.clone().detach()
    unnorm_video = transform_video(video.permute(1,0,2,3), mode='back').permute(1,0,2,3)
    perturbation = torch.zeros_like(video).uniform_(-args.eps/255, args.eps/255)
    perturbation.requires_grad = True
    for i in range(args.iterations_adv):
        gradient = _multi_scale(model, video+perturbation, ori_embedding, perturbation)
                
        gradient = norm_grad(gradient)

        perturbation = perturbation + args.stepsize_adv * gradient.sign()
        perturbation = torch.clamp(perturbation, min=-args.eps/255, max=args.eps/255)
        adv_videos = torch.clamp(unnorm_video + perturbation, min=0, max=1)
        perturbation = adv_videos - unnorm_video
        adv_videos = transform_video(adv_videos.clone().detach().permute(1,0,2,3), mode='forward').permute(1,0,2,3)
        del gradient
        torch.cuda.empty_cache()

    return adv_videos

def bsr_transform_video(x, n_blocks=2, max_angle=24):
    """
    Apply BSR to each frame of input video x: [B, T, C, H, W]
    Return transformed x with same shape
    """
    x = x.unsqueeze(0)
    B, C, T, H, W = x.shape
    x = x.view(B * T, C, H, W)  # Flatten temporal dimension

    bh, bw = H // n_blocks, W // n_blocks
    blocks = x.view(B*T, C, n_blocks, bh, n_blocks, bw)
    blocks = blocks.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B*T, nb, nb, C, bh, bw]
    blocks = blocks.view(B*T, n_blocks * n_blocks, C, bh, bw)  # Flatten blocks

    perm = torch.randperm(n_blocks * n_blocks, device=x.device)
    blocks = blocks[:, perm]

    angles = (torch.rand(B*T, n_blocks * n_blocks, device=x.device) * 2 - 1) * max_angle

    rotated_blocks = []
    for i in range(B*T):
        frame_blocks = []
        for j in range(n_blocks * n_blocks):
            angle = angles[i, j].item()
            img = TF.rotate(blocks[i, j].to(torch.float16), angle, interpolation=TF.InterpolationMode.BILINEAR)
            frame_blocks.append(img)
        rotated_blocks.append(torch.stack(frame_blocks))

    rotated_blocks = torch.stack(rotated_blocks)  # [B*T, nb², C, bh, bw]
    rotated_blocks = rotated_blocks.view(B*T, n_blocks, n_blocks, C, bh, bw)
    rotated_blocks = rotated_blocks.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_trans = rotated_blocks.view(B*T, C, H, W)
    del rotated_blocks, blocks, angles, frame_blocks
    return x_trans.view(C, T, H, W)

def bsr_attack(video, ori_embedding):
    video = video.clone().detach()
    unnorm_video = transform_video(video.permute(1,0,2,3), mode='back').permute(1,0,2,3)
    perturbation = torch.zeros_like(video).uniform_(-args.eps/255, args.eps/255)
    perturbation.requires_grad = True
    momentum = torch.zeros_like(video).to(model.device)

    for i in range(args.iterations_adv):
        n_trans = 3
        grad_sum = torch.zeros_like(video)
        for _ in range(n_trans):
            x_t = bsr_transform_video(video+perturbation)
            embed_t = vision_tower(x_t)

            loss = loss_fn(embed_t, ori_embedding.detach())
            grad = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
            grad_sum += grad.clone()

        gradient = grad_sum / n_trans  
                
        if i%5==0:
            logger.info(f"Feature space {args.loss} {args.attack_method}: [data_idx: {data_idx}][{i}/{args.iterations_adv}] loss={loss.item()}")
        gradient = norm_grad(gradient)
        momentum = args.decay * momentum + gradient

        perturbation = perturbation + args.stepsize_adv * momentum.sign()
        perturbation = torch.clamp(perturbation, min=-args.eps/255, max=args.eps/255)
        adv_videos = torch.clamp(unnorm_video + perturbation, min=0, max=1)
        perturbation = adv_videos - unnorm_video
        adv_videos = transform_video(adv_videos.clone().detach().permute(1,0,2,3), mode='forward').permute(1,0,2,3)
        del gradient, loss, x_t, embed_t
        torch.cuda.empty_cache()

    return adv_videos
logger = setup_logger("feature_space_attack_logger", log_dir=f"./logs/mllm_attack/{args.attack_method}_bicon{args.bicos}_flow{args.flow}")
args = parse_args()
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length, vision_tower = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
# model.eval()
max_frames_num =8
data_idx=0
if args.benchmark == 'mvbench':
    logger.info("Running mvbench benchmark")
    data_list = {
        "Action Sequence": ("action_sequence.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/star/Charades_v1_480/", "decord", True), # has start & end
        "Action Prediction": ("action_prediction.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/star/Charades_v1_480/", "decord", True), # has start & end
        "Action Antonym": ("action_antonym.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/ssv2_video/", "decord", False),
        "Fine-grained Action": ("fine_grained_action.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/Moments_in_Time_Raw/videos/", "decord", False),
        "Unexpected Action": ("unexpected_action.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/FunQA_test/test/", "decord", False),
        "Object Existence": ("object_existence.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/clevrer/video_validation/", "decord", False),
        
        "Object Interaction": ("object_interaction.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/star/Charades_v1_480/", "decord", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/perception/videos/", "decord", False),
        "Moving Direction": ("moving_direction.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/clevrer/video_validation/", "decord", False),
        "Action Localization": ("action_localization.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/sta/sta_video/", "decord", True),  # has start & end
        "Scene Transition": ("scene_transition.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/scene_qa/video/", "decord", False),
        "Action Count": ("action_count.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/perception/videos/", "decord", False),
        
        "Moving Count": ("moving_count.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/clevrer/video_validation/", "decord", False),
        "Moving Attribute": ("moving_attribute.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/clevrer/video_validation/", "decord", False),
        "State Change": ("state_change.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/perception/videos/", "decord", False),
        "Fine-grained Pose": ("fine_grained_pose.json", "/projects/hui007/MLLM/benchmarks/mvbench/NTURGBD120/nturgb+d_rgb/", "decord", False),
        "Character Order": ("character_order.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/perception/videos/", "decord", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/vlnqa/", "decord", False),
        "Episodic Reasoning": ("episodic_reasoning.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", "/projects/hui007/MLLM/benchmarks/mvbench/video/clevrer/video_validation/", "decord", False),
    }

    data_dir = "/projects/hui007/MLLM/benchmarks/mvbench/json"

    all_data = []
    for k, v in data_list.items():
        with open(os.path.join(data_dir, v[0]), 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            all_data.append({
                'task_type': k,
                'prefix': v[1],
                'data_type': v[2],
                'bound': v[3],
                'data': data
            })

    for item in all_data:
        # video_path = os.path.join(item['prefix'], item['data']['video'])

        if item['task_type'] == 'Fine-grained Pose':
            if int(item['data']['video'][1:4]) > 17:
                video_path = os.path.join(f"/home/Dataset/Action/NTURGBD120/nturgb+d_rgb_s{item['data']['video'][1:4]}/nturgb+d_rgb/", item['data']['video'])
            else:
                video_path = os.path.join(item['prefix'], item['data']['video'])
        else:
            video_path = os.path.join(item['prefix'], item['data']['video'])

        try:
            if item['bound']:
                if 'accurate_start' in item['data'] and 'accurate_end' in item['data']:
                    video = load_video_v2(video_path, max_frames_num, force_sample=True, start=item['data']['accurate_start'], end=item['data']['accurate_end'])
                else:
                    video = load_video_v2(video_path, max_frames_num, force_sample=True, start=item['data']['start'], end=item['data']['end'])
            else:
                video = load_video_v2(video_path, max_frames_num, force_sample=True)

            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.bfloat16) # torch.Size([64, 3, 384, 384])
            with torch.no_grad():
                image_emb = vision_tower(video) # torch.Size([64, 729, 1152])
            if args.attack_method == 'pgd':
                adv_videos = pgd_attack(video, image_emb)
                data_idx = data_idx + 1
            elif args.attack_method == 'mifgsm':
                adv_videos = mifgsm_attack(video, image_emb)
            elif args.attack_method == 'difgsm':
                adv_videos = difgsm_attack(video, image_emb)
            elif args.attack_method == 'tifgsm':
                adv_videos = tifgsm_attack(video, image_emb)
            elif args.attack_method == 'sim':
                adv_videos = sim_attack(video, image_emb)
            elif args.attack_method == 'bsr':
                adv_videos = bsr_attack(video, image_emb)
            del image_emb, video
            save_dir = os.path.join(args.output_dir, f"adv_{args.loss}_{args.eps}_{args.iterations_adv}_{args.attack_method}_bicon{args.bicos}{args.single}_flow{args.flow}_llava-next_{args.benchmark}/task_{item['task_type']}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            item['adv_video'] = adv_videos
            torch.save(item, os.path.join(save_dir, f"adv_video_{data_idx}.pt"))
            del adv_videos
            torch.cuda.empty_cache()
            data_idx = data_idx + 1
 
        except Exception as e:
            print(e)
        # adv_videos = bsr_attack(video, image_emb)
        # save_dir = os.path.join(args.output_dir, f"adv_{args.loss}_{args.eps}_{args.iterations_adv}_{args.attack_method}_bicon{args.bicos}{args.single}_flow{args.flow}_video-chat2_{args.benchmark}/task_{item['task_type']}")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir, exist_ok=True)
        # item['adv_video'] = adv_videos
        # torch.save(item, os.path.join(save_dir, f"adv_video_{data_idx}.pt"))
        # del image_emb, adv_videos, video
        # torch.cuda.empty_cache()
        # data_idx = data_idx + 1
        

else:
    logger.info("Running seedbench benchmark")
    with open("/projects/hui007/MLLM/Video-LLaVA/attack_result/seedbench/samples.json", "r") as f:
        data_dict = json.load(f)

    data_list = data_dict['10']     #+ data_dict['11'] + data_dict['12']
    
    for item in data_list:
        try:
            video_path = item['video_path']
            if "segment" in item:
                if item['question_type_id'] == 11:
                    video = load_video_v2(video_path, max_frames_num, force_sample=True, start=item['segment'][0], end=item['segment'][1])
                else:
                    start, end = item['segment'][0] / 15, item['segment'][1] / 15
                    video = load_video_v2(video_path, max_frames_num, force_sample=True, start=start, end=end)       
            else:
                video = load_video_v2(video_path, max_frames_num, force_sample=True)

            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.bfloat16) # torch.Size([64, 3, 384, 384])
        
        except Exception as e:
            print(e)
        with torch.no_grad():
            image_emb = vision_tower(video) # torch.Size([64, 729, 1152])
        if args.attack_method == 'pgd':
            adv_videos = pgd_attack(video, image_emb)
        elif args.attack_method == 'mifgsm':
            adv_videos = mifgsm_attack(video, image_emb)
        elif args.attack_method == 'difgsm':
            adv_videos = difgsm_attack(video, image_emb)
        elif args.attack_method == 'tifgsm':
            adv_videos = tifgsm_attack(video, image_emb)
        elif args.attack_method == 'sim':
            adv_videos = sim_attack(video, image_emb)
        elif args.attack_method == 'bsr':
            adv_videos = bsr_attack(video, image_emb)
        del image_emb, video
        save_dir = os.path.join(args.output_dir, f"adv_{args.loss}_{args.eps}_{args.iterations_adv}_{args.attack_method}_temp{args.temp}_bicon{args.bicos}{args.single}_flow{args.flow}_llava-next_{args.benchmark}/task_{item['question_type_id']}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        item['adv_video'] = adv_videos
        torch.save(item, os.path.join(save_dir, f"adv_video_{data_idx}.pt"))
        del adv_videos
        torch.cuda.empty_cache()
        data_idx = data_idx + 1
    logger.info(f"save path: {save_dir}")
    

