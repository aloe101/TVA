import os
import copy
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

from opentad.utils import create_folder
from opentad.utils.misc import AverageMeter, reduce_loss
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset
import time
# from thop import profile

class ComputeLossWrapper:
    def __init__(self, embedding_orig, reduction='mean', loss=None,):
        self.embedding_orig = embedding_orig
        self.reduction = reduction # mean
        self.loss_str = loss # l2

    def __call__(self, embedding):
        return compute_loss(
            loss_str=self.loss_str, embedding=embedding, #targets=targets,
            embedding_orig=self.embedding_orig, 
            reduction=self.reduction
            )
    
def compute_loss(loss_str, embedding, embedding_orig, 
                 reduction='mean'):
    if loss_str == 'l2':
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'cos':
        loss = cossim(out=embedding, targets=embedding_orig, reduction=reduction)
    return loss

def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch

def cossim(out, targets, reduction='none'):
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    if reduction == 'mean':
        loss = F.cosine_similarity(out, targets).mean()
    return 1-loss



class BiContrastiveLoss(nn.Module):
    def __init__(self, initial_temp=1, final_temp=0.07, total_batches=5):
        super(BiContrastiveLoss, self).__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self.decay_rate = self.calculate_decay_rate()
 
    def forward(self, clean_feat, adv_feat, bicos, single=False, window_size=1):
        B, D, T = clean_feat.shape
        if bicos == 1:
            clean_feat = clean_feat.reshape(B*T, D)
            adv_feat = adv_feat.reshape(B*T, D)
            clean_feat = F.normalize(clean_feat, p=2, dim=1)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)
    
            cos_sim = torch.matmul(clean_feat, adv_feat.t()) / self.current_temp
            print(f"self.current_temp: {self.current_temp}")
    
            labels = torch.eye(cos_sim.size(0), device=cos_sim.device)

            if single:
                loss_clean2adv = 0.0
            else:
                loss_clean2adv = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
            loss_adv_2clean = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()
    
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
        
    
    def update_temperature(self, batch_count):
        self.current_temp = self.initial_temp * np.exp(-self.decay_rate * batch_count)

    def calculate_decay_rate(self):
        return -np.log(self.final_temp / self.initial_temp) / self.total_batches



def taai_mifgsm_one_epoch(
        model,
    dataloader,
    cfg,
    args,
    logger,
    model_ema=None,
):
    num_iters = len(dataloader)       #100
    losses_tracker = {}
    model.eval()      
    adv_batches = []
    if args.loss == 'l2':
        loss_fn = nn.MSELoss(reduction='mean')
    elif args.loss == 'l1':
        loss_fn = nn.L1Loss(reduction='mean')
    criterion = BiContrastiveLoss()  

    for iter_idx, data_dict in enumerate(dataloader):
        perturbation=torch.zeros_like(data_dict['inputs'], dtype=torch.float32).uniform_(-args.eps, args.eps).requires_grad_(True)
        momentum = torch.zeros_like(perturbation)
        with torch.no_grad():
            ori_embedding = model(
            data_dict['inputs'],
            data_dict['masks']
        )

        
        for i in range(args.iterations_adv):
            
            adv_embedding = model(
                data_dict['inputs'] + perturbation,
                data_dict['masks'],
            )
            
            
            loss_1 = loss_fn(adv_embedding, ori_embedding.detach())
            loss_2 = 0.0
            if args.bicos:
                loss_2 = criterion(adv_embedding, ori_embedding.detach(), args.bicos)
            
            if args.flow == 1:
                cos_sim = F.cosine_similarity(adv_embedding[:,:,1:], adv_embedding[:,:,:-1])
                loss_3 = (1-cos_sim).mean()
            
            else:
                loss_3 = 0.0
            

            if perturbation.grad is not None:
                perturbation.grad.zero_()
            perturbation.retain_grad()
            
            start_time = time.time()
            loss.backward(retain_graph=True)
            end_time = time.time()
            
            print(f"iteration {iter_idx}, Attack backward time: {end_time - start_time}")
            
            gradient = perturbation.grad
                          
            
            # gradient = torch.autograd.grad(losses['cost'], perturbation)[0]
            if gradient.isnan().any():
                print(f'attention: nan in gradient ({gradient.isnan().sum()})')
                gradient[gradient.isnan()] = 0.
                # normalize
            gradient = norm_grad(gradient)
            momentum = args.decay * momentum + gradient
            perturbation = perturbation + args.stepsize_adv * momentum.sign()
            # project
            perturbation = torch.clamp(perturbation, -args.eps, args.eps)
            perturbation = torch.clamp(
                    data_dict['inputs'] + perturbation, 0, 255
                ) - data_dict['inputs']
            # clamp to image space
            assert not perturbation.isnan().any()
            assert torch.max(data_dict['inputs'] + perturbation) < 256. + 1e-6 and torch.min(
                    data_dict['inputs'] + perturbation
                ) > -1e-6
            logger.info(f"Feature space {args.loss} taai mifgsm_layer: [{iter_idx}/{num_iters}][{i}/{args.iterations_adv}] total loss={loss.item()} {args.loss} loss={loss_1.item()} bicos_loss={loss_2.item() if args.bicos else 0.0} flow_loss={loss_3.item() if args.flow else 0.0}")
            del adv_embedding, gradient, loss, loss_1, loss_2, loss_3
            torch.cuda.empty_cache()
            criterion.update_temperature(iter_idx)
            
        adv_v = data_dict['inputs'] + perturbation.detach()
        save_dir = os.path.join(args.output_dir, f"adv_{args.loss}_{args.eps}_{args.iterations_adv}_172_taai_tdrop{args.tdrop}_smooth{args.smooth}_flow{args.flow}_bicos{args.bicos}{args.single}_lips{args.lips}_{'anet' if not args.thumos else 'thumos'}")   #{args.bicos}
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for idx,item in enumerate(data_dict['metas']):
            adv = adv_v[idx].detach().cpu().numpy()
            if args.thumos:
                np.save(os.path.join(save_dir, f"adv_{item['video_name']}_{item['window_start_frame']}"), adv)
            else:
                np.save(os.path.join(save_dir, f"adv_{item['video_name']}_{item['resize_length']}"), adv)
        del ori_embedding, adv_v
        torch.cuda.empty_cache()


def norm_grad(grad):
    norm = torch.mean(torch.abs(grad), tuple(range(1, grad.ndim)), keepdim=True)
    grad = grad / (norm + 1e-8)
    return grad
