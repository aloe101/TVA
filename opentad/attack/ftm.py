import os
import copy
import json
import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import random
import math

from opentad.utils import create_folder
from opentad.utils.misc import AverageMeter, reduce_loss
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset
from opentad.attack.ftm_config import exp_configuration
import time
# from thop import profile

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
        B, D, T = clean_feat.shape
        if bicos == 1:
            clean_feat = clean_feat.reshape(B*T, D)
            adv_feat = adv_feat.reshape(B*T, D)
            clean_feat = F.normalize(clean_feat, p=2, dim=1)
            adv_feat = F.normalize(adv_feat, p=2, dim=1)
    
            cos_sim = torch.matmul(clean_feat, adv_feat.t()) / self.current_temp
            print(f"self.current_temp: {self.current_temp}")
    
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

class FeatureTuningWrapper(nn.Module):
    def __init__(self, model, input_size, exp_settings, device):
        super().__init__()
        self.model = model
        self.device = device
        self.ftm = FeatureTuning(model, input_size, exp_settings, device)

    def start_record(self, x):
        self.ftm.start_feature_record()
        with torch.no_grad():
            _ = self.ftm(x)
        self.ftm.end_feature_record()

    def forward(self, x):
        return self.ftm(x)

    def update_ftm_grad(self, total_loss):
        # 收集所有FTM参数
        all_params = []
        all_active_layers = []

        for layer_idx, was_triggered in self.ftm.mixing_triggered.items():
            if was_triggered:
                all_params.append(self.ftm.outputs_tuning[layer_idx])
                all_active_layers.append(layer_idx)

        if not all_params:
            return

        grads = torch.autograd.grad(total_loss, all_params, retain_graph=True, create_graph=False, allow_unused=True)
        for i, (layer_idx, grad) in enumerate(zip(all_active_layers, grads)):
            print(f"Layer {layer_idx}, grad is None? {grad is None}")
        for i, layer_idx in enumerate(all_active_layers):
            param = self.ftm.outputs_tuning[layer_idx]
            grad = grads[i]
            if grad is not None:
                self.ftm.outputs_tuning[layer_idx] = (param - grad).detach().requires_grad_(True)
    def remove(self):
        self.ftm.remove_hooks()

class FeatureTuning(nn.Module):
    def __init__(self, model: nn.Module, input_size, exp_settings, device):
        super().__init__()
        self.exp_settings = exp_settings
        self.device = device
        self.mixup_layer = exp_settings['mixup_layer']
        self.prob = exp_settings['mix_prob']
        self.channelwise = exp_settings['channelwise']

        self.model = model
        self.input_size = input_size
        self.record = False

        self.outputs = {}
        self.outputs_tuning = {}  # feature perturbations for tuning
        self.mixing_triggered = {}
        self.forward_hooks = []

        def get_children(model: torch.nn.Module):
            children = list(model.children())
            flattened_children = []
            if children == []:
                if self.mixup_layer == 'conv_linear_no_last' or self.mixup_layer == 'conv_linear_include_last':
                    if type(model) == torch.nn.Conv2d or type(model) == torch.nn.Linear:
                        return model
                    else:
                        return []
                elif self.mixup_layer == 'bn' or self.mixup_layer == 'relu':
                    if type(model) == torch.nn.BatchNorm2d:
                        return model
                    else:
                        return []
                else:
                    if type(model) == torch.nn.Conv2d:
                        return model
                    else:
                        return []
            else:
                for child in children:
                    try:
                        flattened_children.extend(get_children(child))
                    except TypeError:
                        flattened_children.append(get_children(child))
            return flattened_children

        mod_list = get_children(model)
        self.layer_num = len(mod_list)

        for i, m in enumerate(mod_list):
            self.forward_hooks.append(m.register_forward_hook(self.save_outputs_hook(i)))

    def save_outputs_hook(self, layer_idx) -> Callable:
        exp_settings = self.exp_settings
        mix_upper_bound_feature = exp_settings['mix_upper_bound_feature']
        mix_lower_bound_feature = exp_settings['mix_lower_bound_feature']
        shuffle_image_feature = exp_settings['shuffle_image_feature']
        blending_mode_feature = exp_settings['blending_mode_feature']
        mixed_image_type_feature = exp_settings['mixed_image_type_feature']
        divisor = exp_settings['divisor']

        def hook_fn(module, input, output):
            if type(module) == torch.nn.Linear or output.size()[-1] <= self.input_size // divisor:

                if self.mixup_layer == 'conv_linear_no_last' and (layer_idx + 1) == self.layer_num and type(module) == torch.nn.Linear:
                    pass  # exclude the last fc layer
                else:
                    if layer_idx in self.outputs and self.record == False:  # Feature mixup inference mode
                        c = torch.rand(1).item()
                        # Record selected layers for update
                        self.mixing_triggered[layer_idx] = (c <= self.prob)

                        # If selected, mix the output with clean features and feature perturbations
                        if self.mixing_triggered[layer_idx]:
                            # Configuration for mixing clean features
                            if mixed_image_type_feature == 'A':  # Mix features of other images
                                prev_feature = output.clone().detach()
                            else:  # Mix clean features
                                prev_feature = self.outputs[layer_idx].clone().detach()  # Get stored clean features

                            if shuffle_image_feature == 'SelfShuffle':  # Image-wise feature shuffling
                                idx = torch.randperm(output.shape[0])
                                prev_feature_shuffle = prev_feature[idx].view(prev_feature.size())
                                del idx
                            elif shuffle_image_feature == 'None':
                                prev_feature_shuffle = prev_feature

                            # Random mixing ratio
                            mix_ratio = mix_upper_bound_feature - mix_lower_bound_feature
                            if self.channelwise == True:
                                if output.dim() == 4:
                                    a = (torch.rand(prev_feature.shape[0],
                                                    prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], prev_feature.shape[1], 1, 1).to(self.device)
                                elif output.dim() == 3:
                                    a = (torch.rand(prev_feature.shape[0],
                                                    prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], prev_feature.shape[1], 1).to(self.device)
                                else:
                                    a = (torch.rand(prev_feature.shape[0],
                                                    prev_feature.shape[1]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], prev_feature.shape[1]).to(self.device)
                            else:
                                if output.dim() == 4:
                                    a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], 1, 1, 1).to(self.device)
                                elif output.dim() == 3:
                                    a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], 1, 1).to(self.device)
                                else:
                                    a = (torch.rand(prev_feature.shape[0]) * mix_ratio + mix_lower_bound_feature).view(
                                        prev_feature.shape[0], 1).to(self.device)

                            if self.mixup_layer == 'relu':
                                output = F.relu(output, inplace=True)

                            # mix with feature perturbations
                            output_flat = output.detach().view(output.size(0), -1)  # [B, *]
                            tuning_flat = self.outputs_tuning[layer_idx].detach().view(output.size(0), -1)  # [B, *]

                            output_norm = output_flat.norm(dim=1)  # [B]
                            tuning_norm = tuning_flat.norm(dim=1)  # [B]
                            scale = exp_settings['ftm_beta'] * output_norm / (tuning_norm + 1e-7)  # [B]

                            for _ in range(len(output.shape) - 1):
                                scale = scale.unsqueeze(-1)

                            output1 = output + self.outputs_tuning[layer_idx] * scale

                            # mix with clean features
                            if blending_mode_feature == 'M':  # Linear interpolation
                                output2 = (1 - a) * output1 + a * prev_feature_shuffle
                            elif blending_mode_feature == 'A':  # Addition
                                output2 = output1 + a * prev_feature_shuffle

                            return output2
                        # If not selected, mix the output with feature perturbations
                        else:
                            output_flat = output.detach().view(output.size(0), -1)  # [B, *]
                            tuning_flat = self.outputs_tuning[layer_idx].detach().view(output.size(0), -1)  # [B, *]

                            output_norm = output_flat.norm(dim=1)  # [B]
                            tuning_norm = tuning_flat.norm(dim=1)  # [B]
                            scale = exp_settings['ftm_beta'] * output_norm / (tuning_norm + 1e-7)  # [B]

                            for _ in range(len(output.shape) - 1):
                                scale = scale.unsqueeze(-1)

                            output_perturbed = output + self.outputs_tuning[layer_idx].detach() * scale

                            return output_perturbed

                    elif self.record == True:  # Feature recording mode
                        self.outputs[layer_idx] = output.clone().detach()
                        # Learnable feature perturbations
                        self.outputs_tuning[layer_idx] = torch.zeros_like(output).clone().detach().requires_grad_(True)
                        self.mixing_triggered[layer_idx] = False
                        return

        return hook_fn

    def start_feature_record(self):
        self.record = True

    def end_feature_record(self):
        self.record = False

    def remove_hooks(self):
        for fh in self.forward_hooks:
            fh.remove()
        del self.outputs
        del self.outputs_tuning
        del self.mixing_triggered

    def forward(self, x: Tensor) -> Tensor:
        # Clear mixing triggers at the start of each forward pass
        self.mixing_triggered = {}
        return self.model(x)

def fs_ftm_one_epoch(
    model,
    dataloader,
    cfg,
    args,
    logger,
    model_ema=None,
):
    """Inference and Evaluation the model"""
    num_iters = len(dataloader)       #100
    losses_tracker = {}
    model.eval()        
    adv_batches = []
    if args.loss == 'l2':
        loss_fn = nn.MSELoss(reduction='mean')
    elif args.loss == 'l1':
        loss_fn = nn.L1Loss(reduction='mean')
    criterion = BiContrastiveLoss(initial_temp=args.temp, final_temp=args.temp, total_batches=args.iterations_adv)  

    
    for iter_idx, data_dict in enumerate(dataloader):
        
        # perturbation=torch.zeros_like(data_dict['inputs'], dtype=torch.float32).uniform_(-args.eps, args.eps).detach().requires_grad_(True)
        # perturbation = torch.clamp(
        #         data_dict['inputs'] + perturbation, 0, 255
        #     ) - data_dict['inputs']
        perturbation=torch.zeros_like(data_dict['inputs'], dtype=torch.float32).uniform_(-args.eps, args.eps).requires_grad_(True)
        momentum = torch.zeros_like(perturbation)
        with torch.no_grad():
            ori_embedding = model(
            data_dict['inputs'],
            data_dict['masks'],
        )
        ftm_model = FeatureTuningWrapper(model, input_size=160, exp_settings=exp_configuration[1], device=model.device)  # <- 根据你实际设置input size
        ftm_model.start_record(data_dict['inputs'])  # 记录干净特征
        
        for i in range(args.iterations_adv):
            adv_embedding = model(
                data_dict['inputs'] + perturbation,
                data_dict['masks'],
            )
            loss = loss_fn(adv_embedding, ori_embedding.detach())
            loss_1 = 0.0
            loss_2 = 0.0
            loss_3 = 0.0
            if args.bicos:
                loss_2 = criterion(adv_embedding, ori_embedding.detach(), args.bicos, single=args.single)
                loss = loss+loss_2
            if args.flow == 1:
                cos_sim = F.cosine_similarity(adv_embedding[:,:,1:], adv_embedding[:,:,:-1])
                loss_3 = (1-cos_sim).mean()
                loss = loss + loss_3
                del cos_sim
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            perturbation.retain_grad()
            
            start_time = time.time()
            loss.backward(retain_graph=True)
            end_time = time.time()
            ftm_model.update_ftm_grad(loss)
            
            
            print(f"iteration {iter_idx}, Attack backward time: {end_time - start_time}")
            gradient = perturbation.grad
            
            # gradient = torch.autograd.grad(losses['cost'], perturbation)[0]
            if gradient.isnan().any():  #
                print(f'attention: nan in gradient ({gradient.isnan().sum()})')  #
                gradient[gradient.isnan()] = 0.
                # normalize
            gradient = norm_grad(gradient)
            momentum = args.decay * momentum + gradient
            perturbation = perturbation + args.stepsize_adv * momentum.sign()
            
                # project
                # perturbation = project_perturbation(perturbation, eps, norm)
            perturbation = torch.clamp(perturbation, -args.eps, args.eps)
            perturbation = torch.clamp(
                    data_dict['inputs'] + perturbation, 0, 255
                ) - data_dict['inputs']  # clamp to image space        
            # else:
            #     raise RuntimeError("Gradient is None. Ensure that the perturbation tensor is used in the computation graph.")
            assert not perturbation.isnan().any()
            assert torch.max(data_dict['inputs'] + perturbation) < 256. + 1e-6 and torch.min(
                    data_dict['inputs'] + perturbation
                ) > -1e-6
            
            logger.info(f"Feature space {args.loss} {args.attack_method}: [{iter_idx}/{num_iters}] [{i}/{args.iterations_adv}] loss={loss.item()} l1 loss={loss_1.item() if isinstance(loss_1, torch.Tensor) else loss_1} bicon loss={loss_2.item() if isinstance(loss_2, torch.Tensor) else loss_2} flow loss={loss_3.item() if isinstance(loss_3, torch.Tensor) else loss_3}")
            del adv_embedding, gradient, loss, loss_1, loss_2, loss_3
            torch.cuda.empty_cache()

                # assert (ctorch.compute_norm(perturbation, p=self.norm) <= self.eps + 1e-6).all()
        # todo return best perturbation
        # problem is that model currently does not output expanded loss
        adv_v = data_dict['inputs'] + perturbation.detach()
        ftm_model.remove()
        save_dir = os.path.join(args.output_dir, f"adv_{args.loss}_{args.eps}_{args.iterations_adv}_{args.attack_method}seed{args.seed}_bicon{args.bicos}_flow{args.flow}_174_{'anet' if not args.thumos else 'thumos'}")
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

def ensemble_mifgsm_one_epoch(
        model_1,
        model_2,
        model_3,
    dataloader,
    cfg,
    args,
    logger,
    model_ema=None,
):
    num_iters = len(dataloader)       #100
    losses_tracker = {}
    model_1.eval()
    model_2.eval()
    model_3.eval()        
    adv_batches = []

    for iter_idx, data_dict in enumerate(dataloader):
        perturbation=torch.zeros_like(data_dict['inputs'], dtype=torch.float32).uniform_(-args.eps, args.eps).requires_grad_(True)
        momentum = torch.zeros_like(perturbation)
        with torch.no_grad():
            ori_embedding_1 = model_1(
            data_dict['inputs'],
            data_dict['masks']
        )
            ori_embedding_2 = model_2(
            data_dict['inputs'],
            data_dict['masks']
        )
            ori_embedding_3 = model_3(
            data_dict['inputs'],
            data_dict['masks']
        )
            ori_embedding = torch.cat(
                [ori_embedding_1, ori_embedding_2, ori_embedding_3], dim=1)
        
        for i in range(args.iterations_adv):
            adv_inputs = data_dict['inputs'] + perturbation
            adv_inputs = torch.clamp(adv_inputs, 0, 255)
            adv_embedding_1 = model_1(
                adv_inputs,
                data_dict['masks'],
            )
            adv_embedding_2 = model_2(
                adv_inputs,
                data_dict['masks'],
            )
            adv_embedding_3 = model_3(
                adv_inputs,
                data_dict['masks'],
            )
            adv_embedding = torch.cat(
                [adv_embedding_1, adv_embedding_2, adv_embedding_3], dim=1)
            if args.loss == 'l2':
                loss_fn = nn.MSELoss(reduction='mean')
            elif args.loss == 'l1':
                loss_fn = nn.L1Loss(reduction='mean')
            loss = loss_fn(adv_embedding, ori_embedding.detach())
            # loss_2 = loss_fn(adv_embedding_2, ori_embedding_2.detach())
            # loss_3 = loss_fn(adv_embedding_3, ori_embedding_3.detach())
            
            # loss = (loss_1 + loss_2 + loss_3) / 3

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
            logger.info(f"Feature space {args.loss} mifgsm_ensemble: [{iter_idx}/{num_iters}] {args.loss} loss={loss.item()}")
            # logger.info(f"Losses: model_1={loss_1.item()}, model_2={loss_2.item()}, model_3={loss_3.item()}")
        adv_v = data_dict['inputs'] + perturbation.detach()
        save_dir = os.path.join(args.output_dir, f'adv_{args.loss}_{args.eps}_{args.iterations_adv}_174_ensemble')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for idx,item in enumerate(data_dict['metas']):
            adv = adv_v[idx].detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"adv_{item['video_name']}_{item['window_start_frame']}"), adv)

def multilayer_mifgsm_one_epoch(
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
            
            
            loss = loss_fn(adv_embedding, ori_embedding.detach())
            
            # loss = (loss_1 + loss_2 + loss_3) / 3

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
            logger.info(f"Feature space {args.loss} mifgsm_multi_layer: [{iter_idx}/{num_iters}][{i}/{args.iterations_adv}] {args.loss} loss={loss.item()}")
            # logger.info(f"Losses: layer_1={loss_1.item()}, layer_2={loss_2.item()}, layer_3={loss_3.item()}")
            
        adv_v = data_dict['inputs'] + perturbation.detach()
        save_dir = os.path.join(args.output_dir, f'adv_{args.loss}_{args.eps}_{args.iterations_adv}_174_multilayer')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for idx,item in enumerate(data_dict['metas']):
            adv = adv_v[idx].detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"adv_{item['video_name']}_{item['window_start_frame']}"), adv)
        del ori_embedding, adv_v, gradient
        torch.cuda.empty_cache()


def norm_grad(grad):
    norm = torch.mean(torch.abs(grad), tuple(range(1, grad.ndim)), keepdim=True)
    grad = grad / (norm + 1e-8)
    return grad

