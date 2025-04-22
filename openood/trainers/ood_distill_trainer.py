import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing

import timm

from ..networks.kd_model import DT
from timm.models.vision_transformer import VisionTransformer  
from openood.losses.mi_loss import CLUB



class OODDistillTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, train_loader2: DataLoader,
                 config: Config, fea_dist=True, logits_distill=True, ood_fea_distill=True, dataset='cifar100') -> None:

        self.net = net
        self.train_loader = train_loader
        self.train_loader2 = train_loader2
        self.config = config
        self.logits_distill = logits_distill
        self.fea_distill = fea_dist
        self.ood_fea_distill = ood_fea_distill
        self.dataset = dataset
        self.ood_sample_num = 128 # number of outlier data used for training in k-nn sampled outliers and diffusion generated outliers
        
        if self.dataset == 'cifar10':
            self.num_classes = 10
            self.ood_samples = torch.from_numpy(np.load('../../cifar10_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()   # (10, 1000, 768)
            self.ood_samples = self.ood_samples.reshape(self.ood_samples.shape[0]*self.ood_samples.shape[1], self.ood_samples.shape[2])
            self.num_knn_ood = self.ood_samples.shape[0]*self.ood_samples.shape[1]
            self.mi_loss = CLUB(768, 512, 512).cuda()
            
        elif self.dataset == 'cifar100':
            self.num_classes = 100
            if self.ood_fea_distill:
                self.ood_samples = torch.from_numpy(np.load('../../cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()  # (10000, 768)
                self.ood_samples = self.ood_samples.reshape(self.ood_samples.shape[0]*self.ood_samples.shape[1], self.ood_samples.shape[2])
                self.num_knn_ood = self.ood_samples.shape[0]*self.ood_samples.shape[1]
                self.mi_loss = CLUB(768, 512, 512).cuda()
        else:
            self.num_classes = 1000
            if self.ood_fea_distill:
                self.ood_samples = torch.from_numpy(np.load('../../in1k_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()  # (1000, 2500, 768)
                self.ood_samples = self.ood_samples.reshape(1000 * 2500, self.ood_samples.shape[2])
                self.num_knn_ood = self.ood_samples.shape[0]*self.ood_samples.shape[1]
                self.mi_loss = CLUB(768, 512, 512).cuda()
            
        if fea_dist:
            self.dt = DT(in_dim=768, out_dim=512).cuda()
        
        if self.dataset == 'cifar10':
            self.model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
            self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)
            self.model.load_state_dict(
                torch.load('../results/pytorch_model.bin')
            )
        else:
            self.model = VisionTransformer(num_classes=self.num_classes)
            self.model.load_state_dict(
                torch.load('../results/cifar100_vit_pretrained_finetune_trainer_e50_lr0.0001_default_cifar100_finetune_final_2/s0/best.ckpt')
            )
            
        
        self.kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
        self.optimizer = torch.optim.SGD(
            [
                {
                    'params': net.parameters()
                },
                {
                    'params': self.dt.parameters(), 'lr': 0.01
                },
                {
                    'params': self.mi_loss.p_mu.parameters(), 'lr': 0.0001
                },
                {
                    'params': self.mi_loss.p_logvar.parameters(), 'lr': 0.0001
                }
            ],
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        self.mid_fea_kd_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)

    def train_epoch(self, epoch_idx):
        if self.fea_distill:
            self.dt.train()
        self.net.train()
        self.model.cuda()
        self.model.eval()
        
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            
            if train_step == len(train_dataiter):
                continue
            
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier, feature = self.net(data, return_feature=True)
            log_soft = F.log_softmax(logits_classifier, dim=1)
            with torch.no_grad():
                vit_cls, mid_fea = self.model(data, return_feature=True)
                vit_cls = F.softmax(vit_cls, dim=1)
            
            loss = F.cross_entropy(logits_classifier, target)
            
            if self.logits_distill:
                loss_kl = self.kl_loss(log_soft, vit_cls)
                loss += 4*loss_kl
                
            if self.fea_distill:
                kd_fea = self.dt(mid_fea)
                fea_log_soft = F.log_softmax(feature, dim=1)
                kd_fea = F.softmax(kd_fea, dim=1)
                mid_fea_kd = self.mid_fea_kd_loss(fea_log_soft, kd_fea)
                loss += 8*mid_fea_kd
            
            if self.ood_fea_distill:
                idx = self.ood_sub_samples = torch.randperm(self.num_knn_ood)[:self.ood_sample_num]
                selected_ood_samples = self.ood_samples[idx]  # random selected samples
                cdl = self.mi_loss.forward(selected_ood_samples, feature)
                loss += 0.1*cdl
                
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            
            torch.cuda.empty_cache()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
