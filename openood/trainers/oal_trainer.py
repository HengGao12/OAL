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
from openood.losses.deep_mi_loss import DEEP_CLUB


def unwrap_module(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    else:
        return model

def clone_parameters(parameters):
    return [param.clone().detach() for param in parameters]

class OALTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, train_loader2: DataLoader,
                 config: Config, fea_dist=True, logits_distill=True, ood_fea_distill=True, deep_ood_distill=True, dataset='cifar100') -> None:

        self.net = net
        self.train_loader = train_loader
        self.train_loader2 = train_loader2
        self.config = config
        self.logits_distill = logits_distill
        self.fea_distill = fea_dist
        self.ood_fea_distill = ood_fea_distill
        self.dataset = dataset
        self.deep_ood_distill = deep_ood_distill
        self.arch = 'res18'  # define the backbone: res18, res50
        self.ood_sample_num = 128 # number of outlier data used for training in k-nn sampled outliers and diffusion generated outliers
        self.diffusion_ood_num = 300 # number of Diffusion generated outliers
        self.energy_rl = False  # ablation: using energy loss for regularizing the model
        
        # Load the generated outliers, including both k-NN sampled and Diffusion generated
        if self.dataset == 'cifar10':
            self.num_classes = 10
            if self.ood_fea_distill: # k-NN sampled outliers
                self.ood_samples = torch.from_numpy(np.load('./outlier-data/cifar10_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()  # (10, 1000, 768)
                self.num_knn_ood = self.ood_samples.shape[0]*self.ood_samples.shape[1]
                self.ood_samples = self.ood_samples.reshape(self.ood_samples.shape[0]*self.ood_samples.shape[1], self.ood_samples.shape[2])
                if self.arch == 'res50':
                    self.mi_loss = CLUB(768, 2048, 512).cuda()
                else:
                    self.mi_loss = CLUB(768, 512, 512).cuda()              
            if self.deep_ood_distill: # Diffusion generated OOD data, we run 3 experiments and take mean \pm std
                # self.deep_ood_samples = torch.load('./outlier-data/deep_ood_embedding_cifar10_plus.pt')  # (100, 3, 64, 64) (iter_num, batch_size, height, width)
                # self.deep_ood_samples = torch.load('./outlier-data/deep_ood_embedding_cifar10_plus-1.pt')  # (100, 3, 64, 64)
                self.deep_ood_samples = torch.load('./outlier-data/deep_ood_embedding_cifar10_plus-2.pt')  # (100, 3, 64, 64)
                n, b, h, w= self.deep_ood_samples.shape

                self.deep_ood_samples = self.deep_ood_samples.reshape(n*b, 64, 64)
                self.deep_ood_samples = torch.mean(self.deep_ood_samples, dim=1).squeeze(1)

                self.mi_loss2 = DEEP_CLUB(64, 512, 512).cuda()  
                              
        else:
            self.num_classes = 100
            if self.ood_fea_distill:
                self.ood_samples = torch.from_numpy(np.load('./outlier-data/cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()
                self.num_knn_ood = self.ood_samples.shape[0]*self.ood_samples.shape[1]
                self.ood_samples = self.ood_samples.reshape(self.ood_samples.shape[0] * self.ood_samples.shape[1], self.ood_samples.shape[2])
                if self.arch == 'res50':  # Used only in Ablation Study
                    self.mi_loss = CLUB(768, 2048, 512).cuda()
                else:
                    self.mi_loss = CLUB(768, 512, 512).cuda()
                
            if self.deep_ood_distill:
                self.deep_ood_samples = torch.load('./outlier-data/deep_ood_embedding_cifar100_plus.pt').cuda()    # new outliers: [100, 3, 64, 64] 100 iter, 3 batch, 64 width, 64 height
                # self.deep_ood_samples = torch.load('./outlier-data/deep_ood_embedding_cifar100_plus-1.pt').cuda()
                # self.deep_ood_samples = torch.load('./outlier-data/deep_ood_embedding_cifar100_plus-2.pt').cuda()
                n, b, h, w = self.deep_ood_samples.shape
                
                self.deep_ood_samples = self.deep_ood_samples.reshape(n*b, 64, 64)
                self.deep_ood_samples = torch.mean(self.deep_ood_samples, dim=1).squeeze(1)
                if self.arch == 'res50': # Used only in Ablation Study
                    self.mi_loss2 = DEEP_CLUB(64, 2048, 512).cuda()
                else:
                    self.mi_loss2 = DEEP_CLUB(64, 512, 512).cuda()
            
            if self.energy_rl:
                # load synthesized OOD embeddings
                self.ood_samples = torch.from_numpy(np.load('./outlier-data/cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()
                self.num_knn_ood = self.ood_samples.shape[0]*self.ood_samples.shape[1]
                self.ood_samples = self.ood_samples.reshape(self.ood_samples.shape[0] * self.ood_samples.shape[1], self.ood_samples.shape[2])
                self.dt1=DT(in_dim=768, out_dim=512).cuda()


                self.deep_ood_samples = torch.load('./outlier-data/deep_ood_embedding_cifar100_plus.pt').cuda()    # new outliers: [100, 3, 64, 64] 100 iter, 3 batch, 64 width, 64 height
                n, b, h, w = self.deep_ood_samples.shape
                self.deep_ood_samples = self.deep_ood_samples.reshape(n*b, 64, 64)
                self.deep_ood_samples = torch.mean(self.deep_ood_samples, dim=1).squeeze(1)
                self.dt2=DT(in_dim=64, out_dim=512).cuda()
                
                self.logistic_regression1 = torch.nn.Sequential(
                    torch.nn.Linear(1, 2)
                ).cuda()
                self.logistic_regression2 = torch.nn.Sequential(
                    torch.nn.Linear(1, 2)
                ).cuda()
                
        if fea_dist:
            if self.arch == 'res50':
                self.dt=DT(in_dim=768, out_dim=2048).cuda()
            else:
                self.dt = DT(in_dim=768, out_dim=512).cuda()

        # Load Pre-trained teacher model
        if self.dataset == 'cifar10':
            self.model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
            self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)
            self.model.load_state_dict(
                torch.load('./results/pytorch_model.bin')
            )
        else:
            self.model = VisionTransformer(num_classes=self.num_classes)
            self.model.load_state_dict(
                torch.load('./results/cifar100_vit_pretrained_finetune_trainer_e50_lr0.0001_default_cifar100_finetune_final_2/s0/best.ckpt')
            )
        
        # Define the logits distillation loss
        if self.logits_distill:
            self.kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
        
        # Define the corresponding optimizers
        if self.fea_distill:
            if self.deep_ood_distill:
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
                        },
                        {
                            'params': self.mi_loss2.p_mu.parameters(), 'lr': 0.0001 
                        },
                        {
                            'params': self.mi_loss2.p_logvar.parameters(), 'lr': 0.0001
                        }
                    ],
                    config.optimizer.lr,
                    momentum=config.optimizer.momentum,
                    weight_decay=config.optimizer.weight_decay,
                    nesterov=True,
                )
            else:
                if self.ood_fea_distill:
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
                else:
                    self.optimizer = torch.optim.SGD(
                        [
                            {
                                'params': net.parameters()
                            },
                            {
                                'params': self.dt.parameters(), 'lr': 0.01
                            }
                        ],
                        config.optimizer.lr,
                        momentum=config.optimizer.momentum,
                        weight_decay=config.optimizer.weight_decay,
                        nesterov=True,
                    )                                   
        else: 
            if self.energy_rl:
                self.optimizer = torch.optim.SGD(
                    [
                        {
                            'params': net.parameters()
                        },
                        {
                            'params': self.dt1.parameters(), 'lr': 0.01
                        },
                        {
                            'params': self.logistic_regression1.parameters(), 'lr': 0.01
                        },
                        {
                            'params': self.dt2.parameters(), 'lr': 0.01
                        },
                        {
                            'params': self.logistic_regression2.parameters(), 'lr': 0.01
                        }
                        # {
                        #     'params': self.mi_loss.p_mu.parameters(), 'lr': 0.0001
                        # },
                        # {
                        #     'params': self.mi_loss.p_logvar.parameters(), 'lr': 0.0001
                        # },
                        # {
                        #     'params': self.mi_loss2.p_mu.parameters(), 'lr': 0.0001  # original:0.0001
                        # },
                        # {
                        #     'params': self.mi_loss2.p_logvar.parameters(), 'lr': 0.0001  # original:0.0001
                        # }
                    ],
                    config.optimizer.lr,
                    momentum=config.optimizer.momentum,
                    weight_decay=config.optimizer.weight_decay,
                    nesterov=True,
                )
            else:
                self.optimizer = torch.optim.SGD(
                    [
                        {
                            'params': net.parameters()
                        },
                        {
                            'params': self.mi_loss.p_mu.parameters(), 'lr': 0.0001
                        },
                        {
                            'params': self.mi_loss.p_logvar.parameters(), 'lr': 0.0001
                        },
                        {
                            'params': self.mi_loss2.p_mu.parameters(), 'lr': 0.0001 
                        },
                        {
                            'params': self.mi_loss2.p_logvar.parameters(), 'lr': 0.0001
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
                1e-8 / config.optimizer.lr,
            ),
        )
        
        # Feature distillation loss
        self.mid_fea_kd_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)

    def train_epoch(self, epoch_idx):
        if self.fea_distill:
            self.dt.train()
        self.net.train()
        self.model.cuda()
        self.model.eval()
        
        self.ood_sample_num = 128
        
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
                self.ood_sample_num = 80
            
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier, feature = self.net(data, return_feature=True)
            log_soft = F.log_softmax(logits_classifier, dim=1)

            with torch.no_grad():
                vit_cls, mid_fea = self.model(data, return_feature=True)
                vit_cls = F.softmax(vit_cls, dim=1)
            
            loss = F.cross_entropy(logits_classifier, target)
                
            
            if self.logits_distill:  # logits distillation
                loss_kl = self.kl_loss(log_soft, vit_cls)
                loss += 4*loss_kl
                
            if self.fea_distill:  # feature distillation
                kd_fea = self.dt(mid_fea)
                fea_log_soft = F.log_softmax(feature, dim=1)
                kd_fea = F.softmax(kd_fea, dim=1)
                mid_fea_kd = self.mid_fea_kd_loss(fea_log_soft, kd_fea)
                loss += 8*mid_fea_kd
            
            if self.ood_fea_distill:  # contrastive learning loss 1
                idx = torch.randperm(self.num_knn_ood)[:self.ood_sample_num]
                selected_ood_samples = self.ood_samples[idx]  # random selected samples
                cdl = self.mi_loss.forward(selected_ood_samples, feature)
                loss += 0.1*cdl
            
            if self.deep_ood_distill:  # contrastive learning loss 2
                idx2 = torch.randperm(self.diffusion_ood_num)[:self.ood_sample_num]
                selected_ood_samples2 = self.deep_ood_samples[idx2]
                deep_cdl = self.mi_loss2.forward(selected_ood_samples2, feature)
                loss += 0.2*deep_cdl

            if self.energy_rl:
                idx = torch.randperm(self.num_knn_ood)[:self.ood_sample_num]
                selected_ood_samples = self.ood_samples[idx]  # random selected samples
                idx2 = torch.randperm(self.diffusion_ood_num)[:self.ood_sample_num]
                selected_ood_samples2 = self.deep_ood_samples[idx2]
                # print("GH, the ood feature shape:{}".format(self.dt(self.deep_ood_samples[idx2]).shape))
                Ec_out1 = torch.logsumexp(self.dt1(selected_ood_samples), dim=1)
                Ec_in1 = torch.logsumexp(feature, dim=1)
                # print("GH, the Ec_out shape:{}".format(Ec_out1.shape))  # the Ec_out shape:torch.Size([128])
                # print("GH, the feature length:{}".format(len(feature)))  128
                binary_labels1 = torch.ones((len(feature)+self.ood_sample_num)).cuda()
                binary_labels1[len(feature):] = 0
                # print("GH, the binary_labels1 shape:{}".format(binary_labels1.shape))   # 256
                input_for_lr1 = torch.cat((Ec_in1, Ec_out1), -1)
                # print("GH, the input_for_lr1 shape:{}".format(input_for_lr1.shape))  # the input_for_lr1 shape:torch.Size([256])
                # print("GH, the Ec_out shape:{}".format(Ec_out1.shape))
                criterion1 = torch.nn.CrossEntropyLoss()
                output1 = self.logistic_regression1(input_for_lr1.reshape(-1, 1))
                # print("GH, the output1 shape:{}".format(output1.shape))  # the output1 shape:torch.Size([256, 2]) 
                energy_reg_loss1 = criterion1(output1, binary_labels1.long())

                loss += 2.5 * energy_reg_loss1

                Ec_out2 = torch.logsumexp(self.dt2(selected_ood_samples2), dim=1)
                Ec_in2 = torch.logsumexp(feature, dim=1)
                # print("GH, the Ec_out shape:{}".format(Ec_out.shape))
                # print("GH, the feature length:{}".format(len(feature)))  128
                binary_labels2 = torch.ones((len(feature)+self.ood_sample_num)).cuda()
                binary_labels2[len(feature):] = 0

                input_for_lr2 = torch.cat((Ec_in2, Ec_out2), -1)
                criterion2 = torch.nn.CrossEntropyLoss()
                output2 = self.logistic_regression2(input_for_lr2.reshape(-1, 1))
                energy_reg_loss2 = criterion2(output2, binary_labels2.long())

                loss += 2.5 * energy_reg_loss2

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
