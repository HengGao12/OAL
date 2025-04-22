#!/bin/bash
# sh scripts/ood/ebo/cifar100_test_ood_ebo.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='7'
PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

# python main.py \
#     --config configs/datasets/cifar100/cifar100_224x224.yml \
#     configs/datasets/cifar100/cifar100_ood.yml \
#     configs/networks/resnet18_224x224.yml \
#     configs/pipelines/test/test_ood.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/ebo.yml \
#     --network.checkpoint 'results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt'

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/vit.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --network.checkpoint 'results/cifar100_vit_pretrained_finetune_trainer_e50_lr0.0001_default_cifar100_finetune_final_2/s0/best.ckpt'

# base: results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# oal1: results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-test1/s0/best.ckpt
# oal-w-noise: results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-test1-w-noise/s0/best.ckpt
# oal-w-energy-rl: results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-ablation-supp-oal-w-energy-rl/s0/best.ckpt

# vit: results/cifar100_vit_pretrained_finetune_trainer_e50_lr0.0001_default_cifar100_finetune_final_2/s0/best.ckpt

# ablations
# wo-coe-doe: results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-ablation-wo-coe-doe/s0/best.ckpt
# wo-dcoe: results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-ablation-wo-dcoe/s0/best.ckpt
# wo-distill: results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-ablation-wo-distill/s0/best.ckpt


# res50
# python main.py \
#     --config configs/datasets/cifar100/cifar100_224x224.yml \
#     configs/datasets/cifar100/cifar100_ood.yml \
#     configs/networks/resnet50.yml \
#     configs/pipelines/test/test_ood.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/ebo.yml \
#     --network.checkpoint 'results/cifar100_resnet50_base_oal_e100_lr0.005_default-submission-aaai-test3/s0/best.ckpt'

# base: results/cifar100_resnet50_base_e100_lr0.1_default/s0/best.ckpt
# oal1: results/cifar100_resnet50_base_oal_e100_lr0.005_default-submission-aaai-test1/s0/best.ckpt
