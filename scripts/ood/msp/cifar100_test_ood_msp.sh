#!/bin/bash
# sh scripts/ood/msp/cifar100_test_ood_msp.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood
export CUDA_VISIBLE_DEVICES='1'
PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-test3/s0/best.ckpt' \
    --mark 0

# original: results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# oal1: results/cifar100_resnet18_224x224_base_oal_e100_lr0.005_default-submission-aaai-test1/s0/best.ckpt