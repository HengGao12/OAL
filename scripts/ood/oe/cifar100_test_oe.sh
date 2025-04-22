#!/bin/bash
# sh scripts/ood/oe/cifar100_test_oe.sh
export CUDA_VISIBLE_DEVICES='7'
GPU=1
CPU=1
node=63
jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar100_oe_resnet18_224x224_oe_e100_lr0.1_lam0.5_default/s0/best.ckpt' \
    --mark 0 --merge_option merge
