#!/bin/bash
# sh scripts/ood/vim/cifar100_test_ood_vim.sh
export CUDA_VISIBLE_DEVICES="3"
GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt' \
    --mark 0 \
    --postprocessor.postprocessor_args.dim 256