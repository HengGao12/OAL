#!/bin/bash
# sh scripts/ood/kl_matching/cifar100_test_ood_kl_matching.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='5'
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
    configs/postprocessors/klm.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar100_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001ml_0.001dml/s0/best_epoch441_acc0.8050.ckpt' \
    --mark 0 --merge_option merge
