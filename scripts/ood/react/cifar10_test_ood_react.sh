#!/bin/bash
# sh scripts/ood/react/cifar10_test_ood_react.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood
export CUDA_VISIBLE_DEVICES='5'
PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/react_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/react.yml \
    --network.pretrained False \
    --network.backbone.name resnet18_224x224 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint 'results/cifar10_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt' \
    --num_workers 8 \
    --mark 0