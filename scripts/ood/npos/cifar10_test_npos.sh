#!/bin/bash
# sh scripts/ood/npos/cifar10_test_npos.sh
export CUDA_VISIBLE_DEVICES='7'

python main.py --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/npos_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --network.backbone.name resnet18_224x224 \
    --num_workers 8 \
    --network.checkpoint 'results/cifar10_npos_net_npos_e100_lr0.1_default-new-submission-11-7/s0/best.ckpt' \
    --mark 0 \
    --merge_option merge
# configs/postprocessors/npos.yml \    
#############################results/cifar10_npos_net_npos_e100_lr0.1_default-new-submission###############
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# python scripts/eval_ood.py \
#    --id-data cifar10 \
#    --root ./results/cifar10_npos_net_npos_e100_lr0.1_default-new-submission \
#    --postprocessor npos \
#    --save-score --save-csv
