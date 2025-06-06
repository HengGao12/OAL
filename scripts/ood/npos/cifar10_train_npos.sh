#!/bin/bash
# sh scripts/ood/npos/cifar10_train_npos.sh
export CUDA_VISIBLE_DEVICES='1'

python main.py \
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/networks/npos_net.yml \
    configs/pipelines/train/train_npos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_224x224 \
    --dataset.train.batch_size 128 \
    --trainer.trainer_args.temp 0.1 \
    --trainer.trainer_args.sample_from 600 \
    --trainer.trainer_args.K 300 \
    --trainer.trainer_args.cov_mat 0.1 \
    --trainer.trainer_args.start_epoch_KNN 40 \
    --trainer.trainer_args.ID_points_num 200 \
    --optimizer.num_epochs 100 \
    --optimizer.lr 0.1 \
    --seed 0 \
    --merge_option merge
