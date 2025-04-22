#!/bin/bash
# sh scripts/ood/mds/imagenet_test_ood_mds.sh
export CUDA_VISIBLE_DEVICES='1'
GPU=1
CPU=1
node=63
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/mds.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'results/imagenet_resnet50_base_generative_ood_distill_trainer_e50_lr5e-05_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.0001mi_loss_0.0001ml_0.0001dml_0.01dt/s0/best.ckpt' \
    --merge_option merge
