export CUDA_VISIBLE_DEVICES="7"
GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.'$PYTHONPATH \
python main.py --config configs/datasets/cifar100/cifar100.yml configs/datasets2/cifar100/cifar100.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/generative_ood_distill_pipeline.yml --seed 0 --num_workers 8 --merge_option merge