export CUDA_VISIBLE_DEVICES="1"
GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.'$PYTHONPATH \
python main.py --config configs/datasets/cifar100/cifar100_224x224.yml configs/datasets2/cifar100/cifar100_224x224.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_224x224.yml configs/pipelines/train/oal_pipeline.yml --seed 0 --merge_option merge  # res-18
# python main.py --config configs/datasets/cifar100/cifar100_224x224.yml configs/datasets2/cifar100/cifar100_224x224.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet50.yml configs/pipelines/train/oal_pipeline.yml --seed 0 --merge_option merge  # res-50