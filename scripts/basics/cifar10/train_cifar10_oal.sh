export CUDA_VISIBLE_DEVICES="6"
GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.'$PYTHONPATH \
python main.py --config configs/datasets/cifar10/cifar10_224x224.yml configs/datasets2/cifar10/cifar10_224x224.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet18_224x224.yml configs/pipelines/train/oal_pipeline.yml --seed 0 --num_workers 8