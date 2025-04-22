# Enhancing OOD Detection Using Latent Diffusion
---

### Usage

#### Installation

```sh
conda create -n openood python=3.8
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
wget https://codeload.github.com/Vastlab/libMR/zip/refs/heads/master
cd python
pip install -r requirements.txt
cd ../
pip install .
cd ../
git clone https://github.com/Jingkang50/OpenOOD.git
cd OpenOOD
pip install -e .
pip install timm
```

In order to better adapt to the OpenOOD framework, we changed the  `vision_transformer.py`  in the [Pytorch-Image-Models](https://github.com/huggingface/pytorch-image-models) library as follows:

```python
...
    def forward(self, x, return_feature):
        x = self.forward_features(x)
        x, pre_logits = self.forward_head(x)
        if return_feature:
            return x, pre_logits  
        else:
            return x
   
    def get_fc(self):
        fc = self.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.head
...
```

#### Data Preparation

Our codebase accesses the datasets from `./data/` and pretrained models from `./results/checkpoints/` . One can download the datasets via running  `./scripts/download/download.py`.

```
├── ...
├── data
│   ├── benchmark_imglist
│   ├── images_classic
│   └── images_largescale
├── openood
├── results
│   ├── checkpoints
│   └── ...
├── scripts
├── main.py
├── ...
```

To synthesize OOD samples for outlier exposure training, please refer to [DreamOOD](https://github.com/deeplearning-wisc/dream-ood) code repository for installation. Note that, here we use `xformers==0.0.13`.


#### Get Token Embeddings

```python
python outlier_generation/scripts/get_token_embed.py
```

#### Get ID embeddings from Transformers

```python
# CIFAR-10
python scripts/get_id_features_cifar10.py
# CIFAR-100
python scripts/get_id_features_cifar100.py
```

#### Sample OOD data using k-NN distance search

```python
# CIFAR-10
python scripts/get_embed_cifar10.py
# CIFAR-100
python scripts/get_embed_cifar100.py
```

#### Outlier Synthesis using Stable Diffusion

```sh
bash outlier_generation/generate_outliers_in_pixel_space.sh
```

#### OAL Training

##### Teacher Model Pretrained Weights
Here is the pretrained weights of teacher models for Knowledge Distillation.

| In-Distribution Dataset  |                Download Links                  |
| :---------: | :--------------------------------------: |  
|  CIFAR-10   | [OneDrive](https://1drv.ms/u/c/409fe51635b9369c/EWtKK1PK-QVJlp3aXop6So4BcQe7fBUfo_yC5DGtw3bZUQ?e=hMtw7B) |
|  CIFAR-100  | [OneDrive]() |

```sh
# Trained by OAL on CIAFR-10
bash scripts/basics/cifar10/train_cifar10_oal.sh
# Trained by OAL on CIFAR-100
bash scripts/basics/cifar100/train_cifar100_oal.sh
```

##### OOD Testing

```sh
# Test on CIFAR-100 using EBO score
bash scripts/ood/ebo/cifar100_test_ood_ebo.sh

# Test on CIFAR-10 using EBO score
bash scripts/ood/ebo/cifar10_test_ood_ebo.sh
```
Before running this command, please load the [pre-trained weights](https://1drv.ms/f/c/409fe51635b9369c/EmLzpJb_fFpFtf3aYFgSE4QBBa-kS0teDWRcjyMDMerlfg?e=SMSVNx) for testing.

### Benchmark Evaluation Results

Waiting for official publication.

### Acknowledgments

OAL is developed based on [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main), [Pytorch-Image-Models](https://github.com/huggingface/pytorch-image-models) and [DreamOOD](https://github.com/deeplearning-wisc/dream-ood). Thanks to their great works.
