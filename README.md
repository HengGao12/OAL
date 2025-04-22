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

##### OAL Training

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
```
Before running this command, please load the oal pre-trained weights for testing.