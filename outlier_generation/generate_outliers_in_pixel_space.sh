export CUDA_VISIBLE_DEVICES='5'
python ./outlier_generate.py --plms \
    --n_iter 100 --n_samples 3 \
    --outdir ./nobackup-fast/txt2img-samples-cifar100-test/ \
    --loaded_embedding ./cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy \
    --ckpt ./nobackup-slow/dataset/diffusion/sd-v1-4.ckpt \
    --id_data cifar100 \
    --skip_grid