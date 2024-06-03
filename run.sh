#!/bin/bash

# Hyperparameter settings
learning_rates=(0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001)
weight_decays=(10 5 2 1 0.5 0.2 0.1 0.05 0.02)
model_seeds=(7 77 777)
optimizers=("Adam" "AdamW")

# Directories for the training scripts
vgg_dir="./VGG"
vae_dir="./VAE"

# To prevent weight_decay becomes 0
convert_weight_decay() {
    adjusted_wd=$(awk "BEGIN {printf \"%.10f\", $wd * 0.001}")
    echo $adjusted_wd
}

# Function to run VGG + CIFAR 10 experiment
run_vgg_cifar10() {
    model_seed=$1
    lr=$2
    wd=$3
    optimizer=$4
    epochs=$5
    version=$6

    echo "Running VGG + CIFAR 10 with $optimizer, seed=$model_seed, lr=$lr, wd=$wd for $epochs epochs for test $6"
    # Command to run the VGG + CIFAR 10 training script
    (cd $vgg_dir && python3 main.py -optim $optimizer -s $model_seed -lr $lr -wd $wd -e $epochs -b 128 -v $6)
}

# Function to run VAE + Fashion MNIST experiment
run_vae_fashion_mnist() {
    model_seed=$1
    lr=$2
    wd=$3
    optimizer=$4
    epochs=$5
    version=$6

    echo "Running VAE + Fashion MNIST with $optimizer, seed=$model_seed, lr=$lr, wd=$wd for $epochs epochs for test $6"
    # Command to run the VAE + Fashion MNIST training script
    (cd $vae_dir && python3 main.py -optim $optimizer -s $model_seed -lr $lr -wd $wd -e $epochs -b 128  -v $6)
}

# Grid test: VGG + CIFAR 10
for seed in "${model_seeds[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for wd in "${weight_decays[@]}"; do
                adjusted_wd=$(convert_weight_decay $wd)
                run_vgg_cifar10 $seed $lr $adjusted_wd $optimizer 100 "grid"
            done
        done
    done
done

# Training/Test loss: VGG + CIFAR 10
for seed in "${model_seeds[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        for wd in "${weight_decays[@]}"; do
            adjusted_wd=$(convert_weight_decay $wd)
            run_vgg_cifar10 $seed 0.0001 $adjusted_wd $optimizer 200 "train"
        done
    done
done

# Grid test: VAE + Fashion MNIST
for seed in "${model_seeds[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for wd in "${weight_decays[@]}"; do
                adjusted_wd=$(convert_weight_decay $wd)
                run_vae_fashion_mnist $seed $lr $adjusted_wd $optimizer 100 "grid"
            done
        done
    done
done

# Training/Test loss: VAE + Fashion MNIST
for seed in "${model_seeds[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        for wd in "${weight_decays[@]}"; do
            adjusted_wd=$(convert_weight_decay $wd)
            run_vae_fashion_mnist $seed 0.0001 $adjusted_wd $optimizer 200 "train"
        done
    done
done
