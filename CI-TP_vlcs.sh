#!/bin/sh

#source /etc/network_turbo
export HUGGINGFACE_HUB_CACHE='root/autodl-tmp/hf'
export HF_HOME='root/autodl-tmp/hf'
export XDG_CACHE_HOME='root/autodl-tmp/hf'

export CUDA_VISIBLE_DEVICES=0

for env in 0 1 2 3
do 
python -m domainbed.scripts.train_causal \
    --data_dir=/root/autodl-tmp/data/linjunyu/code/SDViT/domainbed/data \
    --dataset VLCS \
    --algorithm CI_TP\
    --output_dir=./domainbed/VLCS_Output/CI-TP/test_env$env \
    --test_env $env \
    --hparams """{\"backbone\":\"ResNet50\",\"batch_size\":32,\"lr\":1e-4,\"resnet_dropout\":0.2,\"weight_decay\":0.0,\"k\":30}"""
done