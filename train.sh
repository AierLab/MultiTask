#!/bin/bash

# 设置可见的GPU设备
export CUDA_VISIBLE_DEVICES=2,3

# 运行Python训练脚本
python /home/4paradigm/WGWS-Net/train_share.py \
    --lam 0.008 \
    --VGG_lamda 0.2 \
    --learning_rate 0.0001 \
    --fix_sample 9000 \
    --Crop_patches 224 \
    --BATCH_SIZE 12 \
    --EPOCH 120 \
    --T_period 30 \
    --flag K1 \
    --base_channel 18 \
    --print_frequency 100
