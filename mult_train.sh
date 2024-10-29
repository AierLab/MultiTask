#!/bin/bash

# 设置可见的GPU设备，代码里其实也设置了
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3,4,5
export MASTER_ADDR='localhost'
export MASTER_PORT='29502'  # 更改端口号
# export WORLD_SIZE=4 # 设置进程数  
export WORLD_SIZE=2 # 设置进程数, 最好是GPU数量

for RANK in $(seq 0 $(($WORLD_SIZE - 1))); do # 循环启动多个进程
    # 运行Python训练脚本
    python train_share.py \
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
        --print_frequency 100 \
        --word_size $WORLD_SIZE \
        --rank $RANK &  # 使用后台运行
done

wait  # 等待所有后台进程完成