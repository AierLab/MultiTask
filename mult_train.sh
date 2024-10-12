#!/bin/bash

# 设置可见的GPU设备
export CUDA_VISIBLE_DEVICES=2,3,4,5
export MASTER_ADDR='localhost'
export MASTER_PORT='29502'  # 更改端口号
export WORLD_SIZE=1  

for RANK in $(seq 0 $(($WORLD_SIZE - 1))); do
    # 运行Python训练脚本
    python -m torch.distributed.launch --nproc_per_node=1 /home/4paradigm/WGWS-Net/train_share.py \
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
        --rank $RANK &  # 使用后台运行
done

wait  # 等待所有后台进程完成
