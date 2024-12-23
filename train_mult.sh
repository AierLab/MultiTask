#!/bin/bash

# 设置可见的GPU设备，代码里其实也设置了
# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=4
export MASTER_ADDR='localhost'
export MASTER_PORT='29514'  # 更改端口号
# export WORLD_SIZE=4 # 设置进程数  
export WORLD_SIZE=1 # 设置进程数, 最好是GPU数量

# 运行Python训练脚本
python train_mult.py \
    --lam 0.008 \
    --VGG_lamda 0.2 \
    --learning_rate 0.0001 \
    --fix_sample 9000 \
    --Crop_patches 224 \
    --BATCH_SIZE 9 \
    --EPOCH 20 \
    --T_period 30 \
    --flag S1 \
    --base_channel 18 \
    --print_frequency 100 \
    --world_size $WORLD_SIZE \
