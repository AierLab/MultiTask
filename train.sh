#!/bin/bash

# 设置可见的GPU设备
# export CUDA_VISIBLE_DEVICES=2,3,4,5
export WORLD_SIZE=6  # 设置为使用的 GPU 数量
export MASTER_ADDR='localhost'  # 或者替换为主节点的 IP 地址
# export MASTER_PORT='29502'


export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# 运行Python训练脚本
python /home/4paradigm/WGWS-Net/train_mult.py \
    --lam 0.008 \
    --VGG_lamda 0.2 \
    --learning_rate 0.0001 \
    --fix_sample 9000 \
    --Crop_patches 224 \
    --BATCH_SIZE 8 \
    --EPOCH 120 \
    --T_period 30 \
    --flag K1 \
    --base_channel 18 \
    --print_frequency 100 \
    # --rank $RANK &
    # --local_rank 0
# done

# wait
