#!/bin/bash

CUDA_VISIBLE_DEVICES=9 python imputation.py \
  --dataset indoor \
  --batch_size 64 \
  --model_type gpvae \
  --z_dim 64 \
  --prior normal \
  --num_particles 1 \
  --lr 3e-4 \
  --max_iter 40000 \
  --h_dim 64 64 64 \
  --conv_layer \
  --anneal_lr \
  --conv_filter 128 256 256
