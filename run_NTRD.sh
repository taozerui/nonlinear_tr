#!/bin/bash

CUDA_VISIBLE_DEVICES=8 python imputation.py \
  --dataset indoor \
  --batch_size 64 \
  --model_type gptr \
  --in_rank 5 \
  --out_rank 10 \
  --prior sparse_gp \
  --lr 3e-4 \
  --anneal_lr \
  --max_iter 40000 \
  --h_dim 64 64 64 \
  --anneal_beta \
  --beta_max 0.1 \
  --tensorize \
  --conv_layer \
  --conv_filter 128 256 256
