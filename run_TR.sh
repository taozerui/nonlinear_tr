#!/bin/bash

CUDA_VISIBLE_DEVICES=8 python imputation.py \
  --dataset indoor \
  --batch_size 64 \
  --model_type tr \
  --prior normal \
  --in_rank 5 \
  --out_rank 10 \
  --lr 1e-3 \
  --anneal_lr \
  --max_iter 40000 \
  --h_dim 64 64 64 \
  --tensorize \
  --conv_layer \
  --conv_filter 128 256 256
