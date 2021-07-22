#!/usr/bin/env bash

# TODO: change these env vars accordingly
ROOT_DIR=.
DATASET_DIR=datasets
MODEL_DIR=models
DATA_PATH="$DATASET_DIR/fMRI_sixtask_LR_npy/points"
OUT="$MODEL_DIR/aug$split_seed"
OUT_PATH="$OUT/$this_model"

pkill python; cd $ROOT_DIR

masternode=27
split_seed=42
nnodes=12
nproc_per_node=4
is_debug=1
is_restart=0
this_model=pooled_transformer_full
exp=fmri
name=transformer_full

python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$NODERANK --master_addr="192.168.128.$masternode" --master_port=64999 \
main_dist.py \
--batch_size 2 \
--split_seed $split_seed \
--initialization xavier_unif \
--preload_workers 15 \
--num_workers 0  \
--data_path "$DATA_PATH" \
--exp $exp \
--is_debug $is_debug \
--is_restart $is_restart \
--label_file meta.txt \
--lr 0.0003 \
--lr_start 0.0003 \
--lr_mid 0.0002 \
--lr_final 0.0001 \
--model $this_model \
--momentum 0.9 \
--weight_decay 0.0001 \
--n_classes 7 \
--n_epoch 150 \
--n_frames 1 \
--max_frames 28 \
--val_max_frames 28 \
--name $name \
--out_path $OUT_PATH \
--print_every -1 \
--test_size 0.2 \
--val_size 0.1 \
--train_block 300 \
--val_block 100 \
--val_every -1 \
--save_every -1 \
--save_every_epoch 1 \
--opt adam \
--pad_mode loop \
--train_random_head 0 \
--val_random_crop 0 \
--skip_frame_n 1 \
--skip_frame_to_skip 0 \
