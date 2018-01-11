#!/usr/bin/env bash
TF_PY="/home/vitaly/new_tf/bin/python3"

TRAIN_DATA_DIR="/home/vitaly/PycharmProjects/tf-speech-recon/data/train/audio/"
TEST_DATA_DIR="/home/vitaly/PycharmProjects/tf-speech-recon/data/test/audio/"
CHECKPOINT_PATH=""

CHECKPOINT="/home/vitaly/competition/graph/lace_32/lace.ckpt-18000"

MODEL_CONFIG="model_configs/lace_config"

$TF_PY train.py --data_dir=$TRAIN_DATA_DIR --start_checkpoint=$CHECKPOINT --checkpoint_dir=$CHECKPOINT_PATH --arch_config_file=$MODEL_CONFIG
