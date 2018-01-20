#!/usr/bin/env bash
#Path to python
TF_PY="/home/vitaly/new_tf/bin/python3"

TRAIN_DATA_DIR=""
TEST_DATA_DIR=""
CHECKPOINT_PATH=""
SUMMARIES_DIR=""
CHECKPOINT=""

MODEL_CONFIG="model_configs/best_wave_net_config"

$TF_PY train.py --data_dir=$TRAIN_DATA_DIR --start_checkpoint=$CHECKPOINT --checkpoint_dir=$CHECKPOINT_PATH --arch_config_file=$MODEL_CONFIG --summaries_dir=$SUMMARIES_DIR
