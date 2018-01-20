#!/usr/bin/env bash
# Path to python
TF_PY=""

TRAIN_DATA_DIR=""
TEST_DATA_DIR=""

CHECKPOINT=""

MODEL_CONFIG="model_configs/best_wave_net_config"

$TF_PY runNet.py --data_dir=$TEST_DATA_DIR --start_checkpoint=$CHECKPOINT --arch_config_file=$MODEL_CONFIG