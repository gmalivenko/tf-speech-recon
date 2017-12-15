#!/usr/bin/env bash
TF_PY="/home/vitaly/new_tf/bin/python3"

TRAIN_DATA_DIR="/home/vitaly/competition/train/audio/"
TEST_DATA_DIR="/home/vitaly/competition/test/audio/"

CHECKPOINT="/home/vitaly/competition/graph/adv_lace_32/model.ckpt-24000"

MODEL_CONFIG="model_configs/adversarial_lace_config"

$TF_PY runNet.py --data_dir=$TEST_DATA_DIR --start_checkpoint=$CHECKPOINT --arch_config_file=$MODEL_CONFIG