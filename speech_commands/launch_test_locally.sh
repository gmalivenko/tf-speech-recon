#!/usr/bin/env bash
TF_PY="/home/vitaly/new_tf/bin/python3"

TRAIN_DATA_DIR="/home/vitaly/PycharmProjects/tf-speech-recon/data/train/audio/"
TEST_DATA_DIR="/home/vitaly/PycharmProjects/tf-speech-recon/data/test/audio/"

CHECKPOINT="/home/vitaly/PycharmProjects/tf-speech-recon/graph/adv_lace/model.ckpt-9000"

MODEL_CONFIG="model_configs/adversarial_lace_config"

$TF_PY runNet.py --data_dir=$TEST_DATA_DIR --start_checkpoint=$CHECKPOINT --arch_config_file=$MODEL_CONFIG

