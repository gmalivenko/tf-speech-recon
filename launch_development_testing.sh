#!/usr/bin/env bash
TF_PY="/home/anya/tensorflow/bin/python3"

TRAIN_DATA_DIR="/home/anya/tf-speech-recon/train/audio/"
TEST_DATA_DIR="/home/anya/tf-speech-recon/test/audio/"
CHECKPOINT_PATH="/home/anya/tf-speech-recon/check_points/"

CHECKPOINT="/home/anya/tf-speech-recon/check_points/conv1d.ckpt-4000"

MODEL_CONFIG="model_configs/conv1d.config"

$TF_PY test_ckpt.py --data_dir=$TRAIN_DATA_DIR --start_checkpoint=$CHECKPOINT --checkpoint_dir=$CHECKPOINT_PATH --arch_config_file=$MODEL_CONFIG
#!/usr/bin/env bash