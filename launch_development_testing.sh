#!/usr/bin/env bash
TF_PY=""

TRAIN_DATA_DIR="/"
TEST_DATA_DIR="/"
CHECKPOINT_PATH=""

CHECKPOINT=""

MODEL_CONFIG="model_configs/conv1d.config"

$TF_PY test_ckpt.py --data_dir=$TRAIN_DATA_DIR --start_checkpoint=$CHECKPOINT --checkpoint_dir=$CHECKPOINT_PATH --arch_config_file=$MODEL_CONFIG
#!/usr/bin/env bash