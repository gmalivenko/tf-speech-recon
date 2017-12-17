#!/usr/bin/env bash

concurrent trainNetwork:
    tr 1

#sequential prepareSomeStuff(qsub="-l h_vmem=512M -l h_rt=0:05:00"):
	# do some preparation for the training that does not require gpu

concurrent trainNetwork

parallel wave_net_1_block(qsub="-hard -l h_vmem=15G -l h_rt=80:00:00 -l gpu=1"):
    source /etc/lsb-release
    echo "Ubuntu $DISTRIB_RELEASE $DISTRIB_CODENAME"
    source ./cluster_scripts/activate-cuda.sh
    source /u/bozheniuk/tensorflow-gpu/bin/activate
    PY="python3"

    TRAIN_DATA_DIR="/work/asr2/bozheniuk/tmp/speech_dataset/"
    TEST_DATA_DIR=""
    SUM_DIR="/work/asr2/sklyar/tmp/speech_commands_train/wave_net_1/retrain_logs/"
    CHECKPOINT_PATH="/work/asr2/sklyar/tmp/speech_commands_train/wave_net_1/"

    CHECKPOINT=""
    
    MODEL_CONFIG="model_configs/wave_net.config"




    echo "CUDA_VISIBLE_DEVICES = '$CUDA_VISIBLE_DEVICES'"
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # TF will not automatically select a free GPU.
        # So just let the first free GPU be the only visible GPU to TF.
        export CUDA_VISIBLE_DEVICES=$(./cluster_scripts/first-free-gpu.py || echo "")
        if [ "$CUDA_VISIBLE_DEVICES" = "" ]; then
            echo "Error, no GPU found."
            ./cluster_scripts/test-gpus.py
            exit 1
        fi
        echo "Using GPU$CUDA_VISIBLE_DEVICES (mapped as /gpu:0)"
    fi

    $PY train.py --data_dir=$TRAIN_DATA_DIR --summaries_dir=$SUM_DIR --checkpoint_dir=$CHECKPOINT_PATH --start_checkpoint=$CHECKPOINT --arch_config_file=$MODEL_CONFIG
