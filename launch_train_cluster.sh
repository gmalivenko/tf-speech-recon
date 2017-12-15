#!/usr/bin/env bash

concurrent trainNetwork:
    tr 1

#sequential prepareSomeStuff(qsub="-l h_vmem=512M -l h_rt=0:05:00"):
	# do some preparation for the training that does not require gpu

concurrent trainNetwork

parallel mobile_net_1(qsub="-hard -l h_vmem=15G -l h_rt=80:00:00 -l gpu=1"):
    source /etc/lsb-release
    echo "Ubuntu $DISTRIB_RELEASE $DISTRIB_CODENAME"
    source ./cluster_scripts/activate-cuda.sh
    source /u/bozheniuk/tensorflow-gpu/bin/activate
    PY="python3"

    TRAIN_DATA_DIR="/work/asr2/bozheniuk/tmp/speech_dataset/"
    TEST_DATA_DIR=""
    TRAIN_DIR="/work/asr2/bozheniuk/tmp/speech_commands_train/mobile_net/"
    SUM_DIR="/work/asr2/bozheniuk/tmp/retrain_logs/"
    CHECKPOINT_PATH=""

    #CHECKPOINT="/work/asr2/bozheniuk/tmp/speech_commands_train/adv_lace_32/.ckpt-5000"
    CHECKPOINT=""
    
    MODEL_CONFIG="model_configs/mobile_net_config"




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
