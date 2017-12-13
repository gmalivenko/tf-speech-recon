concurrent trainNetwork:
    tr 1

#sequential prepareSomeStuff(qsub="-l h_vmem=512M -l h_rt=0:05:00"):
	# do some preparation for the training that does not require gpu
    
concurrent trainNetwork
parallel start128_testNetwork(qsub="-hard -l h_vmem=15G -l h_rt=80:00:00 -l gpu=1"):
    
    source /etc/lsb-release
    echo "Ubuntu $DISTRIB_RELEASE $DISTRIB_CODENAME"
    source ./config/activate-cuda.sh
    source /u/bozheniuk/tensorflow-gpu/bin/activate
    PY="python3"
    
    echo "CUDA_VISIBLE_DEVICES = '$CUDA_VISIBLE_DEVICES'"
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # TF will not automatically select a free GPU.
        # So just let the first free GPU be the only visible GPU to TF.
        export CUDA_VISIBLE_DEVICES=$(./config/first-free-gpu.py || echo "")
        if [ "$CUDA_VISIBLE_DEVICES" = "" ]; then
            echo "Error, no GPU found."
            ./config/test-gpus.py
            exit 1
        fi
        echo "Using GPU$CUDA_VISIBLE_DEVICES (mapped as /gpu:0)"
    fi

    $PY runNet.py
