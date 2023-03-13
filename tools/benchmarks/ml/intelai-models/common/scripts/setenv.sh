#!/bin/bash

export MODEL_DIR=$HOME/runtime/benchmark-tools/models

if test -e "/mnt/cloudtik/data_disk_1/"
then
    INTELAI_MODELS_WORKSPACE=/mnt/cloudtik/data_disk_1/intelai_models_workspace
else
    INTELAI_MODELS_WORKSPACE=$USER_HOME/intelai_models_workspace
fi

export INTELAI_MODELS_WORKSPACE=$INTELAI_MODELS_WORKSPACE

# Set Jemalloc Preload for better performance
export LD_PRELOAD=/home/cloudtik/runtime/benchmark-tools/jemalloc_lib/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# Set IOMP preload for better performance
export LD_PRELOAD=/home/cloudtik/anaconda3/envs/cloudtik/lib/libiomp5.so:$LD_PRELOAD
