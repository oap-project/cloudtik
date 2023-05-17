#!/bin/bash

USER_HOME=/home/$(whoami)
BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools

# Tool path on local machine
export INTELAI_MODELS_HOME=$BENCHMARK_TOOL_HOME/intelai_models
export MODELS_HOME=$INTELAI_MODELS_HOME/models
export MODEL_DIR=$MODELS_HOME
export SCRIPTS_HOME=$INTELAI_MODELS_HOME/scripts
export CLOUDTIK_MODELS_HOME=$SCRIPTS_HOME/models

if test -e "/mnt/cloudtik/data_disk_1/"
then
    export INTELAI_MODELS_WORKING=/mnt/cloudtik/data_disk_1/intelai_models
else
    export INTELAI_MODELS_WORKING=$USER_HOME/intelai_models
fi

if test -e "/cloudtik/fs"
then
    export INTELAI_MODELS_WORKSPACE="/cloudtik/fs/intelai_models"
else
    export INTELAI_MODELS_WORKSPACE=$INTELAI_MODELS_WORKING
fi

# Set Jemalloc Preload for better performance
export LD_PRELOAD=$HOME/anaconda3/envs/cloudtik/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# Set IOMP preload for better performance
export LD_PRELOAD=$HOME/anaconda3/envs/cloudtik/lib/libiomp5.so:$LD_PRELOAD

function move_to_workspace() {
    # Move a folder (the parameter) into workspace
    if [ $INTELAI_MODELS_WORKSPACE != $INTELAI_MODELS_WORKING ]; then
      mkdir -p $INTELAI_MODELS_WORKSPACE
      cp -r -n $1 $INTELAI_MODELS_WORKSPACE
    fi
}
