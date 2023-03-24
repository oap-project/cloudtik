#!/bin/bash


if test -e "/mnt/cloudtik/data_disk_1/"
then
    INTELAI_MODELS_LOCAL_PATH=/mnt/cloudtik/data_disk_1/intelai_models_local
else
    INTELAI_MODELS_LOCAL_PATH=$USER_HOME/intelai_models_local
fi

if test -e "/cloudtik/fs"
then
    INTELAI_MODELS_PATH="/cloudtik/fs/intelai_models"
else
    INTELAI_MODELS_PATH=$INTELAI_MODELS_LOCAL_PATH
fi

export INTELAI_MODELS_LOCAL_WORKSPACE=$INTELAI_MODELS_LOCAL_PATH/workspace

export MODELS_SCRIPTS_HOME=$INTELAI_MODELS_LOCAL_PATH/scripts
export MODEL_DIR=$INTELAI_MODELS_LOCAL_PATH/models
export INTELAI_MODELS_WORKSPACE=$INTELAI_MODELS_PATH/workspace

# Set Jemalloc Preload for better performance
export LD_PRELOAD=$HOME/anaconda3/envs/cloudtik/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# Set IOMP preload for better performance
export LD_PRELOAD=$HOME/anaconda3/envs/cloudtik/lib/libiomp5.so:$LD_PRELOAD

# Use AMX for DNNL
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

function move_to_shared_dict() {
    if [ $INTELAI_MODELS_PATH != $INTELAI_MODELS_LOCAL_PATH ]; then
      mkdir -p $INTELAI_MODELS_WORKSPACE
      cp -r -n $1 $INTELAI_MODELS_WORKSPACE
    fi
}
