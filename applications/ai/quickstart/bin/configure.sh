#!/bin/bash

USER_HOME=/home/$(whoami)
APPLICATIONS_HOME=$USER_HOME/applications

# Application path on cluster node
export QUICKSTART_HOME=${APPLICATIONS_HOME}/quickstart
export MODEL_DIR=$QUICKSTART_HOME

if test -e "/mnt/cloudtik/data_disk_1/"
then
    export QUICKSTART_WORKING=/mnt/cloudtik/data_disk_1/quickstart
else
    export QUICKSTART_WORKING=$USER_HOME/quickstart
fi

if test -e "/cloudtik/fs"
then
    export QUICKSTART_WORKSPACE="/cloudtik/fs/quickstart"
else
    export QUICKSTART_WORKSPACE=$QUICKSTART_WORKING
fi

# Set Jemalloc Preload for better performance
export LD_PRELOAD=$HOME/anaconda3/envs/cloudtik/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# Set IOMP preload for better performance
export LD_PRELOAD=$HOME/anaconda3/envs/cloudtik/lib/libiomp5.so:$LD_PRELOAD

function move_to_workspace() {
    # Move a folder (the parameter) into workspace
    if [ $QUICKSTART_WORKSPACE != $QUICKSTART_WORKING ]; then
      mkdir -p $QUICKSTART_WORKSPACE
      cp -r -n $1 $QUICKSTART_WORKSPACE
    fi
}
