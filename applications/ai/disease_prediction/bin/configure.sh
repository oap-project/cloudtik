#!/bin/bash

USER_HOME=/home/$(whoami)
APPLICATIONS_HOME=$USER_HOME/applications

# Application path on local machine
DISEASE_PREDICTION_HOME=${APPLICATIONS_HOME}/disease_prediction
DISEASE_PREDICTION_TMP=/tmp/disease_prediction

# Working path on the local machine
if test -e "/mnt/cloudtik/data_disk_1/"
then
    DISEASE_PREDICTION_WORKING=/mnt/cloudtik/data_disk_1/disease_prediction
else
    DISEASE_PREDICTION_WORKING=$USER_HOME/disease_prediction
fi

# Workspace path on shared storage, working path if shared storage not exists
if test -e "/cloudtik/fs"
then
    DISEASE_PREDICTION_WORKSPACE="/cloudtik/fs/disease_prediction"
else
    DISEASE_PREDICTION_WORKSPACE=$DISEASE_PREDICTION_WORKING
fi

DISEASE_PREDICTION_DATA=$DISEASE_PREDICTION_WORKSPACE/data

function move_to_workspace() {
    # Move a folder (the first parameter) into workspace target(the second parameter)
    if [ $DISEASE_PREDICTION_WORKSPACE != $DISEASE_PREDICTION_WORKING ]; then
        SOURCE_TO_MOVE=$1
        TARGET=$2

        if [ "$TARGET" != "" ]; then
            TARGET_HOME=$DISEASE_PREDICTION_WORKSPACE/$TARGET
        else
            TARGET_HOME=$DISEASE_PREDICTION_WORKSPACE
        fi

        SOURCE_NAME="$(basename -- $SOURCE_TO_MOVE)"
        TARGET_DIR=$TARGET_HOME/$SOURCE_NAME

        # rm if the target exists
        if [ -d "$TARGET_DIR" ]; then
            rm -rf "$TARGET_DIR"
        fi

        mkdir -p $TARGET_HOME
        cp -r -n $SOURCE_TO_MOVE $TARGET_HOME
    fi
}
