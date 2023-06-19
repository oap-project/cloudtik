#!/bin/bash

USER_HOME=/home/$(whoami)
APPLICATIONS_HOME=$USER_HOME/applications

# Application path on local machine
FRAUD_DETECTION_HOME=${APPLICATIONS_HOME}/fraud_detection
FRAUD_DETECTION_TMP=/tmp/fraud_detection

# Working path on the local machine
if test -e "/mnt/cloudtik/data_disk_1/"
then
    FRAUD_DETECTION_WORKING=/mnt/cloudtik/data_disk_1/fraud_detection
else
    FRAUD_DETECTION_WORKING=$USER_HOME/fraud_detection
fi

# Workspace path on shared storage, working path if shared storage not exists
if test -e "/cloudtik/fs"
then
    FRAUD_DETECTION_WORKSPACE="/cloudtik/fs/fraud_detection"
else
    FRAUD_DETECTION_WORKSPACE=$FRAUD_DETECTION_WORKING
fi

FRAUD_DETECTION_DATA=$FRAUD_DETECTION_WORKSPACE/data

function move_to_workspace() {
    # Move a folder (the parameter) into workspace
    if [ $FRAUD_DETECTION_WORKSPACE != $FRAUD_DETECTION_WORKING ]; then
        SOURCE_TO_MOVE=$1
        SOURCE_NAME="$(basename -- $SOURCE_TO_MOVE)"
        # rm if the target exists
        if [ -d "$DISEASE_PREDICTION_WORKSPACE/$SOURCE_NAME" ]; then
            rm -rf "$DISEASE_PREDICTION_WORKSPACE/$SOURCE_NAME"
        fi

        mkdir -p $FRAUD_DETECTION_WORKSPACE
        cp -r -n $SOURCE_TO_MOVE $FRAUD_DETECTION_WORKSPACE
    fi
}
