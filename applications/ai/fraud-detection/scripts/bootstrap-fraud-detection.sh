#!/bin/bash

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

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

function is_head_node() {
    if [ -n $IS_HEAD_NODE ]; then
        cloudtik head head-ip
        GET_HEAD_IP_CODE=$?
        if [ ${GET_HEAD_IP_CODE} -eq "0" ]; then
            IS_HEAD_NODE=true
        else
            IS_HEAD_NODE=false
        fi
    fi
}

function prepare() {
    source ~/.bashrc
    sudo apt-get update -y

    mkdir -p $FRAUD_DETECTION_HOME
    sudo chown $(whoami) $FRAUD_DETECTION_HOME

    mkdir -p $FRAUD_DETECTION_WORKING
    sudo chown $(whoami) $FRAUD_DETECTION_WORKING

    mkdir -p $FRAUD_DETECTION_WORKSPACE
    sudo chown $(whoami) $FRAUD_DETECTION_WORKSPACE
}

function install_tools() {
    :;
}

function install_libaries() {
    pip install --no-cache-dir torchmetrics tqdm cmake pyyaml ogb chardet
    pip install --no-cache-dir --pre dgl -f https://data.dgl.ai/wheels/repo.html && \
    pip install --no-cache-dir --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
}

function install_fraud_detection() {
    mkdir -p $FRAUD_DETECTION_TMP
    cd $FRAUD_DETECTION_TMP
    rm -rf $FRAUD_DETECTION_TMP/cloudtik
    git clone https://github.com/oap-project/cloudtik.git
    rm -rf $FRAUD_DETECTION_HOME/*
    mkdir -p $FRAUD_DETECTION_HOME
    cp -r cloudtik/applications/ai/fraud-detection/* $FRAUD_DETECTION_HOME/
    rm -rf $FRAUD_DETECTION_TMP/cloudtik
}

prepare
install_tools
install_libaries
install_fraud_detection
