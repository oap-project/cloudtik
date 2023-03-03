#!/bin/bash

ML_WORKSPACE=/mnt/cloudtik/data_disk_1/ml_workspace
DLRM_HOME=$ML_WORKSPACE/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data


function prepare_data() {
    cd $DLRM_DATA
    curl -O https://sacriteopcail01.z16.web.core.windows.net/day_{$(seq> -s , 1 23)}.gz
    gunzip day_*.gz
}

prepare_data
