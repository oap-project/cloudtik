#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

DLRM_HOME=$QUICKSTART_WORKING/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data


function prepare_data() {
    cd $DLRM_DATA
    curl -O https://sacriteopcail01.z16.web.core.windows.net/day_{$(seq> -s , 1 23)}.gz
    gunzip day_*.gz
}

prepare_data
move_to_workspace $DLRM_HOME

