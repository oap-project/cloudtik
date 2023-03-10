#!/bin/bash


DLRM_HOME=$INTELAI_MODELS_WORKSPACE/dlrm
DLRM_MODEL=$BERT_HOME/model
DLRM_DATA=$BERT_HOME/data


function prepare_data() {
    cd $DLRM_DATA
    curl -O https://sacriteopcail01.z16.web.core.windows.net/day_{$(seq> -s , 1 23)}.gz
    gunzip day_*.gz
}

prepare_data
