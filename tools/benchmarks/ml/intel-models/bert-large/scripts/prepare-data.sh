#!/bin/bash

USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
BERT_MODEL=$ML_WORKSPACE/bert_model

function download_bert_models() {
    mkdir -p $BERT_MODEL

    cd $BERT_MODEL

    ### Download
    # bert_config.json
    gdown https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW
    # vocab.txt
    gdown https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1

    # model.ckpt-28252.data-00000-of-00001
    gdown https://drive.google.com/uc?id=1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_
    # model.ckpt-28252.index
    gdown https://drive.google.com/uc?id=1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0
    # model.ckpt-28252.meta
    gdown https://drive.google.com/uc?id=1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv


}