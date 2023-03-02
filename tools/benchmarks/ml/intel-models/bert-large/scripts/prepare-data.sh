#!/bin/bash

USER_HOME=/home/$(whoami)
ML_WORKSPACE=$USER_HOME/runtime/ml_workspace
BERT_HOME=$ML_WORKSPACE/bert
BERT_MODEL=$BERT_HOME/model
BERT_DATA=$BERT_HOME/data

BERT_INPUT_PREPROCESSING=${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/input_preprocessing/

function download_bert_models() {
    mkdir -p $BERT_MODEL
    cd $BERT_MODEL
    ### Download
    # bert_config.json
    gdown https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW
    # vocab.txt
    gdown https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1

    ### Download TF1 checkpoint
    # model.ckpt-28252.data-00000-of-00001
    gdown https://drive.google.com/uc?id=1chiTBljF0Eh1U5pKs6ureVHgSbtU8OG_
    # model.ckpt-28252.index
    gdown https://drive.google.com/uc?id=1Q47V3K3jFRkbJ2zGCrKkKk-n0fvMZsa0
    # model.ckpt-28252.meta
    gdown https://drive.google.com/uc?id=1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv

}

function download_bert_data() {
    cd $BERT_INPUT_PREPROCESSING
    # md5 sums
    gdown https://drive.google.com/uc?id=1tmMgLwoBvbEJEHXh77sqrXYw5RpqT8R_
    # processed chunks
    gdown https://drive.google.com/uc?id=14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_
    # unpack results and verify md5sums
    tar -xzf results_text.tar.gz && (cd results4 && md5sum --check ../bert_reference_results_text_md5.txt)
}


function prepare_trainint_data() {
    cd $BERT_INPUT_PREPROCESSING

    # For phase1 the seq_len=128:
    export SEQ_LEN=128
    ./parallel_create_hdf5.sh
    python3 ./chop_hdf5_files.py

    # For phase2 the seq_len=512:
    export SEQ_LEN=512
    ./parallel_create_hdf5.sh
    python3 ./chop_hdf5_files.py

}

function prepare_model() {
    cd $BERT_INPUT_PREPROCESSING
    python convert_tf_checkpoint.py --tf_checkpoint $BERT_MODEL/model.ckpt-28252 --bert_config_path $BERT_MODEL/bert_config.json --output_checkpoint $BERT_MODEL/model.ckpt-28252.pt

}

download_bert_models
prepare_model
