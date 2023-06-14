#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../configure.sh

BERT_HOME=$QUICKSTART_WORKING/bert
BERT_MODEL=$BERT_HOME/model
BERT_DATA=$BERT_HOME/data
SQUAD_DATA=$BERT_DATA/squad
SQUAD_MODEL=$BERT_MODEL/bert_squad_model
BERT_INPUT_PREPROCESSING=${QUICKSTART_HOME}/models/language_modeling/pytorch/bert_large/training/input_preprocessing/

PHASE="inference"

if [ ! -n "${QUICKSTART_HOME}" ]; then
  echo "Please set environment variable '\${QUICKSTART_HOME}'."
  exit 1
fi

function usage(){
    echo "Usage: prepare-data.sh  [ --phase training | inference] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --phase)
        # training or inference
        shift
        PHASE=$1
        ;;
    *)
        usage
    esac
    shift
done

function download_bert_training_model() {
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

function download_bert_training_data() {
    cd $BERT_INPUT_PREPROCESSING
    # md5 sums
    gdown https://drive.google.com/uc?id=1tmMgLwoBvbEJEHXh77sqrXYw5RpqT8R_
    # processed chunks
    gdown https://drive.google.com/uc?id=14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_
    # unpack results and verify md5sums
    tar -xzf results_text.tar.gz && (cd results4 && md5sum --check ../bert_reference_results_text_md5.txt)

}

function download_bert_inference_data() {
    mkdir -p $SQUAD_DATA
    cd $SQUAD_DATA
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

}

function prepare_trainint_data() {
    cd $BERT_INPUT_PREPROCESSING

    # For phase1 the seq_len=128:
    export SEQ_LEN=128
    ./parallel_create_hdf5.sh
    python3 ./chop_hdf5_files.py
    mv 2048_shards_uncompressed_128 $BERT_DATA/

    # For phase2 the seq_len=512:
    export SEQ_LEN=512
    ./parallel_create_hdf5.sh
    python3 ./chop_hdf5_files.py
    mv 2048_shards_uncompressed_512 $BERT_DATA/
}

function prepare_training_model() {
    cd $BERT_INPUT_PREPROCESSING
    wget https://raw.githubusercontent.com/mlcommons/training_results_v2.1/main/Intel/benchmarks/bert/implementations/pytorch-cpu/convert_checkpoint_tf2torch.py
    wget https://raw.githubusercontent.com/mlcommons/training_results_v2.1/main/Intel/benchmarks/bert/implementations/pytorch-cpu/modeling_bert_patched.py
    python convert_checkpoint_tf2torch.py --tf_checkpoint $BERT_MODEL/model.ckpt-28252 --bert_config_path $BERT_MODEL/bert_config.json --output_checkpoint $BERT_MODEL/model.ckpt-28252.pt

}

function prepare_inference_model() {
  mkdir -p $SQUAD_MODEL
  cd $BERT_MODEL
  wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
  wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
  wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt -O bert_squad_model/vocab.txt
}

function prepare_inference_libraries() {
  # Clone the Transformers repo in the BERT large inference directory
  cd ${QUICKSTART_HOME}/scripts/language_modeling/pytorch/bert_large/inference/cpu
  rm -rf transformers
  git clone https://github.com/huggingface/transformers.git
  cd transformers
  git checkout v4.18.0
  git apply ../enable_ipex_for_squad.diff
  pip install -e ./
}

if [ "${PHASE}" = "training" ]; then
    download_bert_training_data
    prepare_trainint_data
    download_bert_training_model
    prepare_training_model
elif [ "${PHASE}" = "inference" ]; then
    download_bert_inference_data
    prepare_inference_model
    prepare_inference_libraries
else
    usage
fi

move_to_workspace $BERT_HOME
