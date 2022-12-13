#!/bin/bash

#########################################
# Please ensure that your disk space is greater than 1TB,
# and the memory is recommended to be more than 128G.
#########################################

export USER_HOME=/home/$(whoami)

MLPERF_HOME=$USER_HOME/runtime/mlperf
BERT_MLPERF_HOME=$MLPERF_HOME/bert
BERT_CODE_PATH=$BERT_MLPERF_HOME/code

SCRIPT_DIR=$BERT_CODE_PATH/input_preprocessing

ACTIVATE_ENV=gen_data

BERT_DATA_PATH=$BERT_MLPERF_HOME/data

function prepare_necessary_scripts() {
    mkdir -p ${SCRIPT_DIR}

    nvidia_scripts_link_prefix="https://raw.githubusercontent.com/mlcommons/training_results_v2.0/main/NVIDIA/benchmarks/bert/implementations/pytorch/input_preprocessing"
    intel_scripts_link_prefix="https://raw.githubusercontent.com/mlcommons/training_results_v2.1/main/Intel/benchmarks/bert/implementations/pytorch-cpu"
    wget ${nvidia_scripts_link_prefix}/prepare_data.sh -O ${SCRIPT_DIR}/prepare_data.sh
    wget ${nvidia_scripts_link_prefix}/parallel_create_hdf5.sh -O ${SCRIPT_DIR}/parallel_create_hdf5.sh
    wget ${nvidia_scripts_link_prefix}/create_pretraining_data_wrapper.sh -O ${SCRIPT_DIR}/create_pretraining_data_wrapper.sh
    wget ${nvidia_scripts_link_prefix}/create_pretraining_data.py -O ${SCRIPT_DIR}/create_pretraining_data.py
    wget ${nvidia_scripts_link_prefix}/tokenization..py -O ${SCRIPT_DIR}/tokenization.py
    wget ${nvidia_scripts_link_prefix}/chop_hdf5_files.py -O ${SCRIPT_DIR}/chop_hdf5_files.py
    wget ${nvidia_scripts_link_prefix}/convert_fixed2variable.py -O ${SCRIPT_DIR}/convert_fixed2variable.py
    wget ${nvidia_scripts_link_prefix}/create_pretraining_data.py -O ${SCRIPT_DIR}/create_pretraining_data.py
    wget ${nvidia_scripts_link_prefix}/pick_eval_samples.py -O ${SCRIPT_DIR}/pick_eval_samples.py
    wget ${nvidia_scripts_link_prefix}/convert_fixed2variable.py -O ${SCRIPT_DIR}/convert_fixed2variable.py
    wget ${intel_scripts_link_prefix}/convert_checkpoint_tf2torch.py -O ${BERT_CODE_PATH}/convert_tf_checkpoint.py
    wget ${intel_scripts_link_prefix}/modeling_bert_patched.py -O ${BERT_CODE_PATH}/modeling_bert_patched.py

}

function install_libaries() {
    conda create -n ${ACTIVATE_ENV} python=3.8
    conda install -n ${ACTIVATE_ENV} h5py  -y
    source activate ${ACTIVATE_ENV}
    pip install gdown transformers tensorflow

}

function prepare_data() {
  source activate ${ACTIVATE_ENV}
  cd ${SCRIPT_DIR}
  bash ${SCRIPT_DIR}/prepare_data.sh --outputdir ${BERT_DATA_PATH}
}

install_libaries
prepare_necessary_scripts
prepare_data
