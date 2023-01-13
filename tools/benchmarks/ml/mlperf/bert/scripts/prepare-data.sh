#!/bin/bash


ACTIVATE_ENV=gen_data

BENCHMARK_TOOL_HOME=/cloudtik/fs/benchmark-tools
MLPERF_HOME=$BENCHMARK_TOOL_HOME/mlperf

BERT_MLPERF_HOME=$MLPERF_HOME/bert

BERT_DATA_PATH=$BERT_MLPERF_HOME/data
BERT_SCRIPT_DIR=$BERT_MLPERF_HOME/implementations/train/pytorch-cpu/input_preprocessing


args=$(getopt -a -o h::p: -l outputdir:: -- "$@")
eval set -- "${args}"

while true
do
    case "$1" in
    --outputdir)
        BERT_DATA_PATH=$2
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

function install_libaries() {
    conda create -n ${ACTIVATE_ENV} python=3.8
    conda install -n ${ACTIVATE_ENV} h5py  -y
    source activate ${ACTIVATE_ENV}
    pip install gdown transformers tensorflow

}

function prepare_data() {
  source activate ${ACTIVATE_ENV}
  cd ${BERT_SCRIPT_DIR}
  bash ${BERT_SCRIPT_DIR}/prepare_data.sh --outputdir ${BERT_DATA_PATH}
}

install_libaries
prepare_data
