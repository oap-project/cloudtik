#!/bin/bash

ACTIVATE_ENV=cloudtik_py37

USER_HOME=/home/$(whoami)
BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools
MLPERF_HOME=$BENCHMARK_TOOL_HOME/mlperf

BERT_MLPERF_HOME=$MLPERF_HOME/bert

BERT_DATA_PATH=$BERT_MLPERF_HOME/data
BERT_SCRIPT_DIR=$BERT_MLPERF_HOME/implementations/train/pytorch-cpu/input_preprocessing

args=$(getopt -a -o o: -l outputdir: -- "$@")
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
    source activate ${ACTIVATE_ENV}
    pip -qq install gdown transformers tensorflow
}

function prepare_data() {
  source activate ${ACTIVATE_ENV}
  cd ${BERT_SCRIPT_DIR}
  chmod +x ${BERT_SCRIPT_DIR}/*
  bash ${BERT_SCRIPT_DIR}/prepare_data.sh --outputdir ${BERT_DATA_PATH}
}

install_libaries
prepare_data
