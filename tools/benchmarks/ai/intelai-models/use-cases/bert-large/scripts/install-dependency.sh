#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../../common/scripts/setenv.sh


function install_libraries() {
    pip install tqdm --upgrade
    pip install tensorboardX
    # Clone the Transformers repo in the BERT large inference directory
    cd ${MODELS_HOME}/quickstart/language_modeling/pytorch/bert_large/inference/cpu
    rm -rf transformers
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    git checkout v4.18.0
    git apply ../enable_ipex_for_squad.diff
    pip install -e ./
}

install_libraries
