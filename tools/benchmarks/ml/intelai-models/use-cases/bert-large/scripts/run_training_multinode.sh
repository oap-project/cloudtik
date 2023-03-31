#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../../common/scripts/setenv.sh

export BERT_LARGE_HOME=$INTELAI_MODELS_WORKSPACE/bert
export BERT_LARGE_MODEL=$BERT_LARGE_HOME/model
export DATASET_DIR=$BERT_LARGE_HOME/data
export OUTPUT_DIR=$BERT_LARGE_HOME/output
export BERT_MODEL_CONFIG=${BERT_LARGE_MODEL}/bert_config.json

mkdir -p $OUTPUT_DIR

PRECISION=fp32
BACKEND=gloo

function usage(){
    echo "Usage: run-training_multinode.sh  [ --precision fp32 | bf16 | bf32] [ --backend ccl | gloo] "
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --precision)
        shift
        PRECISION=$1
        ;;
    --backend)
        shift
        BACKEND=$1
        ;;
    *)
        usage
    esac
    shift
done

export PRECISION=$PRECISION
export BACKEND=$BACKEND

LOGICAL_CORES=$(cloudtik head info --cpus-per-worker)
export CORES=$(( LOGICAL_CORES / 2 ))
export HOSTS=$(cloudtik head worker-ips --separator "," --node-status up-to-date)
export SOCKETS=$(cloudtik head info --sockets-per-worker)

cd ${CLOUDTIK_MODELS_HOME}/quickstart/language_modeling/pytorch/bert_large/training/cpu

bash run_ddp_bert_pretrain_phase1.sh $PRECISION
