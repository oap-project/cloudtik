#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

RNNT_HOME=$INTELAI_MODELS_WORKSPACE/rnn-t

RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

# Env vars
export OUTPUT_DIR=$RNNT_OUTPUT_DIR
export DATASET_DIR=$RNNT_DATASET_DIR
export CHECKPOINT_DIR=$RNNT_CHECKPOINT_DIR

mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR


#(fp32, bf16, bf32)
PRECISION=fp32
NUM_STEPS=100
USE_IPEX=0
BACKEND=gloo

function usage(){
    echo "Usage: run-training_multinode.sh  [ --precision fp32 | bf16 | bf32]  [--num-steps 100] [--use-ipex] [ --backend ccl | gloo] "
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
    --num-steps)
        # num for steps
        shift
        NUM_STEPS=$1
        ;;
    --use-ipex)
        # use ipex for better performance
        USE_IPEX=1
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

export USE_IPEX
export NUM_STEPS
export BACKEND

LOGICAL_CORES=$(cloudtik head info --cpus-per-worker)
export CORES=$(( LOGICAL_CORES / 2 ))
export HOSTS=$(cloudtik head worker-ips --separator "," --node-status up-to-date)
export SOCKETS=$(cloudtik head info --sockets-per-worker)

cd ${CLOUDTIK_MODELS_HOME}/quickstart/language_modeling/pytorch/rnnt/training/cpu
bash training_multinode.sh $PRECISION
