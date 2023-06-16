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
source ${SCRIPT_DIR}/../configure.sh

export RESNET50_HOME=$QUICKSTART_WORKSPACE/resnet50
export RESNET50_MODEL=$RESNET50_HOME/model
export DATASET_DIR=$RESNET50_HOME/data
export OUTPUT_DIR=$RESNET50_HOME/output

mkdir -p $OUTPUT_DIR

TRAINING_EPOCHS=1
USE_IPEX=false
PRECISION=fp32
BACKEND=gloo

function usage(){
    echo "Usage: train-distributed.sh [ --training_epochs ] [ --ipex ] [ --precision fp32 | bf16 | bf32 ] [ --backend ccl | gloo ]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --training_epochs)
        # num for steps
        shift
        TRAINING_EPOCHS=$1
        ;;
    --ipex)
        shift
        USE_IPEX=true
        ;;
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

export TRAINING_EPOCHS=$TRAINING_EPOCHS
export USE_IPEX
export PRECISION
export BACKEND

export TRAIN_SCRIPT=${QUICKSTART_HOME}/models/image_recognition/pytorch/common/main.py

LOGICAL_CORES=$(cloudtik head info --cpus-per-worker)
export CORES=$(( LOGICAL_CORES / 2 ))
export HOSTS=$(cloudtik head worker-ips --separator "," --node-status up-to-date)
export SOCKETS=$(cloudtik head info --sockets-per-worker)

cd ${QUICKSTART_HOME}/scripts/image_recognition/pytorch/resnet50/training/cpu
bash training_dist.sh
