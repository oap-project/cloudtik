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

export RESNET50_HOME=$INTELAI_MODELS_WORKSPACE/resnet50
export RESNET50_MODEL=$RESNET50_HOME/model
export DATASET_DIR=$RESNET50_HOME/data
export OUTPUT_DIR=$RESNET50_HOME/output

mkdir -p $OUTPUT_DIR
PRECISION=fp32
BACKEND=gloo
TRAINING_EPOCHS=1
ENABLE_IPEX="false"
function usage(){
    echo "Usage: run-training_multinode.sh  [ --precision fp32 | bf16 | bf32] [ --backend ccl | gloo]  [--training_epochs] [ --ipex]"
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
    --training_epochs)
        # num for steps
        shift
        TRAINING_EPOCHS=$1
        ;;
    --ipex)
        shift
        ENABLE_IPEX="true"
        ;;
    *)
        usage
    esac
    shift
done

export PRECISION=$PRECISION
export BACKEND=$BACKEND
export TRAINING_EPOCHS=$TRAINING_EPOCHS
export ENABLE_IPEX=$ENABLE_IPEX
export TRAIN_SCRIPT=${CLOUDTIK_MODELS_HOME}/models/image_recognition/pytorch/common/main.py

LOGICAL_CORES=$(cloudtik head info --cpus-per-worker)
export CORES=$(( LOGICAL_CORES / 2 ))
export HOSTS=$(cloudtik head worker-ips --separator "," --node-status up-to-date)
export SOCKETS=$(cloudtik head info --sockets-per-worker)

cd ${CLOUDTIK_MODELS_HOME}/quickstart/image_recognition/pytorch/resnet50/training/cpu
bash training_dist.sh
