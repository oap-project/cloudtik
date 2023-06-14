#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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

export SSD_RESNET34_HOME=$QUICKSTART_WORKSPACE/ssd-resnet34
export SSD_RESNET34_MODEL=$SSD_RESNET34_HOME/model
export DATASET_DIR=$SSD_RESNET34_HOME/data
export OUTPUT_DIR=$SSD_RESNET34_HOME/output

export CHECKPOINT_DIR=$SSD_RESNET34_MODEL
mkdir -p $OUTPUT_DIR

USE_IPEX=false
PRECISION=fp32
BACKEND=gloo

function usage(){
    echo "Usage: run-training-distributed.sh [ --ipex ] [ --precision fp32 | bf16 | bf32 ] [ --backend ccl | gloo ]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
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

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

export USE_IPEX
export PRECISION
export BACKEND

LOGICAL_CORES=$(cloudtik head info --cpus-per-worker)
export CORES=$(( LOGICAL_CORES / 2 ))
export HOSTS=$(cloudtik head worker-ips --separator "," --node-status up-to-date)
export SOCKETS=$(cloudtik head info --sockets-per-worker)

cd ${QUICKSTART_HOME}/scripts/object_detection/pytorch/ssd-resnet34/training/cpu
bash throughput_dist.sh $PRECISION
