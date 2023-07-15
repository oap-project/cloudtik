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

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${TRAINING_EPOCHS}" ]; then
  echo "The required environment variable TRAINING_EPOCHS has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, or bf16."
  exit 1
fi

ARGS=""
ARGS="$ARGS -a resnet50 ${DATASET_DIR}"

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "running bf32 path"
elif [[ $PRECISION == "fp16" ]]; then
    ARGS="$ARGS --fp16"
    echo "running fp16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32 bf16"
    exit 1
fi

if [[ "$USE_IPEX" == "true" ]]; then
  ARGS="$ARGS --ipex"
  export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
fi

BACKEND=${BACKEND:-'ccl'}

if [[ "$BACKEND" == "ccl" ]]; then
  oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
  source $oneccl_bindings_for_pytorch_path/env/setvars.sh
fi

CORES=${CORES:-`lscpu | grep Core | awk '{print $4}'`}
SOCKETS=${SOCKETS:-`lscpu | grep Socket | awk '{print $2}'`}
TOTAL_CORES=`expr $CORES \* $SOCKETS`
HOSTS=${HOSTS:-'127.0.0.1'}
NNODES=$(echo $HOSTS | tr ',' '\n' | wc -l)
NUM_RANKS=$(( NNODES * SOCKETS ))

CORES_PER_PROC=$CORES

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=128

rm -rf ${OUTPUT_DIR}/resnet50_dist_training_log_*
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${MODEL_DIR}/models/image_recognition/pytorch/common/main.py}

cloudtik-run \
    --memory-allocator=default \
    --distributed \
    --nnodes ${NNODES} \
    --hosts ${HOSTS} \
    --nproc_per_node ${SOCKETS} \
    --ncores-per-proc ${CORES_PER_PROC} \
    ${TRAIN_SCRIPT} \
    $ARGS \
    -j 0 \
    --seed 2020 \
    --epochs $TRAINING_EPOCHS \
    --world-size ${NUM_RANKS} \
    --dist-backend $BACKEND \
    --train-no-eval \
    -b $BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/resnet50_dist_training_log_${PRECISION}.log
# For the summary of results
wait

throughput=$(grep 'Training throughput:' ${OUTPUT_DIR}/resnet50_dist_training_log_${PRECISION}.log |sed -e 's/.*Training throughput//;s/[^0-9.]//g' |awk '
BEGIN {
       sum = 0;
i = 0;
      }
      {
       sum = sum + $1;
i++;
      }
END   {
       sum = sum / i;
       printf("%.3f", sum);
}')

echo "resnet50;"training distributed throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
