#!/usr/bin/env bash
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

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the main.py exist at the: \${MODEL_DIR}/models/image_recognition/pytorch/common/main.py"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, int8, avx-int8, or bf16."
  exit 1
fi

BATCH_SIZE=1

rm -rf ${OUTPUT_DIR}/resnet50_latency_log*

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
ARGS="$ARGS -e -a resnet50 ../ --dummy"

if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --int8 --configure-dir ${MODEL_DIR}/models/image_recognition/pytorch/common/resnet50_configure_sym.json"
elif [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16 --jit"
    echo "running bf16 path"
elif [[ $PRECISION == "bf32" ]]; then
    ARGS="$ARGS --bf32 --jit"
    echo "running bf32 path"
elif [[ $PRECISION == "fp16" ]]; then
    ARGS="$ARGS --fp16 --jit"
    echo "running fp16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    ARGS="$ARGS --jit"
    echo "running fp32 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, int8, and avx-int8"
    exit 1
fi

if [[ "$USE_IPEX" == "true" ]]; then
  ARGS="$ARGS --ipex"
  export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

CORES_PER_INSTANCE=4

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=$CORES_PER_INSTANCE

NUMBER_INSTANCE=`expr $CORES / $CORES_PER_INSTANCE`

cloudtik-run \
    --use_default_allocator \
    --ninstance ${SOCKETS} \
    --log_path=${OUTPUT_DIR} \
    --log_file_prefix="./resnet50_latency_log_${PRECISION}" \
    ${MODEL_DIR}/models/image_recognition/pytorch/common/main.py \
    $ARGS \
    -j 0 \
    -b $BATCH_SIZE \
    --weight-sharing \
    --number-instance $NUMBER_INSTANCE

wait

TOTAL_CORES=`expr $CORES \* $SOCKETS`
INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/resnet50_latency_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
BEGIN {
        sum = 0;
i = 0;
      }
      {
        sum = sum + $1;
i++;
      }
END   {
sum = sum / i * INSTANCES_PER_SOCKET;
        printf("%.2f", sum);
}')

p99_latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/resnet50_latency_log* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
    printf("%.3f ms", sum);
}')
echo "resnet50;"latency";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
echo "resnet50;"p99_latency";${PRECISION};${BATCH_SIZE};${p99_latency}" | tee -a ${OUTPUT_DIR}/summary.log
