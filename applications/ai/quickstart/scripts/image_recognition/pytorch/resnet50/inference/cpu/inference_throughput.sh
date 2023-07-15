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
    echo "From which the main.py exist at the: \${MODEL_DIR}/models/image_recognition/pytorch/common/main.py.py"
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

rm -rf ${OUTPUT_DIR}/resnet50_throughput_log_*

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
ARGS="$ARGS -e -a resnet50 ../ --dummy"

# default value, you can fine-tune it to get perfect performance.
BATCH_SIZE=112

if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    BATCH_SIZE=116
    echo "running int8 path"
    ARGS="$ARGS --int8 --configure-dir ${MODEL_DIR}/models/image_recognition/pytorch/common/resnet50_configure_sym.json"
elif [[ $PRECISION == "bf16" ]]; then
    BATCH_SIZE=68
    ARGS="$ARGS --bf16 --jit"
    echo "running bf16 path"
elif [[ $PRECISION == "bf32" ]]; then
    BATCH_SIZE=68
    ARGS="$ARGS --bf32 --jit"
    echo "running bf32 path"
elif [[ $PRECISION == "fp16" ]]; then
    BATCH_SIZE=68
    ARGS="$ARGS --fp16 --jit"
    echo "running fp16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    BATCH_SIZE=64
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
TOTAL_CORES=`expr $CORES \* $SOCKETS`

CORES_PER_PROC=$CORES

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

weight_sharing=false
if [ ${WEIGHT_SHAREING} ]; then
  echo "Running RN50 inference throughput with runtime extension enabled."
  weight_sharing=true
fi

if [ "$weight_sharing" = true ]; then
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    CORES_PER_PROC=$CORES
    PROCESSES=`expr $TOTAL_CORES / $CORES_PER_PROC`
    LAST_PROCESS=`expr $PROCESSES - 1`
    PROCESSES_PER_SOCKET=`expr $PROCESSES / $SOCKETS`

    BATCH_PER_STREAM=2
    CORES_PER_STREAM=1
    STREAMS_PER_PROCESS=`expr $CORES / $CORES_PER_STREAM`
    BATCH_SIZE=`expr $BATCH_PER_STREAM \* $STREAMS_PER_PROCESS`

    export OMP_NUM_THREADS=$CORES_PER_STREAM

    for i in $(seq 0 $LAST_PROCESS); do
        numa_node_i=`expr $i / $PROCESSES_PER_SOCKET`
        start_core_i=`expr $i \* $CORES_PER_PROC`
        end_core_i=`expr $start_core_i + $CORES_PER_PROC - 1`
        LOG_i=resnet50_throughput_log_${PRECISION}_${i}.log
        echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
        numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u \
            ${MODEL_DIR}/models/image_recognition/pytorch/common/main_runtime_extension.py \
            $ARGS \
            --seed 2020 \
            -j 0 \
            -b $BATCH_SIZE \
            --number-instance $STREAMS_PER_PROCESS \
            --use-multi-stream-module \
            --instance-number $i 2>&1 | tee $LOG_i &
    done
    wait

else
    cloudtik-run \
        --memory-allocator=default \
        --num-proc ${SOCKETS} \
        --ncores-per-proc ${CORES_PER_PROC} \
        --log-dir=${OUTPUT_DIR} \
        --log-file-prefix="./resnet50_throughput_log_${PRECISION}" \
        ${MODEL_DIR}/models/image_recognition/pytorch/common/main.py \
        $ARGS \
        --seed 2020 \
        -j 0 \
        -b $BATCH_SIZE
fi
wait

throughput=$(grep 'Throughput:'  ${OUTPUT_DIR}/resnet50_throughput_log_${PRECISION}_* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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

echo "resnet50;"throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
