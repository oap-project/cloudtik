#!/bin/bash

args=$(getopt -a -o a:s:i::h -l action:,cluster_config:,workspace_config:,scale_factor:,iteration::,aws_access_key_id::,aws_secret_access_key::,baseline,help, -- "$@")
eval set -- "${args}"

ITERATION=1
BASELINE=false

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

function check_cloudtik_environment() {
    if [ ! -d "${CLOUDTIK_HOME}" ]; then
        echo "Please define CLOUDTIK_HOME to CloudTik source root so that we can use the tpc-ds scripts to generate data or run power test."
        exit 1
    fi
    which cloudtik > /dev/null || (echo "CloudTik is not found. Please install CloudTik first!"; exit 1)
}

function check_aws_benchmark_action() {
    BENCHMARL_ALLOW_ACTIONS=( generate-data power-test  )
    if [ $(contains "${BENCHMARL_ALLOW_ACTIONS[@]}" "$ACTION") == "y" ]; then
        echo "Action $ACTION is allowed for this aws benchmark script."
    else
        echo "Action $ACTION is not allowed for this aws benchmark script. Supported action: ${BENCHMARL_ALLOW_ACTIONS[*]}."
        exit 1
    fi
}

function check_aws_benchmark_config() {
    if [ -f "${CLUSTER_CONFIG}" ]; then
         echo "Found the cluster config file ${CLUSTER_CONFIG}"
    else
         echo "The cluster config file ${CLUSTER_CONFIG} doesn't exist"
    fi

    if [ -f "${WORKSPACE_CONFIG}" ]; then
         echo "Found the workspace config file ${WORKSPACE_CONFIG}"
    else
         echo "The workspace config file ${WORKSPACE_CONFIG} doesn't exist"
    fi
}

function get_workspace_managed_storage_uri() {
    MANAGED_STORAGE_URI=$(cloudtik workspace info ${WORKSPACE_CONFIG} --managed-storage-uri)
}

function generate_tpcds_data() {
    cloudtik submit $CLUSTER_CONFIG $CLOUDTIK_HOME/tools/benchmarks/spark/scripts/tpcds-datagen.scala \
        --conf spark.driver.scaleFactor=${SCALE_FACTOR} \
        --conf spark.driver.fsdir="${MANAGED_STORAGE_URI}" \
        --jars '$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar'
}

function run_tpcds_power_test_with_vanilla_spark() {
    cloudtik submit $CLUSTER_CONFIG $CLOUDTIK_HOME/tools/benchmarks/spark/scripts/tpcds-power-test.scala \
        --conf spark.driver.scaleFactor=${SCALE_FACTOR} \
        --conf spark.driver.fsdir="${MANAGED_STORAGE_URI}" \
        --conf spark.driver.iterations=${ITERATION} \
        --conf spark.driver.useArrow=false \
        --jars '$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar' \
        --num-executors 24 \
        --driver-memory 20g \
        --executor-cores 8 \
        --executor-memory 13g \
        --conf spark.executor.memoryOverhead=1024 \
        --conf spark.memory.offHeap.enabled=true \
        --conf spark.memory.offHeap.size=10g \
        --conf spark.dynamicAllocation.enabled=false \
        --conf spark.sql.shuffle.partitions=384
}

function run_tpcds_power_test_with_gazelle() {
    cloudtik submit $CLUSTER_CONFIG $CLOUDTIK_HOME/tools/benchmarks/spark/scripts/tpcds-power-test.scala \
        --conf spark.driver.scaleFactor=${SCALE_FACTOR} \
        --conf spark.driver.fsdir="${MANAGED_STORAGE_URI}" \
        --conf spark.driver.iterations=${ITERATION} \
        --conf spark.driver.useArrow=true \
        --jars '$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar' \
        --num-executors 24 \
        --driver-memory 20g \
        --executor-cores 8 \
        --executor-memory 8g \
        --conf spark.executor.memoryOverhead=384 \
        --conf spark.memory.offHeap.enabled=true \
        --conf spark.memory.offHeap.size=15g \
        --conf spark.dynamicAllocation.enabled=false \
        --conf spark.executorEnv.CC='$HOME/runtime/oap/bin/x86_64-conda_cos6-linux-gnu-cc' \
        --conf spark.yarn.appMasterEnv.CC='$HOME/runtime/oap/bin/x86_64-conda_cos6-linux-gnu-cc' \
        --conf spark.plugins=com.intel.oap.GazellePlugin \
        --conf spark.executorEnv.LD_LIBRARY_PATH='$HOME/runtime/oap/lib/' \
        --conf spark.executorEnv.LIBARROW_DIR='$HOME/runtime/oap/' \
        --conf spark.driver.extraClassPath='$HOME/runtime/oap/oap_jars/gazelle-plugin-1.4.0-spark-3.2.1.jar' \
        --conf spark.executor.extraClassPath='$HOME/runtime/oap/oap_jars/gazelle-plugin-1.4.0-spark-3.2.1.jar' \
        --conf spark.shuffle.manager=org.apache.spark.shuffle.sort.ColumnarShuffleManager \
        --conf spark.sql.join.preferSortMergeJoin=false \
        --conf spark.sql.inMemoryColumnarStorage.batchSize=20480 \
        --conf spark.sql.execution.arrow.maxRecordsPerBatch=20480 \
        --conf spark.sql.parquet.columnarReaderBatchSize=20480 \
        --conf spark.sql.autoBroadcastJoinThreshold=10M \
        --conf spark.sql.broadcastTimeout=600 \
        --conf spark.sql.crossJoin.enabled=true \
        --conf spark.driver.maxResultSize=20g \
        --conf spark.sql.columnar.window=true \
        --conf spark.sql.columnar.sort=true \
        --conf spark.sql.codegen.wholeStage=true \
        --conf spark.sql.columnar.codegen.hashAggregate=false \
        --conf spark.sql.shuffle.partitions=384 \
        --conf spark.kryoserializer.buffer.max=128m \
        --conf spark.kryoserializer.buffer=32m \
        --conf spark.oap.sql.columnar.preferColumnar=false \
        --conf spark.oap.sql.columnar.sortmergejoin.lazyread=true \
        --conf spark.oap.sql.columnar.sortmergejoin=true \
        --conf spark.sql.execution.sort.spillThreshold=2147483648 \
        --conf spark.executorEnv.MALLOC_CONF=background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,narenas:2 \
        --conf spark.executorEnv.MALLOC_ARENA_MAX=2 \
        --conf spark.oap.sql.columnar.numaBinding=true \
        --conf spark.oap.sql.columnar.coreRange="0-15,32-47|16-31,48-63" \
        --conf spark.oap.sql.columnar.joinOptimizationLevel=18 \
        --conf spark.oap.sql.columnar.shuffle.customizedCompression.codec=lz4 \
        --conf spark.executorEnv.ARROW_ENABLE_NULL_CHECK_FOR_GET=false \
        --conf spark.executorEnv.ARROW_ENABLE_UNSAFE_MEMORY_ACCESS=true \
        --conf spark.executorEnv.AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
        --conf spark.executorEnv.AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
}

function usage() {
    echo "Usage for data generation : $0 -a|--action generate-data --cluster_config [your_cluster.yaml] --workspace_config [your_workspace.yaml] -s|--scale_factor [data scale] " >&2
    echo "Usage for tpc-ds power test with vanilla spark: $0 -a|--action power-test --cluster_config [your_cluster.yaml] --workspace_config [your_workspace.yaml] -s|--scale_factor [data scale] -i|--iteration=[default value is 1] --baseline" >&2
    echo "Usage for tpc-ds power test with gazelle: $0 -a|--action power-test --cluster_config [your_cluster.yaml] --workspace_config [your_workspace.yaml] -s|--scale_factor [data scale] -i|--iteration=[default value is 1] --aws_access_key_id=[key_id] --aws_secret_access_key=[key]" >&2
    echo "Usage: $0 -h|--help"
}


while true
do
    case "$1" in
    -a|--action)
        ACTION=$2
        shift
        ;;
    --cluster_config)
        CLUSTER_CONFIG=$2
        shift
        ;;
    --workspace_config)
        WORKSPACE_CONFIG=$2
        shift
        ;;
    -s|--scale_factor)
        SCALE_FACTOR=$2
        shift
        ;;
    -i|--iteration)
        ITERATION=$2
        shift
        ;;
    --aws_access_key_id)
        AWS_ACCESS_KEY_ID=$2
        shift
        ;;
    --aws_secret_access_key)
        AWS_SECRET_ACCESS_KEY=$2
        shift
        ;;
    --baseline)
        BASELINE=true
        ;;
    -h|--help)
        shift
        usage
        exit 0
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

check_cloudtik_environment
check_aws_benchmark_action
check_aws_benchmark_config
get_workspace_managed_storage_uri

if [ "${ACTION}" == "generate-data" ];then
    generate_tpcds_data
elif [ "${ACTION}" == "power-test" ];then
    if [ ${BASELINE} == "true" ]; then
        run_tpcds_power_test_with_vanilla_spark
    else
        run_tpcds_power_test_with_gazelle
    fi
else
    usage
    exit 1
fi