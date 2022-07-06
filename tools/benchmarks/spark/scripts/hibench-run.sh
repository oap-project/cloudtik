#!/bin/bash

CURRENT_HOME=$(cd $(dirname ${BASH_SOURCE[0]});pwd)
HIBENCH_TOOL=$CURRENT_HOME/hibench-tool

which realpath > /dev/null || sudo apt-get install realpath

args=$(getopt -a -o a:w:h -l action:workload:,cluster_config:,workspace_config:,hibench_config_dir:,docker,managed_cloud_storage,help -- "$@")
eval set -- "${args}"


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
    which cloudtik > /dev/null || (echo "CloudTik is not found. Please install CloudTik first!"; exit 1)
}


function check_hibench_cloudtik_action() {
    HIBENCH_CLOUDTIK_ALLOW_ACTIONS=( update-config run generate-data )
    if [ $(contains "${HIBENCH_CLOUDTIK_ALLOW_ACTIONS[@]}" "$ACTION") == "y" ]; then
        echo "Action $ACTION is allowed for this hibench cloudtik script."
    else
        echo "Action $ACTION is not allowed for this hibench cloudtik script. Supported action: ${HIBENCH_CLOUDTIK_ALLOW_ACTIONS[*]}."
        exit 1
    fi
}


function check_cloudtik_cluster_config() {
    if [ -f "${CLUSTER_CONFIG}" ]; then
        echo "Found the cluster config file ${CLUSTER_CONFIG}"
    else
        echo "The cluster config file ${CLUSTER_CONFIG} doesn't exist"
	    exit 1
    fi
}

function check_cloudtik_workspace_config() {
    if [ -f "${WORKSPACE_CONFIG}" ]; then
        echo "Found the workspace config file ${WORKSPACE_CONFIG}"
    else
        echo "The workspace config file ${WORKSPACE_CONFIG} doesn't exist"
	    exit 1
    fi
}


function check_hibench_config_dir() {
    if [ -d "${HIBENCH_CONFIG_DIR}" ]; then
        echo "Found the hench config directory ${HIBENCH_CONFIG_DIR}"
    else
        echo "The hibench config directory ${HIBENCH_CONFIG_DIR} doesn't exist"
	    exit 1
    fi
}


function prepare_replace_conf_value() {
    if [ "$DOCKER_MODE" == "true" ]; then
        SPARK_HOME="/home/cloudtik/runtime/spark"
        HADOOP_HOME="/home/cloudtik/runtime/hadoop"
    else
        SPARK_HOME="/home/ubuntu/runtime/spark"
        HADOOP_HOME="/home/ubuntu/runtime/hadoop"
    fi

    if [ "$MANAGED_CLOUD_STORAGE" == "true" ]; then
        FS_DEFAULT_NAME=$(cloudtik workspace info ${WORKSPACE_CONFIG} --managed-storage-uri)
    else
        FS_DEFAULT_NAME="hdfs://$(cloudtik head-ip $CLUSTER_CONFIG):9000"
    fi

    HIBENCH_HADOOP_EXAMPLES_JAR=$(echo $(cloudtik exec "$CLUSTER_CONFIG" 'find $HADOOP_HOME  -name hadoop-mapreduce-examples-*.jar | grep -v /sources/'))
    HIBENCH_HADOOP_EXAMPLES_JAR=`echo ${HIBENCH_HADOOP_EXAMPLES_JAR//$'\015'}`
    HIBENCH_HADOOP_EXAMPLES_TEST_JAR=$(echo $(cloudtik exec "$CLUSTER_CONFIG" 'find $HADOOP_HOME  -name hadoop-mapreduce-client-jobclient-*tests.jar | grep -v /sources/'))
    HIBENCH_HADOOP_EXAMPLES_TEST_JAR=`echo ${HIBENCH_HADOOP_EXAMPLES_TEST_JAR//$'\015'}`

}


function update_hibench_config() {
    python $CURRENT_HOME/hibench_config_utils.py $HIBENCH_CONFIG_DIR
    HIBENCH_TMP_CONFIG_DIR=$HIBENCH_CONFIG_DIR/output/hibench
    if [ -d "${HIBENCH_TMP_CONFIG_DIR}" ]; then
        echo "Successfully generated hibench tmp config directory: ${HIBENCH_TMP_CONFIG_DIR}"
    else
        echo "Failed to generate hibench tmp config directory: ${HIBENCH_TMP_CONFIG_DIR}"
	    exit 1
    fi

    prepare_replace_conf_value

    cd $HIBENCH_TMP_CONFIG_DIR
    sed -i "s!{%spark.home%}!${SPARK_HOME}!g" `grep "{%spark.home%}" -rl ./`
    sed -i "s!{%hadoop.home%}!${HADOOP_HOME}!g" `grep "{%hadoop.home%}" -rl ./`
    sed -i "s!{%fs.default.name%}!${FS_DEFAULT_NAME}!g" `grep "{%fs.default.name%}" -rl ./`
    sed -i "s!{%hibench.hadoop.examples.jar%}!${HIBENCH_HADOOP_EXAMPLES_JAR}!g" `grep "{%hibench.hadoop.examples.jar%}" -rl ./`
    sed -i "s!{%hibench.hadoop.examples.test.jar%}!${HIBENCH_HADOOP_EXAMPLES_TEST_JAR}!g" `grep "{%hibench.hadoop.examples.test.jar%}" -rl ./`

    HIBENCH_BASIC_CONFS=( hadoop.conf spark.conf hibench.conf )
    for conf in $HIBENCH_TMP_CONFIG_DIR/*; do
        if [ $(contains "${HIBENCH_BASIC_CONFS[@]}" "$(basename  $conf)") == "y"  ]; then
            echo "Upload local conf: $conf to head node: runtime/benchmark-tools/HiBench/conf/$(basename  $conf)..."
            cloudtik rsync-up "$CLUSTER_CONFIG" "$conf"  "runtime/benchmark-tools/HiBench/conf/$(basename  $conf)"
        else
            remote_hibench_conf=$(cloudtik exec "$CLUSTER_CONFIG" "find runtime/benchmark-tools/HiBench/conf -name $(basename  $conf)")
            remote_hibench_conf=`echo ${remote_hibench_conf//$'\015'}`
            echo "Upload local conf: $conf to head node: $remote_hibench_conf..."
            cloudtik rsync-up "$CLUSTER_CONFIG" "$conf"  "$remote_hibench_conf"
        fi
    done

}


function hibench_generate_data() {
    cloudtik exec "$CLUSTER_CONFIG" "cd \$HOME/runtime/benchmark-tools/HiBench && bash bin/workloads/$WORKLOAD/prepare/prepare.sh"
}


function hibench_run_benchmark() {
    cloudtik exec "$CLUSTER_CONFIG" "cd \$HOME/runtime/benchmark-tools/HiBench && bash bin/workloads/$WORKLOAD/spark/run.sh"
}


function usage() {
    echo "Docker Mode: $0 -a|--action [update-config|run|generate-data] -w|--workload [ml/kmeans| ml/als| ml/bayes] --cluster_config [your_cluster.yaml] --workspace_config [your_workspace.yaml] --hibench_config_dir [your_hibench_config_dirl] -d" >&2
    echo "Host Mode: $0 -a|--action [update-config|run|generate-data] -w|--workload [ml/kmeans| ml/als| ml/bayes] --cluster_config [your_cluster.yaml] --workspace_config [your_workspace.yaml] --hibench_config_dir [your_hibench_config_dirl]" >&2
    echo "Docker Mode with managed_cloud_storage: $0 -a|--action [update-config|run|generate-data] -w|--workload [ml/kmeans| ml/als| ml/bayes] --cluster_config [your_cluster.yaml] --workspace_config [your_workspace.yaml] --hibench_config_dir [your_hibench_config_dirl] -d -managed_cloud_storage" >&2
    echo "Usage: $0 -h|--help"
}


while true
do
    case "$1" in
    -a|--action)
        ACTION=$2
        shift
        ;;
    -w|--workload)
        WORKLOAD=$2
        shift
        ;;
    --cluster_config)
        CLUSTER_CONFIG=$(realpath $2)
        shift
        ;;
    --workspace_config)
        WORKSPACE_CONFIG=$(realpath $2)
        shift
        ;;
    --hibench_config_dir)
        HIBENCH_CONFIG_DIR=$(realpath $2)
        shift
        ;;
    --docker)
        DOCKER_MODE=true
        ;;
    --managed_cloud_storage)
        MANAGED_CLOUD_STORAGE=true
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
check_cloudtik_cluster_config
check_cloudtik_workspace_config
check_hibench_cloudtik_action
check_hibench_config_dir

if [ "${ACTION}" == "update-config" ];then
    update_hibench_config
elif [ "${ACTION}" == "generate-data" ];then
    hibench_generate_data
elif [ "${ACTION}" == "run" ];then
    hibench_run_benchmark
else
    usage
    exit 1
fi




