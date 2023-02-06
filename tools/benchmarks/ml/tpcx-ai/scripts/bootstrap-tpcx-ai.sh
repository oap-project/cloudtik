#!/bin/bash

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools
TPCX_AI_HOME=$BENCHMARK_TOOL_HOME/tpcx-ai

while true
do
    case "$1" in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

function prepare() {
    source ~/.bashrc
    sudo apt-get update -y
    mkdir -p $BENCHMARK_TOOL_HOME
    sudo chown $(whoami) $BENCHMARK_TOOL_HOME
}

function install_jdk8() {
    mv $JAVA_HOME $(dirname $JAVA_HOME)/jdk_bak
    wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz -O /tmp/jdk-8u192-linux-x64.tar.gz && \
    tar -xvf /tmp/jdk-8u192-linux-x64.tar.gz -C /tmp && \
    mv /tmp/jdk1.8.0_192 $JAVA_HOME
    rm -rf /tmp/jdk-8u192-linux-x64.tar.gz
}

function check_and_install_jdk8() {
    jdk_major_version=$(java -version 2>&1 | head -1 | cut -d'"' -f2 | sed '/^1\./s///' | cut -d'.' -f1)
    if [ ${jdk_major_version} -gt 8 ]; then
        install_jdk8
    fi
}

function install_tools() {
    #Installl cmake, GCC 9.0
    sudo apt-get install cmake -y
    sudo apt-get install gcc-9 g++-9 -y
}

function install_libaries() {
    sudo apt-get update && sudo apt-get install openjdk-8-jdk -y
    sudo apt-get install libsndfile1 libsndfile-dev libxxf86vm1 libxxf86vm-dev libglvnd0 libgl-dev -y
}

function install_tpcx_ai_benchmark() {
    sudo apt-get install zip -y
    wget https://d30257nes7d4fq.cloudfront.net/downloads/tpcx-ai/tpcx-ai-tool-v1.0.2.zip -O /tmp/tpcx-ai-tool.zip
    unzip -o /tmp/tpcx-ai-tool.zip -d "$BENCHMARK_TOOL_HOME" && rm /tmp/tpcx-ai-tool.zip && mv $BENCHMARK_TOOL_HOME/tpcx-ai-v* $TPCX_AI_HOME
}

function download_tpcx_ai_files() {
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/confs/parallel-data-gen.sh.patch -O $TPCX_AI_HOME/tools/parallel-data-gen.sh
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/confs/parallel-data-load.sh.patch -O $TPCX_AI_HOME/tools/parallel-data-load.sh
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/confs/default-spark.yaml -O $TPCX_AI_HOME/driver/config/default-spark.yaml
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/confs/default-spark.yaml.template -O $TPCX_AI_HOME/driver/config/default-spark.yaml.template
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/scripts/configure_default_spark_yaml.py -O $TPCX_AI_HOME/configure_default_spark_yaml.py
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/confs/UseCase09.py.patch -O $TPCX_AI_HOME/workload/spark/pyspark/workload-pyspark/UseCase09.py
}

function configure_tpcx_ai_benchmark() {
    download_tpcx_ai_files
    is_head_node
    echo IS_EULA_ACCEPTED=true >> ${TPCX_AI_HOME}/lib/pdgf/Constants.properties
    if [ $IS_HEAD_NODE == "true" ]; then
        echo IS_EULA_ACCEPTED=true >> ${TPCX_AI_HOME}/data-gen/Constants.properties
        echo 'export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop' >>  $TPCX_AI_HOME/setenv.sh
        source $TPCX_AI_HOME/setenv.sh
        cd ${TPCX_AI_HOME} && bash ${TPCX_AI_HOME}/setup-spark.sh
    fi
    chmod u+x -R ${TPCX_AI_HOME}
}

function is_head_node() {
    if [ -n $IS_HEAD_NODE ]; then
        cloudtik head head-ip
        GET_HEAD_IP_CODE=$?
        if [ ${GET_HEAD_IP_CODE} -eq "0" ]; then
            IS_HEAD_NODE=true
        else
            IS_HEAD_NODE=false
        fi
    fi
}

prepare
install_tpcx_ai_benchmark
configure_tpcx_ai_benchmark
check_and_install_jdk8
install_tools
install_libaries
