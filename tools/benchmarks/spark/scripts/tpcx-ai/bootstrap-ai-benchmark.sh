#!/bin/bash

function prepare_prerequisite() {
    source ~/.bashrc
    sudo apt-get update -y
    sudo apt-get install -y git
    export USER_HOME=/home/$(whoami)
    BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools
    TPCX_AI_HONE=$BENCHMARK_TOOL_HOME/tpcx-ai
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

function check_jdk8() {
    jdk_major_version=$(java -version 2>&1 | head -1 | cut -d'"' -f2 | sed '/^1\./s///' | cut -d'.' -f1)
    if [ ${jdk_major_version} -gt 8 ]; then
        install_jdk8
    fi
}

function install_tools() {
    #Installl cmake, GCC 9.0 OpenMPI 4.0+
    sudo apt-get install cmake
    sudo apt-get install gcc-9 g++-9
    sudo apt-get install openmpi-bin openmpi-common openmpi-doc libopenmpi-dev -y
}

function install_libaries() {
    sudo apt-get update && sudo apt-get install openjdk-8-jdk -y
    sudo apt-get install libsndfile1 libsndfile-dev libxxf86vm1 libxxf86vm-dev libglvnd-dev libgl1-mesa-dev -y
}

function install_python_libraries() {
    pip install  librosa==0.8
}

function install_tpcx_ai_benchmark() {
    wget https://d30257nes7d4fq.cloudfront.net/downloads/tpcx-ai/tpcx-ai-v1.0.2.tgz -O /tmp/tpcx-ai.tgz
    tar --extract --file tmp/tpcx-ai.tgz --directory "$TPCX_AI_HONE" --strip-components 1 --no-same-owner && \
    rm tmp/tpcx-ai.tgz
}

function download_tpcx_ai_files() {
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/scripts/tpcx-ai/parallel-data-gen.sh -O $TPCX_AI_HONE/tools/parallel-data-gen.sh
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/scripts/tpcx-ai/parallel-data-load.sh -O $TPCX_AI_HONE/tools/parallel-data-load.sh
    wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/confs/tpcx-ai/default-spark-yaml.yaml -O $TPCX_AI_HONE/driver/config/default-spark-yaml.yaml
}

function configure_tpcx_ai_benchmark() {
    is_head_node
    echo IS_EULA_ACCEPTED=true >> ${TPCx_AI_HOME_DIR}/lib/pdgf/Constants.properties
    if [ $IS_HEAD_NODE == "true" ]; then
        cloudtik head worker-ips > $TPCX_AI_HONE/nodes
        echo IS_EULA_ACCEPTED=true >> ${TPCx_AI_HOME_DIR}/data-gen/Constants.properties
        echo 'export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop' >>  $TPCX_AI_HONE/setenv.sh
        echo "source $TPCX_AI_HONE/setenv.sh" >> ~/.bashrc
        download_tpcx_ai_files
    fi
}

function is_head_node() {
    cloudtik head head-ip
    GET_HEAD_IP_CODE=$?
    if [ ${GET_HEAD_IP_CODE} -eq "0" ]; then
        IS_HEAD_NODE=true
    else
        IS_HEAD_NODE=false
    fi
}

prepare_prerequisite
check_jdk8
install_tools
install_libaries
install_python_libraries
install_tpcx_ai_benchmark
configure_tpcx_ai_benchmark