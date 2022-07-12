#!/bin/bash

args=$(getopt -a -o w::r::h:: -l workload::,repository::,help:: -- "$@")
eval set -- "${args}"

WORKLOAD=all
REPOSITORY=default

function prepare_prerequisite() {
    source ~/.bashrc
    sudo apt-get install -y git
    export USER_HOME=/home/$(whoami)
    BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools
    mkdir -p $BENCHMARK_TOOL_HOME
    sudo chown $(whoami) $BENCHMARK_TOOL_HOME
    sudo apt-get update
}

function install_sbt() {
    sudo apt-get update
    sudo apt-get install apt-transport-https curl gnupg -yqq
    echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo -H gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/scalasbt-release.gpg --import
    sudo chmod 644 /etc/apt/trusted.gpg.d/scalasbt-release.gpg
    sudo apt-get update
    sudo apt-get install sbt -y

    if [ "${REPOSITORY}" == "china" ]; then
        use_sbt_china_repositories
    fi
}

function use_sbt_china_repositories() {
    sudo mkdir -p ~/.sbt
    sudo chown $(whoami) ~/.sbt
    tee > ~/.sbt/repositories << EOF
[repositories]
  local
  huaweicloud-ivy: https://mirrors.huaweicloud.com/repository/ivy/, [organization]/[module]/(scala_[scalaVersion]/)(sbt_[sbtVersion]/)[revision]/[type]s/[artifact](-[classifier]).[ext]
  huaweicloud-maven: https://mirrors.huaweicloud.com/repository/maven/
  bintray-typesafe-ivy: https://dl.bintray.com/typesafe/ivy-releases/, [organization]/[module]/(scala_[scalaVersion]/)(sbt_[sbtVersion]/)[revision]/[type]s/[artifact](-[classifier]).[ext]
  bintray-sbt-plugins: https://dl.bintray.com/sbt/sbt-plugin-releases/, [organization]/[module]/(scala_[scalaVersion]/)(sbt_[sbtVersion]/)[revision]/[type]s/[artifact](-[classifier]).[ext], bootOnly
EOF
}

function install_maven() {
    sudo apt install maven -y
}

function install_spark_sql_perf() {
    install_sbt
    cd ${BENCHMARK_TOOL_HOME}
    if [ ! -d "spark-sql-perf" ]; then
        git clone https://github.com/databricks/spark-sql-perf.git && cd spark-sql-perf
    else
        cd spark-sql-perf && git pull
    fi
    sbt package
}

function install_tpcds_kit() {
    sudo apt-get install -y gcc make flex bison byacc git
    cd ${BENCHMARK_TOOL_HOME} && git clone https://github.com/databricks/tpcds-kit.git
    cd tpcds-kit/tools
    make OS=LINUX
}

function install_tpch_dbgen() {
    sudo apt-get install -y make patch unzip
    cd ${BENCHMARK_TOOL_HOME} && git clone https://github.com/databricks/tpch-dbgen
    cd tpch-dbgen && make clean && make;
}

function install_hibench() {
    which bc > /dev/null || sudo apt-get install bc -y
    install_maven
    cd ${BENCHMARK_TOOL_HOME}
    if [ ! -d "HiBench" ]; then
        git clone https://github.com/Intel-bigdata/HiBench.git && cd HiBench
    else
        cd HiBench && git pull
    fi
    mvn -Psparkbench -Dmodules -Pml -Pmicro -Dspark=3.0 -Dscala=2.12 -DskipTests clean package
}

function install_tpcds() {
    install_spark_sql_perf
    install_tpcds_kit
}

function install_tpch() {
    install_spark_sql_perf
    install_tpch_dbgen
}

function usage() {
    echo "Usage: $0 --workload=[all|tpch|tpcds|hibench] --repository=[default|china]" >&2
    echo "Usage: $0 -h|--help"
}

while true
do
    case "$1" in
    -r|--repository)
        REPOSITORY=$2
        shift
        ;;
    -w|--workload)
        WORKLOAD=$2
        shift
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

prepare_prerequisite

if [ "${WORKLOAD}" == "tpcds" ];then
    install_tpcds
elif [ "${WORKLOAD}" == "tpch" ];then
    install_tpch
elif [ "${WORKLOAD}" == "hibench" ];then
    install_hibench
elif [ "${WORKLOAD}" == "all" ];then
    install_hibench
    install_tpcds
    install_tpch
else
    usage
    exit 1
fi
