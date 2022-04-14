#!/bin/bash
sudo apt-get install -y git
export USER_HOME=/home/$(whoami)
BENCHMARK_TOOL_HOME=$USER_HOME/runtime/benchmark-tools
mkdir -p $BENCHMARK_TOOL_HOME
sudo chown $(whoami):$(whoami) $BENCHMARK_TOOL_HOME
sudo apt-get update

function install_sbt() {
    sudo apt-get update
    sudo apt-get install apt-transport-https curl gnupg -yqq
    echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo -H gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/scalasbt-release.gpg --import
    sudo chmod 644 /etc/apt/trusted.gpg.d/scalasbt-release.gpg
    sudo apt-get update
    sudo apt-get install sbt -y
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
    echo "Usage: $0 --all|--tpcds|--tpch|--hibench|-h|--help" >&2
}

while [[ $# -ge 0 ]]
do
    key="$1"
    case $key in
    "")
        shift 1
        echo "Start to deploy all benchmark for Spark Runtime ..."
        install_hibench
        install_tpcds
        install_tpch
        exit 0
        ;;
    --tpcds)
        shift 1
        install_tpcds
        exit 0
        ;;
    --tpch)
        shift 1
        install_tpch
        exit 0
        ;;
    --hibench)
        shift 1
        install_hibench
        exit 0
        ;;
    -h|--help)
        shift 1
        usage
        exit 0
        ;;
    *)    # unknown option
        echo "Unknown option"
        usage
        exit 1
        ;;
    esac
done