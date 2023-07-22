#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

if [ -z "$SPARK_VERSION" ]; then
    # if SPARK_VERSION is not set, set a default Spark version
    export SPARK_VERSION=3.2.1
fi

# Set Hadoop version based on Spark version
export HADOOP_VERSION=3.3.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

# JDK install function
. "$ROOT_DIR"/common/scripts/jdk-install.sh

# Hadoop install function
. "$ROOT_DIR"/common/scripts/hadoop-install.sh

# Cloud storage fuse functions
. "$ROOT_DIR"/common/scripts/cloud-storage-fuse.sh

function install_spark() {
    # install Spark
    export SPARK_HOME=$RUNTIME_PATH/spark

    if [ ! -d "${SPARK_HOME}" ]; then
     (cd $RUNTIME_PATH && wget -q --show-progress https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz -O spark.tgz && \
        mkdir -p "$SPARK_HOME" && \
        tar --extract --file spark.tgz --directory "$SPARK_HOME" --strip-components 1 --no-same-owner && \
        ln -rs $SPARK_HOME/examples/jars/spark-examples_*.jar $SPARK_HOME/examples/jars/spark-examples.jar && \
        rm spark.tgz)
        echo "export SPARK_HOME=$SPARK_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
        # Config for PySpark when Spark installed
        echo "export PYTHONPATH=\${SPARK_HOME}/python:\${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip" >> ~/.bashrc
        echo "export PYSPARK_PYTHON=\${CONDA_ROOT}/envs/\${CLOUDTIK_ENV}/bin/python" >> ~/.bashrc
        echo "export PYSPARK_DRIVER_PYTHON=\${CONDA_ROOT}/envs/\${CLOUDTIK_ENV}/bin/python" >> ~/.bashrc
    fi

    if [ "$METASTORE_ENABLED" == "true" ] && [ "$HIVE_FOR_METASTORE_JARS" == "true" ] && [ $IS_HEAD_NODE == "true" ]; then
        # To be improved: we may need to install Hive anyway
        # Spark Hive Metastore nees quit some Hive dependencies
        # "hive-metastore", "hive-exec", "hive-common", "hive-serde"
        # org.apache.hadoop:hadoop-client
        # com.google.guava:guava
        # So we download Hive instead
        export HIVE_HOME=$RUNTIME_PATH/hive
        export HIVE_VERSION=3.1.2
        if [ ! -d "${HIVE_HOME}" ]; then
         (cd $RUNTIME_PATH && wget -q --show-progress https://downloads.apache.org/hive/hive-${HIVE_VERSION}/apache-hive-${HIVE_VERSION}-bin.tar.gz -O hive.tar.gz && \
            mkdir -p "$HIVE_HOME" && \
            tar --extract --file hive.tar.gz --directory "$HIVE_HOME" --strip-components 1 --no-same-owner && \
            rm hive.tar.gz)
            echo "export HIVE_HOME=$HIVE_HOME">> ${USER_HOME}/.bashrc
        fi
    fi
}

function install_jupyter_for_spark() {
    if [ $IS_HEAD_NODE == "true" ];then
        # Install Jupyter and spylon-kernel for Spark
        if ! type jupyter >/dev/null 2>&1; then
          echo "Install JupyterLab..."
          pip --no-cache-dir -qq install jupyter_server==1.19.1 jupyterlab==3.4.3
        fi

        export SPYLON_KERNEL=$USER_HOME/.local/share/jupyter/kernels/spylon-kernel

        if  [ ! -d "${SPYLON_KERNEL}" ]; then
            pip --no-cache-dir -qq install spylon-kernel==0.4.1;
            python -m spylon_kernel install --user;
        fi

        # Creating the jupyter data folders
        mkdir -p $RUNTIME_PATH/jupyter
    fi
}

function install_tools() {
    which jq > /dev/null || (sudo  apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install jq -y > /dev/null)
    which vim > /dev/null || (sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install vim -y > /dev/null)
}

function install_yarn_with_spark_jars() {
    # Copy spark jars to hadoop path
    jars=('spark-[0-9]*[0-9]-yarn-shuffle.jar' 'jackson-databind-[0-9]*[0-9].jar' 'jackson-core-[0-9]*[0-9].jar' 'jackson-annotations-[0-9]*[0-9].jar' 'metrics-core-[0-9]*[0-9].jar' 'netty-all-[0-9]*[0-9].Final.jar' 'commons-lang3-[0-9]*[0-9].jar')
    find ${HADOOP_HOME}/share/hadoop/yarn/lib -name netty-all-[0-9]*[0-9].Final.jar| xargs -i mv -f {} {}.old
    for jar in ${jars[@]};
    do
	    find ${SPARK_HOME}/jars/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib;
	    find ${SPARK_HOME}/yarn/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib;
    done
}

function download_hadoop_cloud_jars() {
    HADOOP_TOOLS_LIB=${HADOOP_HOME}/share/hadoop/tools/lib
    HADOOP_HDFS_LIB=${HADOOP_HOME}/share/hadoop/hdfs/lib

    GCS_HADOOP_CONNECTOR="gcs-connector-hadoop3-latest.jar"
    if [ ! -f "${HADOOP_TOOLS_LIB}/${GCS_HADOOP_CONNECTOR}" ]; then
        # Download gcs-connector to ${HADOOP_HOME}/share/hadoop/tools/lib/* for gcp cloud storage support
        wget -q -nc -P "${HADOOP_TOOLS_LIB}"  https://storage.googleapis.com/hadoop-lib/gcs/${GCS_HADOOP_CONNECTOR}
    fi

    # Copy Jetty Utility jars from HADOOP_HDFS_LIB to HADOOP_TOOLS_LIB for Azure cloud storage support
    JETTY_UTIL_JARS=('jetty-util-ajax-[0-9]*[0-9].v[0-9]*[0-9].jar' 'jetty-util-[0-9]*[0-9].v[0-9]*[0-9].jar')
    for jar in ${JETTY_UTIL_JARS[@]};
    do
	    find "${HADOOP_HDFS_LIB}" -name $jar | xargs -i cp {} "${HADOOP_TOOLS_LIB}";
    done
}

function download_spark_cloud_jars() {
    SPARK_JARS=${SPARK_HOME}/jars
    SPARK_HADOOP_CLOUD_JAR="spark-hadoop-cloud_2.12-${SPARK_VERSION}.jar"
    if [ ! -f "${SPARK_JARS}/${SPARK_HADOOP_CLOUD_JAR}" ]; then
        wget -q -nc -P "${SPARK_JARS}" https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.12/${SPARK_VERSION}/${SPARK_HADOOP_CLOUD_JAR}
    fi
}

function install_hadoop_with_cloud_jars() {
    # Download jars are possible long running tasks and should be done on install step instead of configure step.
    download_hadoop_cloud_jars
}

function install_spark_with_cloud_jars() {
    download_spark_cloud_jars

    # Copy cloud storage jars of different cloud providers to Spark classpath
    cloud_storge_jars=( \
        'hadoop-aws-[0-9]*[0-9].jar' \
        'aws-java-sdk-bundle-[0-9]*[0-9].jar' \
        'gcs-connector-hadoop3-*.jar' \
        'hadoop-azure-[0-9]*[0-9].jar' \
        'azure-storage-[0-9]*[0-9].jar' \
        'hadoop-aliyun-[0-9]*[0-9].jar' \
        'aliyun-java-sdk-*.jar' \
        'jdom-*.jar' \
        'jettison-*.jar' \
        'aliyun-sdk-oss-*.jar' \
        'hadoop-huaweicloud-[0-9]*[0-9].jar' \
        'wildfly-openssl-[0-9]*[0-9].Final.jar' \
        'jetty-util-ajax-[0-9]*[0-9].v[0-9]*[0-9].jar' \
        'jetty-util-[0-9]*[0-9].v[0-9]*[0-9].jar' \
        )
    for jar in ${cloud_storge_jars[@]};
    do
	    find "${HADOOP_HOME}"/share/hadoop/tools/lib/ -name $jar | xargs -i cp {} "${SPARK_HOME}"/jars;
	    find "${HADOOP_HOME}"/share/hadoop/common/lib/ -name $jar | xargs -i cp {} "${SPARK_HOME}"/jars;
    done
}

set_head_option "$@"
install_jdk
install_hadoop
install_spark
install_jupyter_for_spark
install_tools
install_yarn_with_spark_jars
install_hadoop_with_cloud_jars
install_spark_with_cloud_jars
install_cloud_fuse
clean_install_cache
