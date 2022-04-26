#!/bin/bash

args=$(getopt -a -o h::p: -l head::,provider: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    -p|--provider)
        PROVIDER=$2
        shift
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

export HADOOP_VERSION=3.3.1
export SPARK_VERSION=3.2.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

function install_jdk() {
    # install JDK
    export JAVA_HOME=$RUNTIME_PATH/jdk

    if [ ! -d "${JAVA_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
          gunzip jdk-8u192-linux-x64.tar.gz && \
          tar -xf jdk-8u192-linux-x64.tar && \
          rm jdk-8u192-linux-x64.tar && \
          mv jdk1.8.0_192 jdk)
      echo "export JAVA_HOME=$JAVA_HOME">> ${USER_HOME}/.bashrc
      echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

function install_hadoop() {
    # install Hadoop
    export HADOOP_HOME=$RUNTIME_PATH/hadoop

    if [ ! -d "${HADOOP_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop-${HADOOP_VERSION}.tar.gz && \
          tar -zxf hadoop-${HADOOP_VERSION}.tar.gz && \
          mv hadoop-${HADOOP_VERSION} hadoop && \
          rm hadoop-${HADOOP_VERSION}.tar.gz)
      echo "export HADOOP_HOME=$HADOOP_HOME">> ${USER_HOME}/.bashrc
      echo "export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop">> ${USER_HOME}/.bashrc
      echo "export JAVA_HOME=$JAVA_HOME" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
      echo "export PATH=\$HADOOP_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

function install_spark() {
    # install Spark
    export SPARK_HOME=$RUNTIME_PATH/spark

    if [ ! -d "${SPARK_HOME}" ]; then
     (cd $RUNTIME_PATH && wget -q --show-progress https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
        tar -zxf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
        mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
        rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz)
      echo "export SPARK_HOME=$SPARK_HOME">> ${USER_HOME}/.bashrc
      echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

function install_jupyter_for_spark() {
    if [ $IS_HEAD_NODE == "true" ];then
        # Install Jupyter and spylon-kernel for Spark
        if ! type jupyter >/dev/null 2>&1; then
          echo "Install JupyterLab..."
          pip -qq install jupyterlab
        fi

        export SPYLON_KERNEL=$USER_HOME/.local/share/jupyter/kernels/spylon-kernel

        if  [ ! -d "${SPYLON_KERNEL}" ]; then
            pip -qq install spylon-kernel;
            python -m spylon_kernel install --user;
        fi
    fi
}

function install_tools() {
    which jq > /dev/null || sudo apt-get -qq update -y; sudo apt-get -qq install jq -y
    which vim > /dev/null || sudo apt-get -qq update -y; sudo apt-get -qq install vim -y
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
        wget -q --show-progress -nc -P "${HADOOP_TOOLS_LIB}"  https://storage.googleapis.com/hadoop-lib/gcs/${GCS_HADOOP_CONNECTOR}
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
        wget -q --show-progress -nc -P "${SPARK_JARS}"  https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.12/${SPARK_VERSION}/${SPARK_HADOOP_CLOUD_JAR}
    fi
}

function install_hadoop_with_cloud_jars() {
    # Download jars are possible long running tasks and should be done on install step instead of configure step.
    download_hadoop_cloud_jars

    #Add share/hadoop/tools/lib/* into classpath
    echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
}

function install_spark_with_cloud_jars() {
    download_spark_cloud_jars

    # Copy cloud storage jars of different cloud providers to Spark classpath
    cloud_storge_jars=('hadoop-aws-[0-9]*[0-9].jar' 'aws-java-sdk-bundle-[0-9]*[0-9].jar' 'hadoop-azure-[0-9]*[0-9].jar' 'azure-storage-[0-9]*[0-9].jar' 'wildfly-openssl-[0-9]*[0-9].Final.jar' 'jetty-util-ajax-[0-9]*[0-9].v[0-9]*[0-9].jar' 'jetty-util-[0-9]*[0-9].v[0-9]*[0-9].jar' 'gcs-connector-hadoop3-*.jar')
    for jar in ${cloud_storge_jars[@]};
    do
	    find "${HADOOP_HOME}"/share/hadoop/tools/lib/ -name $jar | xargs -i cp {} "${SPARK_HOME}"/jars;
    done
}

install_jdk
install_hadoop
install_spark
install_jupyter_for_spark
install_tools
install_yarn_with_spark_jars
install_hadoop_with_cloud_jars
install_spark_with_cloud_jars
