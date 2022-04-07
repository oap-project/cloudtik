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

export HADOOP_VERSION=3.2.0
export SPARK_VERSION=3.1.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

function install_jdk() {
    # install JDK
    export JAVA_HOME=$RUNTIME_PATH/jdk

    if [ ! -d "${JAVA_HOME}" ]; then
      (cd $RUNTIME_PATH && wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
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
      (cd $RUNTIME_PATH && wget http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop-${HADOOP_VERSION}.tar.gz && \
          tar -zxf hadoop-${HADOOP_VERSION}.tar.gz && \
          mv hadoop-${HADOOP_VERSION} hadoop && \
          rm hadoop-${HADOOP_VERSION}.tar.gz)
      echo "export HADOOP_HOME=$HADOOP_HOME">> ${USER_HOME}/.bashrc
      echo "export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop">> ${USER_HOME}/.bashrc
      echo "export JAVA_HOME=$JAVA_HOME" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
      echo "export PATH=\$HADOOP_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

function install_spark() {
    # install Spark
    export SPARK_HOME=$RUNTIME_PATH/spark

    if [ ! -d "${SPARK_HOME}" ]; then
     (cd $RUNTIME_PATH && wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
        tar -zxf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
        mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
        rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz)
      echo "export SPARK_HOME=$SPARK_HOME">> ${USER_HOME}/.bashrc
      echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

function install_jupyter_for_spark() {
    # Install Jupyter and spylon-kernel for Spark
    if ! type jupyter >/dev/null 2>&1; then
      echo "Install JupyterLab..."
      pip install jupyterlab
    fi

    export SPYLON_KERNEL=$USER_HOME/.local/share/jupyter/kernels/spylon-kernel

    if  [ ! -d "${SPYLON_KERNEL}" ]; then
        pip install spylon-kernel;
        python -m spylon_kernel install --user;
    fi
}

function install_tools() {
    which jq || sudo apt-get install jq -y
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

    GCS_HADOOP_CONNECTOR="gcs-connector-hadoop3-2.2.0.jar"
    if [ ! -f "${HADOOP_TOOLS_LIB}/${GCS_HADOOP_CONNECTOR}" ]; then
        # Download gcs-connector to ${HADOOP_HOME}/share/hadoop/tools/lib/* for gcp cloud storage support
        wget -nc -P "${HADOOP_TOOLS_LIB}"  https://storage.googleapis.com/hadoop-lib/gcs/${GCS_HADOOP_CONNECTOR}
    fi

    JETTY_UTIL_AJAX="jetty-util-ajax-9.3.24.v20180605.jar"
    if [ ! -f "${HADOOP_TOOLS_LIB}/${JETTY_UTIL_AJAX}" ]; then
        # Download jetty-util to ${HADOOP_HOME}/share/hadoop/tools/lib/* for Azure cloud storage support
        wget -nc -P "${HADOOP_TOOLS_LIB}"  https://repo1.maven.org/maven2/org/eclipse/jetty/jetty-util-ajax/9.3.24.v20180605/${JETTY_UTIL_AJAX}
    fi

    JETTY_UTIL="jetty-util-9.3.24.v20180605.jar"
    if [ ! -f "${HADOOP_TOOLS_LIB}/${JETTY_UTIL}" ]; then
        wget -nc -P "${HADOOP_TOOLS_LIB}"  https://repo1.maven.org/maven2/org/eclipse/jetty/jetty-util/9.3.24.v20180605/${JETTY_UTIL}
    fi
}

function install_hadoop_with_cloud_jars() {
    # Download jars are possible long running tasks and should be done on install step instead of configure step.
    download_hadoop_cloud_jars

    #Add share/hadoop/tools/lib/* into classpath
    echo "export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
}

function install_spark_with_cloud_jars() {
    # Copy cloud storage jars of different cloud providers to Spark classpath
    cloud_storge_jars=('hadoop-aws-[0-9]*[0-9].jar' 'aws-java-sdk-bundle-[0-9]*[0-9].jar' 'hadoop-azure-[0-9]*[0-9].jar' 'azure-storage-[0-9]*[0-9].jar' 'wildfly-openssl-[0-9]*[0-9].Final.jar' 'jetty-util-ajax-[0-9]*[0-9].v[0-9]*[0-9].jar' 'jetty-util-[0-9]*[0-9].v[0-9]*[0-9].jar' 'gcs-connector-hadoop3-[0-9]*[0-9].jar')
    for jar in ${cloud_storge_jars[@]};
    do
	    find "${HADOOP_HOME}"/share/hadoop/tools/lib/ -name $jar | xargs -i cp {} "${SPARK_HOME}"/jars;
    done
}

function install_ganglia_server() {
    # Simply do the install, if they are already installed, it doesn't take time
    sudo apt-get update -y
    sudo apt-get install -y apache2 php libapache2-mod-php php-common php-mbstring php-gmp php-curl php-intl php-xmlrpc php-zip php-gd php-mysql php-xml
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ganglia-monitor rrdtool gmetad ganglia-webfrontend
}

function install_ganglia_client() {
    sudo apt-get update -y
    sudo apt-get install -y ganglia-monitor
}

function install_ganglia() {
    if [ $IS_HEAD_NODE == "true" ];then
        install_ganglia_server
    else
        install_ganglia_client
    fi
}

install_jdk
install_hadoop
install_spark
install_jupyter_for_spark
install_tools
install_yarn_with_spark_jars
install_hadoop_with_cloud_jars
install_spark_with_cloud_jars
install_ganglia