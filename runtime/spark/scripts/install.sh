#!/bin/bash

export HADOOP_VERSION=3.2.0
export SPARK_VERSION=3.1.1

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# install JDK
export JAVA_HOME=$RUNTIME_PATH/jdk

if [ ! -d "${JAVA_HOME}" ]; then
  (cd $RUNTIME_PATH && wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
      gunzip jdk-8u192-linux-x64.tar.gz && \
      tar -xf jdk-8u192-linux-x64.tar && \
      rm jdk-8u192-linux-x64.tar && \
      mv jdk1.8.0_192 jdk)
  echo "export JAVA_HOME=$JAVA_HOME">> ${USER_HOME}/.bashrc
  echo "export PATH=$JAVA_HOME/bin:$PATH" >> ${USER_HOME}/.bashrc
fi

# install Hadoop
export HADOOP_HOME=$RUNTIME_PATH/hadoop

if [ ! -d "${HADOOP_HOME}" ]; then
  (cd $RUNTIME_PATH && wget --quiet http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop.tar.gz && \
      tar -zxvf hadoop.tar.gz && \
      mv hadoop-${HADOOP_VERSION} hadoop && \
      rm hadoop.tar.gz)
  echo "export HADOOP_HOME=$HADOOP_HOME">> ${USER_HOME}/.bashrc
  echo "export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop">> ${USER_HOME}/.bashrc
  echo "export JAVA_HOME=$JAVA_HOME" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
  echo "export PATH=$HADOOP_HOME/bin:$PATH" >> ${USER_HOME}/.bashrc
fi
  
# install Spark
export SPARK_HOME=$RUNTIME_PATH/spark 

if [ ! -d "${SPARK_HOME}" ]; then
 (cd $RUNTIME_PATH && wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz)
  echo "export SPARK_HOME=$SPARK_HOME">> ${USER_HOME}/.bashrc
  echo "export PATH=$SPARK_HOME/bin:$PATH" >> ${USER_HOME}/.bashrc
fi
