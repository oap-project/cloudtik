#!/bin/bash

function install_fuse_dfs() {
    if [ -z "${HADOOP_VERSION}" ]; then
        HADOOP_VERSION=3.3.1
    fi
    sudo wget  -q --show-progress https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/fuse_dfs-${HADOOP_VERSION} -O /usr/bin/fuse_dfs
    sudo wget  -q --show-progress https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/fuse_dfs_wrapper-${HADOOP_VERSION}.sh -O /usr/bin/fuse_dfs_wrapper.sh
    sudo chmod +x /usr/bin/fuse_dfs
    sudo chmod +x /usr/bin/fuse_dfs_wrapper.sh
}

function install_hadoop() {
    # Install Hadoop
    if [ -z "${USER_HOME}" ]; then
        USER_HOME=/home/$(whoami)
    fi
    if [ -z "${RUNTIME_PATH}" ]; then
        RUNTIME_PATH=$USER_HOME/runtime
    fi
    if [ -z "${HADOOP_VERSION}" ]; then
        HADOOP_VERSION=3.3.1
    fi
    export HADOOP_HOME=$RUNTIME_PATH/hadoop

    if [ ! -d "${HADOOP_HOME}" ]; then
        arch=$(uname -m)
        if [ "${arch}" == "aarch64" ]; then
            arch_hadoop="-aarch64"
        else
            arch_hadoop=""
        fi
        hadoop_download_url="http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}${arch_hadoop}.tar.gz"

        (cd $RUNTIME_PATH && wget -q --show-progress ${hadoop_download_url} -O hadoop.tar.gz && \
            mkdir -p "$HADOOP_HOME" && \
            tar --extract --file hadoop.tar.gz --directory "$HADOOP_HOME" --strip-components 1 --no-same-owner && \
            rm hadoop.tar.gz && \
            wget -q --show-progress https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-azure-${HADOOP_VERSION}.jar -O $HADOOP_HOME/share/hadoop/tools/lib/hadoop-azure-${HADOOP_VERSION}.jar)
        echo "export HADOOP_HOME=$HADOOP_HOME">> ${USER_HOME}/.bashrc
        echo "export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop">> ${USER_HOME}/.bashrc
        echo "export PATH=\$HADOOP_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
        echo "export JAVA_HOME=$JAVA_HOME" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
        #Add share/hadoop/tools/lib/* into classpath
        echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
    fi
    install_fuse_dfs
}
