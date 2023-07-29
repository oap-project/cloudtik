#!/bin/bash

function install_jdk() {
    # Install JDK
    if [ -z "${USER_HOME}" ]; then
        USER_HOME=/home/$(whoami)
    fi
    if [ -z "${RUNTIME_PATH}" ]; then
        RUNTIME_PATH=$USER_HOME/runtime
    fi
    export JAVA_HOME=$RUNTIME_PATH/jdk

    if [ ! -d "${JAVA_HOME}" ]; then
        arch=$(uname -m)
        if [ "${arch}" == "aarch64" ]; then
            arch_jdk=${arch}
        else
            arch_jdk="x64"
        fi
        jdk_download_url="https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.16.1%2B1/OpenJDK11U-jdk_${arch_jdk}_linux_hotspot_11.0.16.1_1.tar.gz"

        mkdir -p $RUNTIME_PATH
        (cd $RUNTIME_PATH && wget -q --show-progress ${jdk_download_url} -O openjdk.tar.gz && \
            mkdir -p "$JAVA_HOME" && \
            tar --extract --file openjdk.tar.gz --directory "$JAVA_HOME" --strip-components 1 --no-same-owner && \
            rm openjdk.tar.gz)
        echo "export JAVA_HOME=$JAVA_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}
