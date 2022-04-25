#!/bin/bash

args=$(getopt -a -o h::p: -l head:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

export PRESTO_VERSION=0.271.1

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

function install_tools() {
    which uuid > /dev/null || sudo apt-get -qq update -y; sudo apt-get -qq install uuid -y
}

function install_presto() {
    # install Presto
    export PRESTO_HOME=$RUNTIME_PATH/presto

    if [ ! -d "${PRESTO_HOME}" ]; then
        (cd $RUNTIME_PATH && wget -q --show-progress https://repo1.maven.org/maven2/com/facebook/presto/presto-server/${PRESTO_VERSION}/presto-server-${PRESTO_VERSION}.tar.gz && \
            tar -zxf presto-server-${PRESTO_VERSION}.tar.gz && \
            mv presto-server-${PRESTO_VERSION} presto && \
            rm presto-server-${PRESTO_VERSION}.tar.gz)

        if [ $IS_HEAD_NODE == "true" ]; then
            # Download presto cli on head
            (cd $RUNTIME_PATH && wget -q --show-progress https://repo1.maven.org/maven2/com/facebook/presto/presto-cli/${PRESTO_VERSION}/presto-cli-${PRESTO_VERSION}-executable.jar && \
            mv presto-cli-${PRESTO_VERSION}-executable.jar $PRESTO_HOME/bin/presto && \
            chmod +x $PRESTO_HOME/bin/presto)

            echo "export PRESTO_HOME=$PRESTO_HOME">> ${USER_HOME}/.bashrc
            echo "export PATH=\$PRESTO_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
        else
            echo "export PRESTO_HOME=$PRESTO_HOME">> ${USER_HOME}/.bashrc
        fi
    fi
}

install_jdk
install_tools
install_presto
