#!/bin/bash

function install_jdk() {
    # Install JDK
    export JAVA_HOME=$RUNTIME_PATH/jdk

    if [ ! -d "${JAVA_HOME}" ]; then
      (cd $RUNTIME_PATH && wget -q --show-progress https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.16.1%2B1/OpenJDK11U-jdk_x64_linux_hotspot_11.0.16.1_1.tar.gz -O openjdk.tar.gz && \
          mkdir -p "$JAVA_HOME" && \
          tar --extract --file openjdk.tar.gz --directory "$JAVA_HOME" --strip-components 1 --no-same-owner && \
          rm openjdk.tar.gz)
        echo "export JAVA_HOME=$JAVA_HOME">> ${USER_HOME}/.bashrc
        echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}
