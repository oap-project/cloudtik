ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik-deps:"$BASE_IMAGE"

ARG TRINO_VERSION=389
ARG HADOOP_VERSION=3.3.1
ARG CLOUDTIK_REGION=""

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

# Ganglia
RUN sudo apt-get update -y \
    && sudo apt-get install -y ganglia-monitor \
    && arch=$(uname -m) && wget https://d30257nes7d4fq.cloudfront.net/downloads/ganglia/modpython-${arch}.so -O modpython.so \
    && sudo cp modpython.so /usr/lib/ganglia && sudo chmod 644 /usr/lib/ganglia/modpython.so \
    && rm modpython.so \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

# Install JDK
ENV JAVA_HOME            $RUNTIME_PATH/jdk
ENV PATH                 $JAVA_HOME/bin:$PATH

# JDK download links refer to https://github.com/adoptium/containers
# and https://github.com/docker-library/docs/blob/master/eclipse-temurin/README.md
RUN wget https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.16.1%2B1/OpenJDK11U-jdk_x64_linux_hotspot_11.0.16.1_1.tar.gz -O openjdk.tar.gz && \
    mkdir -p "$JAVA_HOME" && \
    tar --extract --file openjdk.tar.gz --directory "$JAVA_HOME" --strip-components 1 --no-same-owner && \
    rm openjdk.tar.gz

# Install Hadoop
ENV HADOOP_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV PATH $HADOOP_HOME/bin:$PATH

RUN wget http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop.tar.gz && \
    mkdir -p "$HADOOP_HOME" && \
    tar --extract --file hadoop.tar.gz --directory "$HADOOP_HOME" --strip-components 1 --no-same-owner && \
    rm hadoop.tar.gz && \
    wget -nc -P "${HADOOP_HOME}/share/hadoop/tools/lib" https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-azure-${HADOOP_VERSION}.jar" https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-azure-${HADOOP_VERSION}.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-aliyun-${HADOOP_VERSION}.jar" https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-aliyun-${HADOOP_VERSION}.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-huaweicloud-${HADOOP_VERSION}.jar" https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-huaweicloud-${HADOOP_VERSION}.jar && \
    echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# Install Trino
ENV TRINO_VERSION        ${TRINO_VERSION}
ENV TRINO_HOME           $RUNTIME_PATH/trino
ENV PATH                 $TRINO_HOME/bin:$PATH

RUN wget https://repo1.maven.org/maven2/io/trino/trino-server/${TRINO_VERSION}/trino-server-${TRINO_VERSION}.tar.gz -O trino-server.tar.gz && \
    mkdir -p "$TRINO_HOME" && \
    tar --extract --file trino-server.tar.gz --directory "$TRINO_HOME" --strip-components 1 --no-same-owner && \
    rm trino-server.tar.gz && \
    wget https://repo1.maven.org/maven2/io/trino/trino-cli/${TRINO_VERSION}/trino-cli-${TRINO_VERSION}-executable.jar && \
    mv trino-cli-${TRINO_VERSION}-executable.jar $TRINO_HOME/bin/trino && \
    chmod +x $TRINO_HOME/bin/trino

RUN if [ "${CLOUDTIK_REGION}" = "PRC" ]; \
        then \
            sudo sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list \
            && sudo sed -i 's/security.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list \
            && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/; \
        fi

WORKDIR /home/cloudtik/
