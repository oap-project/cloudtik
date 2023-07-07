ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik-deps:"$BASE_IMAGE"

# Install spark based Analytics + AI platform components
ARG SPARK_VERSION=3.2.1
ARG HADOOP_VERSION=3.3.1

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

# Ganglia and Fuse
RUN sudo apt-get update -y \
    && sudo apt-get install -y ganglia-monitor \
    && arch=$(uname -m) && wget https://d30257nes7d4fq.cloudfront.net/downloads/ganglia/modpython-${arch}.so -O modpython.so \
    && sudo cp modpython.so /usr/lib/ganglia && sudo chmod 644 /usr/lib/ganglia/modpython.so \
    && rm modpython.so \
    # Install fuse tools
    && wget -N https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb \
    && sudo dpkg -i packages-microsoft-prod.deb \
    && echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" |sudo tee /etc/apt/sources.list.d/gcsfuse.list \
    && sudo apt-get update -y && sudo apt-get install -y \
         libfuse3-dev \
         fuse3 \
    && sudo apt-get update -y && sudo apt-get install -y \
         s3fs \
         blobfuse2 \
         gcsfuse \
         gdebi-core \
    && wget -N https://gosspublic.alicdn.com/ossfs/ossfs_1.80.7_ubuntu20.04_amd64.deb \
    && sudo gdebi --n ossfs_1.80.7_ubuntu20.04_amd64.deb \
    && rm -rf ossfs_1.80.7_ubuntu20.04_amd64.deb \
    && rm -rf packages-microsoft-prod.deb \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo rm -rf /etc/apt/sources.list.d/gcsfuse.list \
    && sudo rm -rf /etc/apt/sources.list.d/microsoft-prod.list \
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
    wget -O "$HADOOP_HOME/share/hadoop/hdfs/hadoop-hdfs-nfs-${HADOOP_VERSION}.jar" https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-hdfs-nfs-${HADOOP_VERSION}.jar && \
    echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# Install Spark
ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           $RUNTIME_PATH/spark
ENV PATH                 $SPARK_HOME/bin:$PATH

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz -O spark.tgz && \
    mkdir -p "$SPARK_HOME" && \
    tar --extract --file spark.tgz --directory "$SPARK_HOME" --strip-components 1 --no-same-owner && \
    ln -rs $SPARK_HOME/examples/jars/spark-examples_*.jar $SPARK_HOME/examples/jars/spark-examples.jar && \
    rm spark.tgz && \
    wget -nc -P "${SPARK_HOME}/jars" https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.12/${SPARK_VERSION}/spark-hadoop-cloud_2.12-${SPARK_VERSION}.jar && \
    echo "export PYTHONPATH=\${SPARK_HOME}/python:\${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip" >> ~/.bashrc && \
    echo "export PYSPARK_PYTHON=\${CONDA_ROOT}/envs/\${CLOUDTIK_ENV}/bin/python" >> ~/.bashrc && \
    echo "export PYSPARK_DRIVER_PYTHON=\${CONDA_ROOT}/envs/\${CLOUDTIK_ENV}/bin/python" >> ~/.bashrc

# Install python packages
WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    && pip --no-cache-dir install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    && python -m spylon_kernel install --user
