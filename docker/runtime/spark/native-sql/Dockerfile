ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

ARG NODE_EXPORTER_VERSION=1.6.1
ARG PROMETHEUS_VERSION=2.45.0
ARG SPARK_VERSION=3.2.1
ARG HADOOP_VERSION=3.3.1

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

# Install prometheus and node exporter
ENV NODE_EXPORTER_HOME $RUNTIME_PATH/node_exporter
ENV PROMETHEUS_HOME $RUNTIME_PATH/prometheus

RUN wget https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz -O node_exporter.tar.gz && \
    mkdir -p "$NODE_EXPORTER_HOME" && \
    tar --extract --file node_exporter.tar.gz --directory "$NODE_EXPORTER_HOME" --strip-components 1 --no-same-owner && \
    rm node_exporter.tar.gz && \
    wget https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz -O prometheus.tar.gz && \
    mkdir -p "$PROMETHEUS_HOME" && \
    tar --extract --file prometheus.tar.gz --directory "$PROMETHEUS_HOME" --strip-components 1 --no-same-owner && \
    rm prometheus.tar.gz

# Install JDK
ENV JAVA_HOME            $RUNTIME_PATH/jdk
ENV PATH                 $JAVA_HOME/bin:$PATH

# JDK download links refer to https://github.com/adoptium/containers
# and https://github.com/docker-library/docs/blob/master/eclipse-temurin/README.md
RUN wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz -O jdk-8u192-linux-x64.tar.gz && \
    mkdir -p "$JAVA_HOME" && \
    tar --extract --file jdk-8u192-linux-x64.tar.gz --directory "$JAVA_HOME" --strip-components 1 --no-same-owner && \
    rm jdk-8u192-linux-x64.tar.gz

# Install Hadoop
ENV HADOOP_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV PATH $HADOOP_HOME/bin:$PATH

RUN wget http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop.tar.gz && \
    mkdir -p "$HADOOP_HOME" && \
    tar --extract --file hadoop.tar.gz --directory "$HADOOP_HOME" --strip-components 1 --no-same-owner && \
    rm hadoop.tar.gz && \
    wget -nc -P "${HADOOP_HOME}/share/hadoop/tools/lib" https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-azure-${HADOOP_VERSION}.jar" ${CLOUDTIK_DOWNLOADS}/hadoop/hadoop-azure-${HADOOP_VERSION}.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-aliyun-${HADOOP_VERSION}.jar" ${CLOUDTIK_DOWNLOADS}/hadoop/hadoop-aliyun-${HADOOP_VERSION}.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-huaweicloud-${HADOOP_VERSION}.jar" ${CLOUDTIK_DOWNLOADS}/hadoop/hadoop-huaweicloud-${HADOOP_VERSION}.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/hdfs/hadoop-hdfs-nfs-${HADOOP_VERSION}.jar" ${CLOUDTIK_DOWNLOADS}/hadoop/hadoop-hdfs-nfs-${HADOOP_VERSION}.jar && \
    echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# Install Spark
ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           $RUNTIME_PATH/spark
ENV PATH                 $SPARK_HOME/bin:$PATH

RUN wget ${CLOUDTIK_DOWNLOADS}/spark/spark-${SPARK_VERSION}-bin-hadoop3.tgz -O spark.tgz && \
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
RUN export PATH="$HOME/anaconda3/envs/${CLOUDTIK_ENV}/bin:$PATH" \
    && pip --no-cache-dir install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    && python -m spylon_kernel install --user

ARG OAP_VERSION=1.4.0.spark32

ENV OAP_VERSION        ${OAP_VERSION}
ENV OAP_HOME           $RUNTIME_PATH/oap
ENV LD_LIBRARY_PATH $HOME/runtime/oap/lib/:$LD_LIBRARY_PATH
ENV CODEGEN_OPTION  " -O1 -march=native -fno-semantic-interposition "
ENV LIBARROW_DIR      $HOME/runtime/oap/
ENV CC                         $HOME/runtime/oap/bin/x86_64-conda_cos6-linux-gnu-cc

# Install OAP by Conda
RUN $HOME/anaconda3/bin/conda create -p "${OAP_HOME}" -c conda-forge -c intel-bigdata -c intel -y oap=${OAP_VERSION} \
    && $HOME/anaconda3/bin/conda clean -itqy
