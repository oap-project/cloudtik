ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

# Install spark based Analytics + AI platform components
ARG SPARK_VERSION=3.2.1
ARG HADOOP_VERSION=3.3.1
ARG CLOUDTIK_REGION=""

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

RUN sudo sed -i 's/mirrors.aliyun.com/archive.ubuntu.com/' /etc/apt/sources.list \
    && pip config unset global.index-url

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
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-azure-${HADOOP_VERSION}.jar" https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-azure-${HADOOP_VERSION}.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-aliyun-${HADOOP_VERSION}.jar" https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-aliyun-${HADOOP_VERSION}.jar && \
    wget -O "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-huaweicloud-${HADOOP_VERSION}.jar" https://d30257nes7d4fq.cloudfront.net/downloads/hadoop/hadoop-huaweicloud-${HADOOP_VERSION}.jar && \
    echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# Install Spark
ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           $RUNTIME_PATH/spark
ENV PATH                 $SPARK_HOME/bin:$PATH

RUN wget https://d30257nes7d4fq.cloudfront.net/downloads/spark/spark-${SPARK_VERSION}-bin-hadoop3.tgz -O spark.tgz && \
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

RUN if [ "${CLOUDTIK_REGION}" = "PRC" ]; \
        then \
            sudo sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list \
            && sudo sed -i 's/security.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list \
            && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/; \
        fi
