ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

# Install spark based Analytics + AI platform components
ARG HADOOP_VERSION=3.3.1
ARG SPARK_VERSION=3.2.1

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

# Ganglia
RUN sudo apt-get update -y \
    && sudo apt-get install -y ganglia-monitor \
    && wget https://d30257nes7d4fq.cloudfront.net/downloads/ganglia/modpython.so \
    && sudo cp modpython.so /usr/lib/ganglia && sudo chmod 644 /usr/lib/ganglia/modpython.so \
    && rm modpython.so \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

#Install JDK
ENV JAVA_HOME            $RUNTIME_PATH/jdk
ENV PATH                 $JAVA_HOME/bin:$PATH

RUN wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
    gunzip jdk-8u192-linux-x64.tar.gz && \
    tar -xf jdk-8u192-linux-x64.tar && \
    rm jdk-8u192-linux-x64.tar && \
    mv jdk1.8.0_192 jdk

# Install Hadoop
ENV HADOOP_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV PATH $HADOOP_HOME/bin:$PATH

RUN wget http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop.tar.gz && \
    tar -zxf hadoop.tar.gz && \
    mv hadoop-${HADOOP_VERSION} hadoop && \
    rm hadoop.tar.gz && \
    wget -nc -P "${HADOOP_HOME}/share/hadoop/tools/lib"  https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar && \
    echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# Install Spark
ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           $RUNTIME_PATH/spark
ENV PATH                 $SPARK_HOME/bin:$PATH

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    wget -nc -P "${SPARK_HOME}/jars" https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.12/${SPARK_VERSION}/spark-hadoop-cloud_2.12-${SPARK_VERSION}.jar && \
    echo "export PYTHONPATH=\${SPARK_HOME}/python:\${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip" >> ~/.bashrc && \
    echo "export PYSPARK_PYTHON=\${CONDA_PREFIX}/envs/cloudtik_py37/bin/python" >> ~/.bashrc && \
    echo "export PYSPARK_DRIVER_PYTHON=\${CONDA_PREFIX}/envs/cloudtik_py37/bin/python" >> ~/.bashrc

# Install python packages
WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    && pip install -r  /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    && python -m spylon_kernel install --user

