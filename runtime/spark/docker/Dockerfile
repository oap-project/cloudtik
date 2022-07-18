ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

# Install spark based Analytics + AI platform components
ARG HADOOP_VERSION=3.3.1
ARG SPARK_VERSION=3.2.1

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

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
    echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# Install Spark
ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           $RUNTIME_PATH/spark
ENV PATH                 $SPARK_HOME/bin:$PATH

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
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

RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    && pip install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    && export CXX=/usr/bin/g++-9 \
    && HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_GLOO=1  pip install horovod[all-frameworks]
