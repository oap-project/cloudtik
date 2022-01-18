ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

# Install spark based Analytics + AI platform components
ARG HADOOP_VERSION=3.2.0
ARG SPARK_VERSION=3.1.1


ENV RUNTIME_PATH /home/cloudtik/runtime
WORKDIR /home/cloudtik/runtime

#hadoop
ENV HADOOP_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_HDFS_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_MAPRED_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_YARN_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV YARN_CONF_DIR $HADOOP_HOME/etc/hadoop

RUN sudo wget --quiet http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop.tar.gz && \
    sudo tar -zxvf hadoop.tar.gz && \
    sudo mv hadoop-${HADOOP_VERSION} hadoop && \
    sudo rm hadoop.tar.gz

#java
ENV JAVA_HOME            /home/cloudtik/runtime/jdk
ENV PATH                          $JAVA_HOME/bin:$PATH

RUN sudo wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
    sudo gunzip jdk-8u192-linux-x64.tar.gz && \
    sudo tar -xf jdk-8u192-linux-x64.tar && \
    sudo rm jdk-8u192-linux-x64.tar && \
    sudo ln -s jdk1.8.0_192 jdk

#spark
ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           $RUNTIME_PATH/spark


RUN sudo wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    sudo tar -zxvf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    sudo mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
    sudo rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz


WORKDIR /home/cloudtik/


COPY environment.yml /tmp/env.yml

RUN conda env create -n runtime --file /tmp/env.yml --force

#jupyter config
RUN echo 'export PYTHONPATH=${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip \n\
export PYSPARK_PYTHON=${CONDA_PREFIX}/bin/python\n\
export PYSPARK_DRIVER_PYTHON=${CONDA_PREFIX}/bin/python\n\' >> '$CONDA_PREFIX/etc/jupyterrc'

#access jupyter server remotely
RUN mkdir -p /home/cloudtik/jupyter && echo Y |jupyter notebook --generate-config && \
  sed -i  "1 ic.NotebookApp.ip = '*'" ~/.jupyter/jupyter_notebook_config.py && \
  sed -i  '1 ic.NotebookApp.open_browser = False' ~/.jupyter/jupyter_notebook_config.py && \
  sed -i  '1 ic.NotebookApp.allow_root = True'   ~/.jupyter/jupyter_notebook_config.py && \
  sed -i  "1 ic.NotebookApp.notebook_dir = '/home/cloudtik/jupyter'" ~/.jupyter/jupyter_notebook_config.py

# jupyter kernel
RUN pip install -U https://dist.apache.org/repos/dist/dev/incubator/toree/0.5.0-incubating-rc1/toree-pip/toree-0.5.0.tar.gz && \
    jupyter toree install --spark_opts='--master=yarn' --kernel_name=Spark  --spark_home=$SPARK_HOME && \
    pip install spylon-kernel&&python -m spylon_kernel install

#horovod
RUN conda install -c anaconda -y gfortran_linux-64 && \
    HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod[tensorflow,keras,pytorch,mxnet,spark]==0.21.3
