# Distributed Machine Learning and Deep Learning Examples

Here we provide a guide for you to run AI related examples based on CloudTik AI runtime
which includes selected ML/DL frameworks and libraries.

We provide these examples in two forms:
1. Python jobs
2. Jupyter notebooks

## Running the examples using python jobs

### Running Spark Distributed Deep Learning example
This example runs Spark distributed deep learning with Hyperopt, Horovod and Tensorflow Keras API.
It trains a simple ConvNet on the MNIST dataset using Keras + Horovod using Cloudtik Spark Runtime.

Download [Spark Deep Learning with Horovod and Tensorflow Keras](jobs/keras/mnist-keras-spark-horovod-hyperopt-mlflow.py)
and execute:
```
cloudtik submit /path/to/your-cluster-config.yaml local-download-path/mnist-keras-spark-horovod-hyperopt-mlflow.py -f "your-cloud-storage-fsdir"
```

Replace the cloud storage fsdir with the workspace cloud storage uri or hdfs dir. For example S3,  "s3a://cloudtik-workspace-bucket"


### Running Spark Distributed Machine Learning example
This example runs Spark distributed training using scikit-learn.
It illustrates a complete end-to-end example of loading data, training a model, distributed hyperparameter tuning, and model inference
under CloudTik Spark cluster. It also illustrates how to use MLflow and model Registry.

Download [Spark Machine Learning with Scikit-Learn](jobs/scikit-learn/iris-scikit_learn-spark-hyperopt-mlflow.py)
and execute:
```
cloudtik submit /path/to/your-cluster-config.yaml local-download-path/iris-scikit_learn-spark-hyperopt-mlflow.py -f "your-cloud-storage-fsdir"
```

Replace the cloud storage fsdir with the workspace cloud storage uri or hdfs dir. For example S3,  "s3a://cloudtik-workspace-bucket"


## Running the examples using Jupyter notebooks

### Running Spark Distributed Deep Learning example

This notebook example runs Spark distributed deep learning with Hyperopt, Horovod and Tensorflow Keras API.
It trains a simple ConvNet on the MNIST dataset using Keras + Horovod using Cloudtik Spark Runtime.
 
1. Upload notebook [Spark Deep Learning with Horovod and Tensorflow Keras](notebooks/mnist-keras-spark-horovod-hyperopt-mlflow.ipynb) to JupyterLab.
You can also download and cloudtik rsync-up the file to ~/jupyter of cluster head:

```
cloudtik rsync-up /path/to/your-cluster-config.yaml local-download-path/mnist-keras-spark-horovod-hyperopt-mlflow.ipynb '~/jupyter/mnist-keras-spark-horovod-hyperopt-mlflow.ipynb'
```

2. Open this notebook on JupyterLab, and choose the Python 3 kernel to run the notebook.

3. Optionally, you can check the training experiments and model registry through MLflow Web UI after the notebook finishes.

### Running Spark Distributed Machine Learning example

This notebook example runs Spark distributed training using scikit-learn.
It illustrates a complete end-to-end example of loading data, training a model, distributed hyperparameter tuning, and model inference
under CloudTik Spark cluster. It also illustrates how to use MLflow and model Registry.

1. Upload notebook [Spark Machine Learning with Scikit-Learn](notebooks/iris-scikit_learn-spark-hyperopt-mlflow.ipynb) to JupyterLab.
You can also download and cloudtik rsync-up the file to ~/jupyter of cluster head:

```
cloudtik rsync-up /path/to/your-cluster-config.yaml local-download-path/iris-scikit_learn-spark-hyperopt-mlflow.ipynb '~/jupyter/iris-scikit_learn-spark-hyperopt-mlflow.ipynb'
```

2. Open this notebook on JupyterLab, and choose the Python 3 kernel to run the notebook.

3. Optionally, you can check the training experiments and model registry through MLflow Web UI after the notebook finishes.