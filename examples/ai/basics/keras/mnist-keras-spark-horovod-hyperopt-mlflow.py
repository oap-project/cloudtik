# Common Imports
import argparse
import os
import subprocess
import sys
from time import time
import pickle
from distutils.version import LooseVersion


# Settings
parser = argparse.ArgumentParser(description='Horovod on Spark Keras MNIST Example')
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train (default: 2)')
parser.add_argument('--trials', type=int, default=2,
                    help='number of trails to parameter tuning (default: 2)')
parser.add_argument('--fsdir', default=None,
                    help='the file system dir (default: None)')
parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')

if __name__ == '__main__':
    args = parser.parse_args()
    fsdir = args.fsdir

    # CloudTik cluster preparation or information
    from cloudtik.runtime.spark.api import ThisSparkCluster
    from cloudtik.runtime.ai.api import ThisAICluster

    cluster = ThisSparkCluster()

    # Scale the cluster as need
    # cluster.scale(workers=1)

    # Wait for all cluster workers to be ready
    cluster.wait_for_ready(min_workers=1)

    # Total worker cores
    cluster_info = cluster.get_info()

    if not args.num_proc:
        total_worker_cpus = cluster_info.get("total-worker-cpus-ready")
        if total_worker_cpus:
            args.num_proc = int(total_worker_cpus / 2)
        if not args.num_proc:
            args.num_proc = 1

    default_storage = cluster.get_default_storage()
    if not fsdir:
        fsdir = default_storage.get("default.storage.uri") if default_storage else None
        if not fsdir:
            print("Must specify storage filesystem dir using --fsdir.")
            sys.exit(1)

    ai_cluster = ThisAICluster()
    mlflow_url = ai_cluster.get_services()["mlflow"]["url"]

    # Serialize and deserialize keras model
    import io
    import h5py


    def serialize_keras_model(model):
        """Serialize model into byte array encoded into base 64."""
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as f:
            keras.models.save_model(model, f)
        return bio.getvalue()


    def deserialize_keras_model(model_bytes, custom_objects):
        """Deserialize model from byte array encoded in base 64."""
        bio = io.BytesIO(model_bytes)
        with h5py.File(bio, 'r') as f:
            with keras.utils.custom_object_scope(custom_objects):
                return keras.models.load_model(f)


    # Checkpoint utilities
    CHECKPOINT_HOME = "/tmp/ai/checkpoints"


    def get_checkpoint_file(log_dir, file_id):
        return os.path.join(log_dir, 'checkpoint-{file_id}.bin'.format(file_id=file_id))


    def save_checkpoint(log_dir, model, optimizer, file_id, meta=None):
        filepath = get_checkpoint_file(log_dir, file_id)
        print('Written checkpoint to {}'.format(filepath))

        model_bytes = serialize_keras_model(model)
        state = {
            'model': model_bytes,
        }
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if meta is not None:
            state['meta'] = meta

        # write file
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)


    def load_checkpoint(log_dir, file_id):
        filepath = get_checkpoint_file(log_dir, file_id)
        with open(filepath, 'rb') as f:
            return pickle.load(f)


    def create_log_dir(experiment_name):
        log_dir = os.path.join(CHECKPOINT_HOME, str(time()), experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir


    # Initialize SparkSession
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf().setAppName('spark-horovod-keras').set('spark.sql.shuffle.partitions', '16')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Download MNIST dataset and upload to storage

    # Download
    data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2'
    mnist_data_path = os.path.join('/tmp', 'mnist.bz2')
    if not os.path.exists(mnist_data_path):
        subprocess.check_output(['wget', data_url, '-O', mnist_data_path])

    # Upload to the default distributed storage
    os.system("hadoop fs -mkdir /tmp")
    os.system("hadoop fs -put   /tmp/mnist.bz2  /tmp")

    # Feature processing

    # Load to Spark dataframe
    df = spark.read.format('libsvm').option('numFeatures', '784').load(mnist_data_path)

    # One-hot encode labels into SparseVectors
    from pyspark.ml.feature import OneHotEncoder

    encoder = OneHotEncoder(inputCols=['label'],
                            outputCols=['label_vec'],
                            dropLast=False)
    model = encoder.fit(df)
    train_df = model.transform(df)

    # Split to train/test
    train_df, test_df = train_df.randomSplit([0.9, 0.1])
    if train_df.rdd.getNumPartitions() < args.num_proc:
        train_df = train_df.repartition(args.num_proc)

    # Define training function and Keras model
    import numpy as np

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D

    import horovod.spark.keras as hvd
    from horovod.spark.common.backend import SparkBackend
    from horovod.spark.common.store import Store

    import pyspark.sql.types as T
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.sql.functions import udf

    # The break change from Tensorflow keras 2.11:
    # Horovod supported optimizers moved to legacy
    from packaging import version
    if version.parse(tf.keras.__version__.replace("-tf", "+tf")) < version.parse("2.11"):
        from tensorflow.keras import optimizers
    else:
        from tensorflow.keras.optimizers import legacy as optimizers

    # Disable GPUs when building the model to prevent memory leaks
    if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
        # See https://github.com/tensorflow/tensorflow/issues/33168
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        keras.backend.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))

    # Set the parameters
    num_proc = args.num_proc
    print("Train processes: {}".format(num_proc))

    batch_size = args.batch_size
    print("Train batch size: {}".format(batch_size))

    epochs = args.epochs
    print("Train epochs: {}".format(epochs))

    # Create store for data accessing
    store_path = fsdir + "/tmp"
    # AWS and GCP cloud storage authentication just work with empty storage options
    # Azure cloud storage authentication needs a few options
    storage_options = {}
    if default_storage and "azure.storage.account" in default_storage:
        storage_options["anon"] = False
        storage_options["account_name"] = default_storage["azure.storage.account"]
    store = Store.create(store_path, storage_options=storage_options)

    #  Horovod distributed training
    def train(learning_rate):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        optimizer = optimizers.Adadelta(learning_rate)
        loss = keras.losses.categorical_crossentropy

        backend = SparkBackend(num_proc=num_proc,
                               stdout=sys.stdout, stderr=sys.stderr,
                               use_gloo=args.use_gloo, use_mpi=args.use_mpi,
                               prefix_output_with_timestamp=True)
        keras_estimator = hvd.KerasEstimator(backend=backend,
                                             store=store,
                                             model=model,
                                             optimizer=optimizer,
                                             loss=loss,
                                             metrics=['accuracy'],
                                             feature_cols=['features'],
                                             label_cols=['label_vec'],
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             verbose=1)

        keras_model = keras_estimator.fit(train_df).setOutputCols(['label_prob'])
        return keras_model


    def test_model(model, show_samples=False):
        pred_df = model.transform(test_df)
        argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
        pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))

        evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
        accuracy = evaluator.evaluate(pred_df)
        print('Test accuracy:', accuracy)

        if show_samples:
            pred_df = pred_df.sampleBy('label',
                                       fractions={0.0: 0.1, 1.0: 0.1, 2.0: 0.1, 3.0: 0.1, 4.0: 0.1,
                                                  5.0: 0.1, 6.0: 0.1, 7.0: 0.1, 8.0: 0.1, 9.0: 0.1})
            pred_df.show(150)

        return 1 - accuracy


    def load_model_of_checkpoint(log_dir, file_id):
        checkpoint = load_checkpoint(log_dir, file_id)
        meta = checkpoint.get('meta', {})
        custom_objects = meta.get('custom_objects', {})
        keras_model = deserialize_keras_model(checkpoint['model'], custom_objects)
        model = hvd.KerasModel(
            model=keras_model,
            custom_objects=custom_objects,
            _floatx=meta.get('floatx'),
            _metadata=meta.get('metadata'))
        return model


    #  Hyperopt training function
    import mlflow

    checkpoint_dir = create_log_dir('keras-mnist')
    print("Log directory:", checkpoint_dir)


    def hyper_objective(learning_rate):
        with mlflow.start_run():
            model = train(learning_rate)

            # Write checkpoint
            meta = {
                'custom_objects': model.getCustomObjects(),
                'floatx': model._get_floatx(),
                'metadata': model._get_metadata(),
            }
            save_checkpoint(checkpoint_dir, model.getModel(), None, learning_rate, meta)

            test_loss = test_model(model)

            mlflow.log_metric("learning_rate", learning_rate)
            mlflow.log_metric("loss", test_loss)
        return {'loss': test_loss, 'status': STATUS_OK}


    # Do a super parameter tuning with hyperopt

    from hyperopt import fmin, tpe, hp, STATUS_OK

    trials = args.trials
    print("Hyper parameter tuning trials: {}".format(trials))

    search_space = hp.uniform('learning_rate', 0, 1)
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("MNIST: Keras + Spark + Horovod")
    argmin = fmin(
        fn=hyper_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=trials)
    print("Best parameter found: ", argmin)

    # Train final model with the best parameters
    best_model = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
    metadata = best_model._get_metadata()
    floatx = best_model._get_floatx()
    model_name = 'mnist-keras-spark-horovod'
    mlflow.tensorflow.log_model(best_model.getModel(), model_name, registered_model_name=model_name)

    # Load the model from MLflow and run a transformation
    model_uri = "models:/{}/latest".format(model_name)
    print('Inference with model: {}'.format(model_uri))
    saved_keras_model = mlflow.tensorflow.load_model(model_uri)
    saved_model = hvd.KerasModel(
        model=saved_keras_model,
        feature_columns=['features'],
        label_columns=['label_vec'],
        _floatx=floatx,
        _metadata=metadata).setOutputCols(['label_prob'])
    test_model(saved_model, True)

    # Clean up
    spark.stop()
