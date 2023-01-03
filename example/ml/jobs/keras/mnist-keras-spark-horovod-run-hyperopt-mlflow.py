# Common Imports
import argparse
import os
import sys
from time import time
import pickle


# Settings
parser = argparse.ArgumentParser(description='Horovod on Spark Keras MNIST Example')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train (default: 1)')
parser.add_argument('--trials', type=int, default=2,
                    help='number of trails to parameter tuning (default: 2)')
parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')

if __name__ == '__main__':
    args = parser.parse_args()

    # CloudTik cluster preparation or information
    from cloudtik.runtime.spark.api import ThisSparkCluster
    from cloudtik.runtime.ml.api import ThisMLCluster

    cluster = ThisSparkCluster()

    # Scale the cluster as need
    # cluster.scale(workers=1)

    # Wait for all cluster workers to be ready
    cluster.wait_for_ready(min_workers=1)

    # Total worker cores
    cluster_info = cluster.get_info()
    total_workers = cluster_info.get("total-workers")
    if not total_workers:
        total_workers = 1

    ml_cluster = ThisMLCluster()
    mlflow_url = ml_cluster.get_services()["mlflow"]["url"]

    # Initialize SparkSession
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf().setAppName('spark-horovod-pytorch').set('spark.sql.shuffle.partitions', '16')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

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
    CHECKPOINT_HOME = "/tmp/ml/checkpoints"


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
        os.makedirs(log_dir)
        return log_dir


    # Define training function and Keras model
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K

    import tensorflow as tf
    import horovod.keras as hvd

    # Set the parameters
    num_proc = total_workers
    print("Train processes: {}".format(num_proc))

    batch_size = args.batch_size
    print("Train batch size: {}".format(batch_size))

    epochs = args.epochs
    print("Train epochs: {}".format(epochs))

    num_classes = 10
    img_rows, img_cols = 28, 28


    def load_data():
        # The data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return x_train, y_train, x_test, y_test

    #  Horovod distributed training function
    def train_horovod(learning_rate):
        # Horovod: initialize Horovod.
        hvd.init()

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.visible_device_list = str(hvd.local_rank())
        # K.set_session(tf.Session(config=config))

        # Input image dimensions
        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_rows, img_cols)
        else:
            input_shape = (img_rows, img_cols, 1)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Horovod: adjust learning rate based on number of GPUs.
        optimizer = keras.optimizers.Adadelta(learning_rate * hvd.size())

        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)
        loss = keras.losses.categorical_crossentropy

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        ]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

        # Load data
        x_train, y_train, x_test, y_test = load_data()

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  epochs=epochs,
                  verbose=1 if hvd.rank() == 0 else 0,
                  validation_data=(x_test, y_test))

        if hvd.rank() == 0:
            # Return the model bytes
            return serialize_keras_model(model)


    def test_model(model):
        x_train, y_train, x_test, y_test = load_data()
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score[0]


    def load_model_of_checkpoint(log_dir, file_id):
        checkpoint = load_checkpoint(log_dir, file_id)
        meta = checkpoint.get('meta', {})
        custom_objects = meta.get('custom_objects', {})
        model = deserialize_keras_model(checkpoint['model'], custom_objects)
        return model


    #  Hyperopt training function
    import mlflow
    import horovod.spark

    checkpoint_dir = create_log_dir('horovod-keras-mnist')
    print("Log directory:", checkpoint_dir)


    def hyper_objective(learning_rate):
        with mlflow.start_run():
            model_bytes = horovod.spark.run(
                train_horovod, args=(learning_rate,), num_proc=num_proc,
                stdout=sys.stdout, stderr=sys.stderr, verbose=2,
                use_gloo=args.use_gloo, use_mpi=args.use_mpi,
                prefix_output_with_timestamp=True)[0]
            model = deserialize_keras_model(model_bytes, custom_objects={})

            # Write checkpoint
            save_checkpoint(checkpoint_dir, model, None, learning_rate)

            test_loss = test_model(model)

            mlflow.log_metric("learning_rate", learning_rate)
            mlflow.log_metric("loss", test_loss)
        return {'loss': test_loss, 'status': STATUS_OK}


    # Do a super parameter tuning with hyperopt

    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    trials = args.trials
    print("Hyper parameter tuning trials: {}".format(trials))

    search_space = hp.uniform('learning_rate', 0, 1)
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("MNIST: Horovod + Hyperopt + Keras")
    argmin = fmin(
        fn=hyper_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=trials)
    print("Best parameter found: ", argmin)

    # Train final model with the best parameters
    best_model = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
    model_name = 'keras-mnist-model'
    mlflow.keras.log_model(best_model, model_name, registered_model_name=model_name)

    # Load the model from MLflow and run a transformation
    model_uri = "models:/{}/latest".format(model_name)
    print('Inference with model: {}'.format(model_uri))
    saved_model = mlflow.keras.load_model(model_uri)
    test_model(saved_model)

    # Clean up
    spark.stop()
