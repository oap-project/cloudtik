# Common Imports
import argparse
import os
import sys
from time import time
import pickle

# Settings
parser = argparse.ArgumentParser(description='Single Node Keras MNIST Example')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train (default: 2)')
parser.add_argument('--trials', type=int, default=2,
                    help='number of trails to parameter tuning (default: 2)')
parser.add_argument('--fsdir', default=None,
                    help='the file system dir (default: None)')

if __name__ == '__main__':
    args = parser.parse_args()
    fsdir = args.fsdir

    # CloudTik cluster preparation or information
    from cloudtik.runtime.spark.api import ThisSparkCluster
    from cloudtik.runtime.ai.api import ThisAICluster

    cluster = ThisSparkCluster()

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
    import keras
    from horovod.runner.common.util import codec


    def serialize_keras_model(model):
        """Serialize model into byte array encoded into base 64."""
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as f:
            keras.models.save_model(model, f)
        return codec.dumps_base64(bio.getvalue())


    def deserialize_keras_model(model_bytes, custom_objects):
        """Deserialize model from byte array encoded in base 64."""
        model_bytes = codec.loads_base64(model_bytes)
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


    # Define training function and Keras model
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K

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

    # Single node train function
    def train(learning_rate):
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

        optimizer = keras.optimizers.Adadelta(learning_rate)
        loss = keras.losses.categorical_crossentropy

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        callbacks = [
            keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'),
        ]

        # Load data
        x_train, y_train, x_test, y_test = load_data()

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        # Return the model
        return model


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


    def test_checkpoint(log_dir, file_id):
        model = load_model_of_checkpoint(log_dir, file_id)
        return test_model(model)


    #  Hyperopt training function
    import mlflow

    checkpoint_dir = create_log_dir('single-node-keras-mnist')
    print("Log directory:", checkpoint_dir)


    def hyper_objective(learning_rate):
        with mlflow.start_run():
            model = train(learning_rate)

            # Write checkpoint
            save_checkpoint(checkpoint_dir, model, None, learning_rate)

            model = load_model_of_checkpoint(checkpoint_dir, learning_rate)
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
    mlflow.set_experiment("MNIST: Keras + Single Node")
    argmin = fmin(
        fn=hyper_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=trials)
    print("Best parameter found: ", argmin)

    # Train final model with the best parameters and save the model
    best_model = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
    model_name = "mnist-keras-single-node"
    mlflow.tensorflow.log_model(
        best_model, model_name, registered_model_name=model_name)

    # Load the model from MLflow and run an evaluation
    model_uri = "models:/{}/latest".format(model_name)
    print('Inference with model: {}'.format(model_uri))
    saved_model = mlflow.tensorflow.load_model(model_uri)
    test_model(saved_model)
