import argparse
import logging
import os
import zipfile
import time
import tempfile


# Training settings
parser = argparse.ArgumentParser(description='MXNet MNIST Example')

parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable training on GPU (default: False)')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

parser.add_argument('--trials', type=int, default=2,
                    help='number of trails to parameter tuning (default: 2)')
parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')


import mxnet as mx
# MXNet uses protobuf 3.5.1 which conflicts with TensorFlow protobuf 3.9.0
import horovod.mxnet as hvd
from mxnet import autograd, gluon, nd
from mxnet.test_utils import download

import mlflow


# Function to get mnist iterator given a rank
def get_mnist_iterator(world_size, rank):
    data_dir = "data-%d" % rank
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    zip_file_path = download('http://data.mxnet.io/mxnet/data/mnist.zip',
                             dirname=data_dir)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(data_dir)

    input_shape = (1, 28, 28)
    batch_size = args.batch_size

    train_iter = mx.io.MNISTIter(
        image="%s/train-images-idx3-ubyte" % data_dir,
        label="%s/train-labels-idx1-ubyte" % data_dir,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=False,
        num_parts=world_size,
        part_index=rank
    )

    val_iter = mx.io.MNISTIter(
        image="%s/t10k-images-idx3-ubyte" % data_dir,
        label="%s/t10k-labels-idx1-ubyte" % data_dir,
        input_shape=input_shape,
        batch_size=batch_size,
        flat=False,
    )

    return train_iter, val_iter


# Function to define neural network
def conv_nets():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(10))
    return net


# Function to evaluate accuracy for a model
def evaluate(model, data_iter, context):
    data_iter.reset()
    metric = mx.metric.Accuracy()
    for _, batch in enumerate(data_iter):
        data = batch.data[0].as_in_context(context)
        label = batch.label[0].as_in_context(context)
        output = model(data.astype(args.dtype, copy=False))
        metric.update([label], [output])

    return metric.get()


def train_horovod(learning_rate):
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    # Initialize Horovod
    hvd.init()

    # Horovod: pin context to local rank
    context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())

    # Load training and validation data
    train_data, val_data = get_mnist_iterator(hvd.size(), hvd.rank())

    # Build model
    model = conv_nets()
    model.cast(args.dtype)
    model.hybridize()

    # Create optimizer
    optimizer_params = {'momentum': args.momentum,
                        'learning_rate': learning_rate * hvd.size()}
    opt = mx.optimizer.create('sgd', **optimizer_params)

    # Initialize parameters
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                 magnitude=2)
    model.initialize(initializer, ctx=context)

    # Horovod: fetch and broadcast parameters
    params = model.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Horovod: create DistributedTrainer, a subclass of gluon.Trainer
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    trainer = hvd.DistributedTrainer(params, opt, compression=compression,
                                     gradient_predivide_factor=args.gradient_predivide_factor)

    # Create loss function and train metric
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()

    # Train model
    for epoch in range(args.epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        for nbatch, batch in enumerate(train_data, start=1):
            data = batch.data[0].as_in_context(context)
            label = batch.label[0].as_in_context(context)
            with autograd.record():
                output = model(data.astype(args.dtype, copy=False))
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(args.batch_size)
            metric.update([label], [output])

            if nbatch % 100 == 0:
                name, acc = metric.get()
                logging.info('[Epoch %d Batch %d] Training: %s=%f' %
                             (epoch, nbatch, name, acc))

        if hvd.rank() == 0:
            elapsed = time.time() - tic
            speed = nbatch * args.batch_size * hvd.size() / elapsed
            logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f',
                         epoch, speed, elapsed)

        # Evaluate model accuracy
        _, train_acc = metric.get()
        name, val_acc = evaluate(model, val_data, context)
        if hvd.rank() == 0:
            logging.info('Epoch[%d]\tTrain: %s=%f\tValidation: %s=%f', epoch, name,
                         train_acc, name, val_acc)

        if hvd.rank() == 0 and epoch == args.epochs - 1:
            assert val_acc > 0.96, "Achieved accuracy (%f) is lower than expected\
                                    (0.96)" % val_acc
    if hvd.rank() == 0:
        return serialize_gluon_model(model)


def serialize_gluon_model(model):
    """Serialize model into byte array."""
    name = "gluon_model_{}".format(time.time())
    model.export(name)
    symbol_file = name + "-symbol.json"
    params_file = name + "-0000.params"
    with open(symbol_file, 'rb') as f_symbol:
        with open(params_file, 'rb') as f_params:
            return f_symbol.read(), f_params.read()


# Checkpoint utilities
CHECKPOINT_HOME = "/tmp/ml/checkpoints"


def get_checkpoint_file(log_dir, file_id):
    return os.path.join(log_dir, 'checkpoint-{file_id}.model'.format(file_id=file_id))


def save_checkpoint(log_dir, model, optimizer, file_id, meta=None):
    filepath = get_checkpoint_file(log_dir, file_id)
    print('Written checkpoint to {}'.format(filepath))
    mlflow.gluon.save_model(model, filepath)


def create_log_dir(experiment_name):
    log_dir = os.path.join(CHECKPOINT_HOME, str(time.time()), experiment_name)
    os.makedirs(log_dir)
    return log_dir


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.no_cuda:
        # Disable CUDA if there are no GPUs.
        if not mx.test_utils.list_gpus():
            args.no_cuda = True

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

    worker_ips = cluster.get_worker_node_ips()

    ml_cluster = ThisMLCluster()
    mlflow_url = ml_cluster.get_services()["mlflow"]["url"]

    ctx = mx.cpu() if args.no_cuda else mx.gpu()

    _, test_data = get_mnist_iterator(1, 0)

    def test_model(model):
        name, val_acc = evaluate(model, test_data, ctx)
        print("Test {}={}".format(name, val_acc))
        return 1 - val_acc


    def load_model_of_checkpoint(log_dir, file_id):
        filepath = get_checkpoint_file(log_dir, file_id)
        return mlflow.gluon.load_model(filepath, ctx)


    def deserialize_gluon_model(model_symbol, model_params):
        """Deserialize model from byte array."""
        tmp_symbol_file = tempfile.mktemp(prefix="gluon_model_")
        with open(tmp_symbol_file, 'wb') as f:
            f.write(model_symbol)

        tmp_params_file = tempfile.mktemp(prefix="gluon_model_")
        with open(tmp_params_file, 'wb') as f:
            f.write(model_params)

        model = gluon.nn.SymbolBlock.imports(tmp_symbol_file, ['data'], tmp_params_file, ctx=ctx)
        return model


    #  Hyperopt training function
    import horovod

    # Set the parameters
    num_proc = total_workers
    print("Train processes: {}".format(num_proc))

    checkpoint_dir = create_log_dir('mnist-mxnet-horovod')
    print("Log directory:", checkpoint_dir)

    # Generate the host list
    host_slots = ["{}:1".format(worker_ip) for worker_ip in worker_ips]
    hosts = ",".join(host_slots)
    print("Hosts to run:", hosts)

    def hyper_objective(learning_rate):
        with mlflow.start_run():
            (model_symbol, model_params) = horovod.run(
                train_horovod, args=(learning_rate,),
                num_proc=num_proc, hosts=hosts,
                use_gloo=args.use_gloo, use_mpi=args.use_mpi,
                verbose=2)[0]
            model = deserialize_gluon_model(model_symbol, model_params)

            # Write checkpoint
            save_checkpoint(checkpoint_dir, model, None, learning_rate)

            test_loss = test_model(model)

            mlflow.log_metric("learning_rate", learning_rate)
            mlflow.log_metric("loss", test_loss)
        return {'loss': test_loss, 'status': STATUS_OK}


    # Do a super parameter tuning with hyperopt
    from hyperopt import fmin, tpe, hp, STATUS_OK

    trials = args.trials
    print("Hyper parameter tuning trials: {}".format(trials))

    search_space = hp.uniform('learning_rate', 0.01, 0.02)
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("MNIST: MXNet + Horovod Run")
    argmin = fmin(
        fn=hyper_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=trials)
    print("Best parameter found: ", argmin)

    # Train final model with the best parameters
    best_model = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
    model_name = 'mnist-mxnet-horovod-run'
    mlflow.gluon.log_model(best_model, model_name, registered_model_name=model_name)

    # Load the model from MLflow and run a transformation
    model_uri = "models:/{}/latest".format(model_name)
    print('Inference with model: {}'.format(model_uri))
    saved_model = mlflow.gluon.load_model(model_uri)
    test_model(saved_model)
