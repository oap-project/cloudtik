import argparse
import os
import subprocess
import sys
import numpy as np
from distutils.version import LooseVersion

parser = argparse.ArgumentParser(description='PyTorch Spark MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--master',
                    help='spark master to connect to')
parser.add_argument('--num-proc', type=int, default=2,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train (default: 2)')
parser.add_argument('--work-dir', default='/tmp',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')
parser.add_argument('--data-dir', default='/tmp',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')
parser.add_argument('--enable-profiler', action='store_true',
                    help='Enable profiler')
parser.add_argument('--fsdir', default=None,
                    help='the file system dir (default: None)')
parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')


def train_model(args):
    fsdir = args.fsdir

    # CloudTik cluster preparation or information
    from cloudtik.runtime.spark.api import ThisSparkCluster

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
            args.num_proc = int(total_worker_cpus / 1)
        if not args.num_proc:
            args.num_proc = 1

    default_storage = cluster.get_default_storage()
    if not fsdir:
        fsdir = default_storage.get("default.storage.uri") if default_storage else None
        if not fsdir:
            print("Must specify storage filesystem dir using --fsdir.")
            sys.exit(1)

    # Initialize SparkSession
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf().setAppName('mnist-pytorch-lightning-spark').set('spark.sql.shuffle.partitions', '16')
    # This is Spark tasks from Estimator.transform to be able to pick up system libraries
    # TODO: A better way to configure this automatically for user
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_library_path:
        conf.set("spark.executorEnv.LD_LIBRARY_PATH", ld_library_path)
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

    # Train/test split
    train_df, test_df = df.randomSplit([0.9, 0.1])
    if train_df.rdd.getNumPartitions() < args.num_proc:
        train_df = train_df.repartition(args.num_proc)

    # Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x and 2.6.x
    # https://github.com/horovod/horovod/pull/3263
    try:
        # tensorflow has to be imported BEFORE pytorch_lightning, otherwise we see the segfault right away
        import tensorflow as tf
        if LooseVersion('2.5.0') <= LooseVersion(tf.__version__) < LooseVersion('2.7.0'):
            print('Skipping test as Pytorch Lightning conflicts with present Tensorflow 2.6.x', file=sys.stderr)
            sys.exit(0)
    except ImportError:
        pass

    # do not run this test for pytorch lightning below min supported version
    import pytorch_lightning as pl
    from horovod.spark.lightning.estimator import MIN_PL_VERSION
    if LooseVersion(pl.__version__) < LooseVersion(MIN_PL_VERSION):
        print(
            "Skip test for pytorch_ligthning=={}, min support version is {}".format(pl.__version__, MIN_PL_VERSION))
        return

    from pytorch_lightning import LightningModule

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    import horovod.spark.lightning as hvd

    from horovod.spark.common.backend import SparkBackend
    from horovod.spark.common.store import Store

    import pyspark.sql.types as T
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.sql.functions import udf

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

    # Define the PyTorch model without any Horovod-specific parameters
    class Net(LightningModule):
        def __init__(self, lr=1.0):
            super(Net, self).__init__()
            self.lr = lr
            self.save_hyperparameters()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.float().reshape((-1, 1, 28, 28))
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

        def configure_optimizers(self):
            return optim.Adadelta(self.parameters(), lr=self.lr)

        def training_step(self, batch, batch_idx):
            if batch_idx == 0:
                print(f"training data batch size: {batch['label'].shape}")
            x, y = batch['features'], batch['label']
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y.long())
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            if batch_idx == 0:
                print(f"validation data batch size: {batch['label'].shape}")
            x, y = batch['features'], batch['label']
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y.long())
            self.log('val_loss', loss)

        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() if len(outputs) > 0 else float('inf')
            self.log('avg_val_loss', avg_loss)

    def train(learning_rate):
        model = Net(learning_rate)

        # Train a Horovod Spark Estimator on the DataFrame
        backend = SparkBackend(num_proc=num_proc,
                               stdout=sys.stdout, stderr=sys.stderr,
                               use_gloo=args.use_gloo, use_mpi=args.use_mpi,
                               prefix_output_with_timestamp=True)

        from pytorch_lightning.callbacks import Callback

        epochs = args.epochs

        class MyDummyCallback(Callback):
            def __init__(self):
                self.epcoh_end_counter = 0
                self.train_epcoh_end_counter = 0
                self.validation_epoch_end_counter = 0

            def on_init_start(self, trainer):
                print('Starting to init trainer!')

            def on_init_end(self, trainer):
                print('Trainer is initialized.')

            def on_epoch_end(self, trainer, model):
                print('A train or eval epoch ended.')
                self.epcoh_end_counter += 1

            def on_train_epoch_end(self, trainer, model, unused=None):
                print('A train epoch ended.')
                self.train_epcoh_end_counter += 1

            def on_validation_epoch_end(self, trainer, model, unused=None):
                print('A val epoch ended.')
                self.validation_epoch_end_counter += 1

            def on_train_end(self, trainer, model):
                print("Training ends:"
                      f"epcoh_end_counter={self.epcoh_end_counter}, "
                      f"train_epcoh_end_counter={self.train_epcoh_end_counter}, "
                      f"validation_epoch_end_counter={self.validation_epoch_end_counter} \n")
                assert self.train_epcoh_end_counter <= epochs
                assert self.epcoh_end_counter == self.train_epcoh_end_counter + self.validation_epoch_end_counter

        callbacks = [MyDummyCallback()]

        if LooseVersion(torch.__version__) < LooseVersion('1.13'):
            """
            torch.distributed.ReduceOp is used in ModelCheckpoint and EarlyStopping.
            Since torch 1.13, it doesn't support condition check in Lightning code.
            Broken line in lightning code (https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/strategies/horovod.py#L179)
            Below error will be thrown:
            >>> from torch.distributed import ReduceOp
            >>> op = None
            >>> op in (ReduceOp.SUM, None)
            Traceback (most recent call last):
                File "<stdin>", line 1, in <module>
                TypeError: __eq__(): incompatible function arguments. The following argument types are supported:
                1. (self: torch._C._distributed_c10d.ReduceOp, arg0: c10d::ReduceOp::RedOpType) -> bool
                2. (self: torch._C._distributed_c10d.ReduceOp, arg0: torch._C._distributed_c10d.ReduceOp) -> bool
            Invoked with: <torch.distributed.distributed_c10d.ReduceOp object at 0x7fba78c9e0b0>, None
            """
            # ModelCheckpoint
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
            callbacks.append(ModelCheckpoint(monitor='val_loss', mode="min",
                                             save_top_k=1, verbose=True))
            # EarlyStopping
            from pytorch_lightning.callbacks.early_stopping import EarlyStopping
            callbacks.append(EarlyStopping(monitor='val_loss',
                                           min_delta=0.001,
                                           patience=3,
                                           verbose=True,
                                           mode='min'))

        torch_estimator = hvd.TorchEstimator(backend=backend,
                                             store=store,
                                             model=model,
                                             input_shapes=[[-1, 1, 28, 28]],
                                             feature_cols=['features'],
                                             label_cols=['label'],
                                             batch_size=args.batch_size,
                                             epochs=args.epochs,
                                             validation=0.1,
                                             verbose=1,
                                             callbacks=callbacks,
                                             profiler="simple" if args.enable_profiler else None)

        return torch_estimator.fit(train_df).setOutputCols(['label_prob'])

    def test_model(model, show_samples=False):
        pred_df = model.transform(test_df)
        argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
        pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
        evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
        accuracy = evaluator.evaluate(pred_df)
        print('Test accuracy:', accuracy)

        if show_samples:
            pred_df = pred_df.sampleBy('label', fractions={0.0: 0.1, 1.0: 0.1, 2.0: 0.1, 3.0: 0.1, 4.0: 0.1,
                                                           5.0: 0.1, 6.0: 0.1, 7.0: 0.1, 8.0: 0.1, 9.0: 0.1})
            pred_df.show(150)

        return 1 - accuracy

    # Train model
    torch_model = train(args.lr)

    # Test model
    test_model(torch_model, True)

    # Clean up
    spark.stop()


if __name__ == '__main__':
    args = parser.parse_args()
    train_model(args)
