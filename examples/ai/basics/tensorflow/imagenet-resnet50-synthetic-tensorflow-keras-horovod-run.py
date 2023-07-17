# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
from timeit import default_timer as timer

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow ImageNet Synthetic Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')

parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')

parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')

import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras import applications

from packaging import version
if version.parse(tf.keras.__version__.replace("-tf", "+tf")) < version.parse("2.11"):
    from tensorflow.keras import optimizers
else:
    from tensorflow.keras.optimizers import legacy as optimizers


def train_horovod(learning_rate):
    args.cuda = not args.no_cuda
    device = 'GPU' if args.cuda else 'CPU'

    # Horovod: initialize Horovod.
    hvd.init()

    if hvd.rank() == 0:
        print('Model: %s' % args.model)
        print('Batch size: %d' % args.batch_size)
        print('Number of %ss: %d' % (device, hvd.size()))

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    if args.cuda:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Set up standard model.
    model = getattr(applications, args.model)(weights=None)
    opt = optimizers.SGD(learning_rate)

    # Synthetic dataset
    data = tf.random.uniform([args.batch_size, 224, 224, 3])
    target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((data, target)).cache().repeat().batch(args.batch_size)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt, compression=compression)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer=opt,
                  experimental_run_tf_function=False)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
    ]

    class TimingCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.img_secs = []

        def on_train_end(self, logs=None):
            img_sec_mean = np.mean(self.img_secs)
            img_sec_conf = 1.96 * np.std(self.img_secs)
            print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
            print('Total img/sec on %d %s(s): %.1f +-%.1f' %
                 (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

        def on_epoch_begin(self, epoch, logs=None):
            self.starttime = timer()

        def on_epoch_end(self, epoch, logs=None):
            time = timer() - self.starttime
            img_sec = args.batch_size * args.num_batches_per_iter / time
            print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, device))
            # skip warm up epoch
            if epoch > 0:
                self.img_secs.append(img_sec)

    # Horovod: write logs on worker 0.
    if hvd.rank() == 0:
        timing = TimingCallback()
        callbacks.append(timing)

    # Train the model.
    model.fit(
        dataset,
        batch_size=args.batch_size,
        steps_per_epoch=args.num_batches_per_iter,
        callbacks=callbacks,
        epochs=args.num_iters,
        verbose=0,
    )


if __name__ == '__main__':
    args = parser.parse_args()

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
        total_workers = cluster_info.get("total-workers-ready")
        if total_workers:
            args.num_proc = total_workers
        if not args.num_proc:
            args.num_proc = 1

    worker_ips = cluster.get_worker_node_ips(node_status="up-to-date")

    # Run training function
    import cloudtik.runtime.ai.runner as runner

    # Set the parameters
    num_proc = args.num_proc
    print("Train processes: {}".format(num_proc))

    # Generate the host list
    worker_num_proc = int(num_proc / len(worker_ips))
    if not worker_num_proc:
        worker_num_proc = 1
    host_slots = ["{}:{}".format(worker_ip, worker_num_proc) for worker_ip in worker_ips]
    hosts = ",".join(host_slots)
    print("Hosts to run:", hosts)

    learning_rate = args.base_lr
    print("Train learning rate: {}".format(learning_rate))

    runner.run(
        train_horovod, args=(learning_rate,),
        num_proc=num_proc, hosts=hosts, launcher="horovod",
        use_gloo=args.use_gloo, use_mpi=args.use_mpi,
        verbose=2)
