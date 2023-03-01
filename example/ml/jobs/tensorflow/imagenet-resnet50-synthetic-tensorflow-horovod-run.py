import argparse
import os
import timeit

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

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
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
import horovod.tensorflow as hvd
from tensorflow.keras import applications


def train_horovod(learning_rate):
    args.cuda = not args.no_cuda

    # Horovod: initialize Horovod.
    hvd.init()

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
    opt = tf.optimizers.SGD(learning_rate)

    data = tf.random.uniform([args.batch_size, 224, 224, 3])
    target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

    @tf.function
    def benchmark_step(first_batch):
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

        # Horovod: use DistributedGradientTape
        with tf.GradientTape() as tape:
            probs = model(data, training=True)
            loss = tf.losses.sparse_categorical_crossentropy(target, probs)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape, compression=compression)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

    def log(s, nl=True):
        if hvd.rank() != 0:
            return
        print(s, end='\n' if nl else '')

    log('Model: %s' % args.model)
    log('Batch size: %d' % args.batch_size)
    device = 'GPU' if args.cuda else 'CPU'
    log('Number of %ss: %d' % (device, hvd.size()))

    with tf.device(device):
        # Warm-up
        log('Running warmup...')
        benchmark_step(first_batch=True)
        timeit.timeit(lambda: benchmark_step(first_batch=False),
                      number=args.num_warmup_batches)

        # Benchmark
        log('Running benchmark...')
        img_secs = []
        for x in range(args.num_iters):
            time = timeit.timeit(lambda: benchmark_step(first_batch=False),
                                 number=args.num_batches_per_iter)
            img_sec = args.batch_size * args.num_batches_per_iter / time
            log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
            img_secs.append(img_sec)

        # Results
        img_sec_mean = np.mean(img_secs)
        img_sec_conf = 1.96 * np.std(img_secs)
        log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
        log('Total img/sec on %d %s(s): %.1f +-%.1f' %
            (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))


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

    #  Hyperopt training function
    import horovod

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

    horovod.run(
        train_horovod, args=(learning_rate,),
        num_proc=num_proc, hosts=hosts,
        use_gloo=args.use_gloo, use_mpi=args.use_mpi,
        verbose=2)
