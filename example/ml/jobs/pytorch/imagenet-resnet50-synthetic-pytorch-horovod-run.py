import argparse
import timeit

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Synthetic Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')

parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
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

parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--gloo', action='store_true', dest='use_gloo',
                    help='Run Horovod using the Gloo controller. This will '
                         'be the default if Horovod was not built with MPI support.')
parser.add_argument('--mpi', action='store_true', dest='use_mpi',
                    help='Run Horovod using the MPI controller. This will '
                         'be the default if Horovod was built with MPI support.')

# Since we don't have cuda:cudnn in the runtime
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import numpy as np


def train_horovod(learning_rate):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    hvd.init()

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        # Since we don't have cuda:cudnn in the runtime
        # cudnn.benchmark = True

    # Set up standard model.
    model = getattr(models, args.model)()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate * lr_scaler)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Set up fixed fake data
    data = torch.randn(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()

    def benchmark_step():
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def log(s, nl=True):
        if hvd.rank() != 0:
            return
        print(s, end='\n' if nl else '')

    log('Model: %s' % args.model)
    log('Batch size: %d' % args.batch_size)
    device = 'GPU' if args.cuda else 'CPU'
    log('Number of %ss: %d' % (device, hvd.size()))

    # Warm-up
    log('Running warmup...')
    timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
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
        total_workers = cluster_info.get("total-workers")
        if total_workers:
            args.num_proc = total_workers
        if not args.num_proc:
            args.num_proc = 1

    worker_ips = cluster.get_worker_node_ips()

    #  Hyperopt training function
    import horovod

    # Set the parameters
    num_proc = args.num_proc
    print("Train processes: {}".format(num_proc))

    # Generate the host list
    host_slots = ["{}:1".format(worker_ip) for worker_ip in worker_ips]
    hosts = ",".join(host_slots)
    print("Hosts to run:", hosts)

    learning_rate = args.base_lr
    print("Train learning rate: {}".format(learning_rate))

    horovod.run(
        train_horovod, args=(learning_rate,),
        num_proc=num_proc, hosts=hosts,
        use_gloo=args.use_gloo, use_mpi=args.use_mpi,
        verbose=2)
