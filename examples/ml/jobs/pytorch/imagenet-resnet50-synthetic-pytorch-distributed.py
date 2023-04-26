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

parser.add_argument('--backend', default='', type=str,
                    help='Distributed PyTorch backend')
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

parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--int8', action='store_true', default=False,
                    help='enable ipex int8 path')
parser.add_argument('--bf16', action='store_true', default=False,
                    help='enable ipex bf16 path')
parser.add_argument('--bf32', action='store_true', default=False,
                    help='enable ipex bf32 path')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='enable ipex fp16 path')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable ipex jit fusion path')

import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import horovod.torch as hvd
import numpy as np


def train_distributed(learning_rate):
    import torch
    import os
    import torch.distributed as dist

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    backend = args.backend if args.backend else "gloo"
    if args.cuda and not args.backend:
        backend = "nccl"

    # Just use Horovod to get the world_size and rank
    hvd.init()

    # setup
    os.environ['RANK'] = str(hvd.rank())
    os.environ['WORLD_SIZE'] = str(hvd.size())
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = "29500"

    dist.init_process_group(backend, rank=hvd.rank(), world_size=hvd.size())

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())

        torch.backends.cudnn.benchmark = True

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

    if args.ipex:
        import intel_extension_for_pytorch as ipex

        # for ipex path, always convert model to channels_last for bf16, fp32.
        # TODO: int8 path: https://jira.devtools.intel.com/browse/MFDNN-6103
        if not args.int8:
            model = model.to(memory_format=torch.channels_last)

        if args.bf32:
            ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
            print("using bf32 fmath mode\n")

        if args.jit and args.int8:
            assert False, "jit path is not available for int8 path using ipex"
    else:
        # for official pytorch, int8 and jit path is not enabled.
        assert not args.int8, "int8 path is not enabled for official pytorch"
        assert not args.jit, "jit path is not enabled for official pytorch"

    optimizer = optim.SGD(model.parameters(), lr=learning_rate * lr_scaler)

    scaler = None
    if args.ipex:
        # for bf32 path, calling ipex.optimize to calling ipex conv which enabled bf32 path
        if args.bf32:
            sample_input = torch.randn(args.batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            model, optimizer = ipex.optimize(model, dtype=torch.float32,
                                             optimizer=optimizer, sample_input=sample_input)

        if args.bf16:
            sample_input = torch.randn(args.batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            model, optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer,
                                             weights_prepack=True, split_master_weight_for_bf16=False,
                                             sample_input=sample_input)

        if args.fp16:
            scaler = torch.cpu.amp.GradScaler()
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.half, auto_kernel_selection=True,
                                             fuse_update_step=False)

    model = torch.nn.parallel.DistributedDataParallel(model)

    # Set up fixed fake data
    if args.ipex:
        data = torch.randn(args.batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
        if args.bf16:
            data = data.to(torch.bfloat16)
        if args.fp16:
            data = data.to(torch.half)
    else:
        data = torch.randn(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()

    def benchmark_step():
        optimizer.zero_grad()

        if args.ipex:
            if args.bf16:
                with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                    output = model(data)
                output = output.to(torch.float32)
            elif args.fp16:
                with torch.cpu.amp.autocast(dtype=torch.half):
                    output = model(data)
                output = output.to(torch.float32)
            else:
                output = model(data)
        else:
            output = model(data)
        loss = F.cross_entropy(output, target)

        if args.ipex and args.fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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

    # cleanup
    dist.destroy_process_group()


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
        total_worker_cpus = cluster_info.get("total-worker-cpus-ready")
        if total_worker_cpus:
            args.num_proc = int(total_worker_cpus / 2)
        if not args.num_proc:
            args.num_proc = 1

    worker_ips = cluster.get_worker_node_ips(node_status="up-to-date")
    if len(worker_ips) == 0:
        raise RuntimeError("No up-to-date worker.")

    master_addr = worker_ips[0]

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
        train_distributed, args=(learning_rate,),
        num_proc=num_proc, hosts=hosts,
        use_gloo=args.use_gloo, use_mpi=args.use_mpi,
        verbose=2)
