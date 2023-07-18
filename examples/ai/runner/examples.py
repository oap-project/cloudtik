import argparse


from cloudtik.runtime.spark.api import ThisSparkCluster
import cloudtik.runtime.ai.runner as runner


def train(name):
    import time

    world_size = runner.get_world_size()
    rank = runner.get_rank()
    local_size = runner.get_local_size()
    local_rank = runner.get_local_rank()

    print("Running with {} launcher: world_size={}, rank={}, local_size={}, local_rank={}".format(
        name, world_size, rank, local_size, local_rank))
    time.sleep(5)
    return rank


def main(args):
    cluster = ThisSparkCluster()

    # Wait for all cluster workers to be ready
    cluster.wait_for_ready(min_workers=1)

    # The ready workers
    worker_ips = cluster.get_worker_node_ips(node_status="up-to-date")

    if not args.num_proc:
        args.num_proc = len(worker_ips) * 2
        if not args.num_proc:
            args.num_proc = 1

    num_proc = args.num_proc
    print("Processes to run: {}".format(num_proc))

    # Generate the host list
    worker_num_proc = int(num_proc / len(worker_ips))
    if not worker_num_proc:
        worker_num_proc = 1
    host_slots = ["{}:{}".format(worker_ip, worker_num_proc) for worker_ip in worker_ips]
    hosts = ",".join(host_slots)
    print("Hosts to run:", hosts)

    name = "local"
    print("Running with launcher:", name)
    results = runner.run(
        train, args=(name,),
        num_proc=num_proc, launcher=name)
    print("{} launcher returns: {}".format(name, results))

    name = "distributed"
    print("Running with launcher:", name)
    results = runner.run(
        train, args=(name,),
        num_proc=num_proc, hosts=hosts, launcher=name)
    print("{} launcher returns: {}".format(name, results))

    name = "mpi"
    print("Running with launcher:", name)
    results = runner.run(
        train, args=(name,),
        num_proc=num_proc, hosts=hosts, launcher=name)
    print("{} launcher returns: {}".format(name, results))

    name = "rsh"
    print("Running with launcher:", name)
    results = runner.run(
        train, args=(name,),
        num_proc=num_proc, hosts=hosts, launcher=name)
    print("{} launcher returns: {}".format(name, results))

    if not args.no_horovod:
        name = "horovod"
        print("Running with launcher:", name)
        results = runner.run(
            train, args=(name,),
            num_proc=num_proc, hosts=hosts, launcher=name)
        print("{} launcher returns: {}".format(name, results))

    if not args.no_horovod_spark:
        # Initialize SparkSession
        from pyspark import SparkConf
        from pyspark.sql import SparkSession

        conf = SparkConf().setAppName(
            'horovod-on-spark-example').set('spark.sql.shuffle.partitions', '16')
        spark = SparkSession.builder.config(conf=conf).getOrCreate()

        name = "horovod.spark"
        print("Running with launcher:", name)
        results = runner.run(
            train, args=(name,),
            num_proc=num_proc, hosts=hosts, launcher=name)
        print("{} launcher returns: {}".format(name, results))

        # Clean up
        spark.stop()

    if not args.no_horovod_ray:
        import ray
        ray.init(address="auto")

        name = "horovod.ray"
        print("Running with launcher:", name)
        results = runner.run(
            train, args=(name,),
            num_proc=num_proc, hosts=hosts, launcher=name)
        print("{} launcher returns: {}".format(name, results))

        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Examples for using runner API with different launchers '
                    'use unified run API')
    parser.add_argument('--num-proc', '--num_proc', type=int,
                        help='number of processes to run')
    parser.add_argument('--no-horovod', '--no_horovod',
                        action='store_true', default=False,
                        help='Not run Horovod launcher')
    parser.add_argument('--no-horovod-spark', '--no_horovod_spark',
                        action='store_true', default=False,
                        help='Not run Horovod Spark launcher')
    parser.add_argument('--no-horovod-ray', '--no_horovod_ray',
                        action='store_true', default=False,
                        help='Not run Horovod Ray launcher')
    args = parser.parse_args()
    main(args)
