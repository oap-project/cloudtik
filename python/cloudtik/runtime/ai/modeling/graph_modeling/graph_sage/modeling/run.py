import argparse
import os
import sys

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.utils import \
    existing_file
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.build_graph import \
    build_graph
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.partition_graph import \
    partition_graph
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.map_embeddings import \
    map_embeddings_distributed, map_embeddings
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.launch import \
    launch_jobs, launch_local

GNN_HOME_PATH = os.path.abspath(os.path.dirname(__file__))


def _get_dataset_output_dir(temp_dir):
    return os.path.join(temp_dir, "dataset")


def _get_dataset_dir(temp_dir, dataset_name):
    dataset_output_dir = _get_dataset_output_dir(temp_dir)
    return os.path.join(dataset_output_dir, dataset_name)


def _get_partition_dir(temp_dir):
    return os.path.join(temp_dir, "partition")


def _get_partition_config(temp_dir, graph_name):
    partition_dir = _get_partition_dir(temp_dir)
    return os.path.join(partition_dir, graph_name + ".json")


def _get_ip_config(temp_dir):
    return os.path.join(temp_dir,  "ip_config.txt")


def _get_model_file(output_dir):
    return os.path.join(output_dir, "model_graphsage_2L_64.pt")


def _get_node_embeddings_dir(output_dir):
    return output_dir


def _get_node_embeddings_file(output_dir, node_embeddings_name):
    node_embeddings_dir = _get_node_embeddings_dir(output_dir)
    return os.path.join(
        node_embeddings_dir, node_embeddings_name + ".pt")


def _get_mapped_output_file(output_dir):
    return os.path.join(output_dir, "tabular_with_gnn_embeddings.csv")


def _get_hosts(hosts):
    if not hosts:
        # TODO: use cloudtik API to retrieve the hosts to run
        raise RuntimeError("Distributed training must specify the hosts argument.")

    host_list = hosts.split(',')
    return host_list


def _save_ip_config(ip_config_file, hosts):
    host_list = _get_hosts(hosts)
    with open(ip_config_file, "w+") as f:
        for host in host_list:
            f.write("{}\n".format(host))


def _get_num_parts(hosts):
    host_list = _get_hosts(hosts)
    return len(host_list)


def _build_graph(args):
    if not args.input_file:
        raise ValueError(
            "Must specify the input file which contains the processed data.")
    if not args.temp_dir:
        raise ValueError(
            "Must specify the temp dir which is used to store intermediate data.")

    dataset_output_dir = _get_dataset_output_dir(args.temp_dir)
    build_graph(
        input_file=args.input_file,
        output_dir=dataset_output_dir,
        dataset_name=args.dataset_name,
        tabular2graph=args.tabular2graph
    )


def _partition_graph(args):
    if not args.temp_dir:
        raise ValueError(
            "Must specify the temp dir which stored the intermediate data.")
    partition_dir = _get_partition_dir(args.temp_dir)
    dataset_dir = _get_dataset_dir(
        args.temp_dir, args.dataset_name)
    if not args.num_parts:
        args.num_parts = _get_num_parts(args.hosts)
    partition_graph(
        dataset_dir=dataset_dir,
        output_dir=partition_dir,
        graph_name=args.graph_name,
        num_parts=args.num_parts,
        num_hops=args.num_hops
    )


def _train(args):
    if not args.temp_dir:
        raise ValueError(
            "Must specify the temp dir which stored the intermediate data.")
    # make sure the output dir exists
    if not args.output_dir:
        raise ValueError(
            "Must specify the output dir for storing results.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Call launch which run a single local training processes
    model_file = _get_model_file(args.output_dir)
    node_embeddings_file = _get_node_embeddings_file(
        args.output_dir, args.node_embeddings_name)
    dataset_dir = _get_dataset_dir(
        args.temp_dir, args.dataset_name)

    workspace = GNN_HOME_PATH
    exec_script = os.path.join(GNN_HOME_PATH, "model", "train.py")
    job_command = (
        'numactl -N 0 {python_exe} -u '
        '{exec_script} '
        '--model_file {model_file} '
        '--node_embeddings_file {node_embeddings_file} '
        '--dataset_dir {dataset_dir} '
        '--num_epoch {num_epochs} '
        '--num_hidden {num_hidden} '
        '--num_layers {num_layers} '
        '--lr {lr} '
        '--fan_out {fan_out} '
        '--batch_size {batch_size} '
        '--batch_size_eval {batch_size_eval} '
        '--eval_every {eval_every} '
        '--num_dl_workers {num_dl_workers} '
        .format(
            python_exe=sys.executable,
            exec_script=exec_script,
            model_file=model_file,
            node_embeddings_file=node_embeddings_file,
            dataset_dir=dataset_dir,
            num_epochs=args.num_epochs,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            lr=args.lr,
            fan_out=args.fan_out,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
            eval_every=args.eval_every,
            num_dl_workers=args.num_dl_workers,
        )
    )

    launch_local(
        job_command, workspace,
        num_workers=args.num_dl_workers,
        num_omp_threads=args.num_omp_threads
    )


def _train_distributed(args):
    # For distributed training, the temp_dir and the output_dir must be shared
    # directory being able to accessed by the nodes.
    if not args.temp_dir:
        raise ValueError(
            "Must specify the temp dir which stored the intermediate data.")
    # make sure the output dir exists
    if not args.output_dir:
        raise ValueError(
            "Must specify the output dir for storing results.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Call launch which run the distributed training processes
    model_file = _get_model_file(args.output_dir)
    node_embeddings_file = _get_node_embeddings_file(
        args.output_dir, args.node_embeddings_name)
    dataset_dir = _get_dataset_dir(
        args.temp_dir, args.dataset_name)
    part_config = _get_partition_config(args.temp_dir, args.graph_name)
    ip_config = _get_ip_config(args.temp_dir)

    # Save IP config to shared ip config file
    _save_ip_config(ip_config, args.hosts)

    workspace = GNN_HOME_PATH
    exec_script = os.path.join(GNN_HOME_PATH, "model", "distributed", "train.py")
    job_command = (
        'numactl -N 0 {python_exe} '
        '{exec_script} '
        '--model_file {model_file} '
        '--node_embeddings_file {node_embeddings_file} '
        '--dataset_dir {dataset_dir} '
        '--graph_name {graph_name} '
        '--ip_config {ip_config} '
        '--part_config {part_config} '
        '--num_epochs {num_epochs} '
        '--num_hidden {num_hidden} '
        '--num_layers {num_layers} '
        '--lr {lr} '
        '--fan_out {fan_out} '
        '--batch_size {batch_size} '
        '--batch_size_eval {batch_size_eval} '
        '--eval_every {eval_every} '
        '--log_every {log_every} '
        '--remove_edge '
        .format(
            python_exe=sys.executable,
            exec_script=exec_script,
            model_file=model_file,
            node_embeddings_file=node_embeddings_file,
            dataset_dir=dataset_dir,
            graph_name=args.graph_name,
            ip_config=ip_config,
            part_config=part_config,
            num_epochs=args.num_epochs,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            lr=args.lr,
            fan_out=args.fan_out,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
            eval_every=args.eval_every,
            log_every=args.log_every,
        )
    )
    launch_jobs(
        job_command, workspace,
        ip_config=ip_config,
        part_config=part_config,
        num_servers=args.num_servers,
        num_trainers=args.num_trainers,
        num_samplers=args.num_samplers,
        num_server_threads=args.num_server_threads,
        num_omp_threads=args.num_omp_threads
    )


def _map_and_save_embeddings(args):
    if not args.input_file:
        raise ValueError(
            "Must specify the input file which contains the processed data.")
    if not args.output_dir:
        raise ValueError(
            "Must specify the output dir which stored the node embeddings.")

    mapped_output_file = _get_mapped_output_file(args.output_dir)
    node_embeddings_dir = _get_node_embeddings_dir(args.output_dir)
    if args.single_node:
        map_embeddings(
            processed_data_file=args.input_file,
            node_embeddings_dir=node_embeddings_dir,
            node_embeddings_name=args.node_embeddings_name,
            output_file=mapped_output_file,
            tabular2graph=args.tabular2graph
        )
    else:
        if not args.temp_dir:
            raise ValueError(
                "Must specify the temp dir which stored the intermediate data.")
        partition_dir = _get_partition_dir(args.temp_dir)
        map_embeddings_distributed(
            processed_data_file=args.input_file,
            partition_dir=partition_dir,
            node_embeddings_dir=node_embeddings_dir,
            node_embeddings_name=args.node_embeddings_name,
            output_file=mapped_output_file,
            tabular2graph=args.tabular2graph
        )


def run(args):
    # run build the graph
    if not args.no_build_graph:
        _build_graph(args)

    if not args.single_node and not args.no_partition_graph:
        _partition_graph(args)

    if not args.no_train_graph:
        if args.single_node:
            _train(args)
        else:
            _train_distributed(args)

    if not args.no_map_embeddings:
        _map_and_save_embeddings(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN Training")
    parser.add_argument(
        "--single-node", "--single_node",
        default=False, action="store_true",
        help="To do single node training")
    parser.add_argument(
        "--no_build_graph", "--no_build_graph",
        default=False, action="store_true",
        help="whether to build graph")
    parser.add_argument(
        "--no-partition-graph", "--no_partition_graph",
        default=False, action="store_true",
        help="whether to partition graph")
    parser.add_argument(
        "--no-train-graph", "--no_train_graph",
        default=False, action="store_true",
        help="whether to train graph")
    parser.add_argument(
        "--no-map-embeddings", "--no_map_embeddings",
        default=False, action="store_true",
        help="whether to map embeddings")

    parser.add_argument(
        "--input-file", "--input_file",
        type=existing_file,
        help="Input file with path of the processed data in csv) ")
    parser.add_argument(
        "--temp-dir", "--temp_dir",
        type=str,
        help="The path to the intermediate data")
    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        help="The path to the output")
    parser.add_argument(
        "--dataset-name", "--dataset_name",
        type=str, default="tabformer_hetero",
        help="The dataset name")
    parser.add_argument(
        "--tabular2graph",
        type=str, required=True,
        help="The path to the tabular2graph.yaml")

    # Train
    parser.add_argument(
        "--node-embeddings-name", "--node_embeddings_name",
        type=str, default="node_emb",
        help="The path to the node embedding file")

    # Distributed training
    parser.add_argument(
        "--hosts",
        type=str,
        help="List of hosts separated with comma for launching tasks. ")
    # Partition graph parameters
    parser.add_argument(
        "--graph_name", "--graph_name",
        type=str, default="tabformer_full_homo",
        help="The graph name")
    parser.add_argument(
        "--num-parts", "--num_parts",
        type=int, default=0,
        help="number of partitions")
    parser.add_argument(
        "--num-hops", "--num_hops",
        type=int, default=1,
        help="number of hops of nodes we include in a partition as HALO nodes")

    parser.add_argument(
        "--num-trainers", "--num_trainers",
        type=int, default=1,
        help="The number of trainer processes per machine",
    )
    parser.add_argument(
        "--num-samplers", "--num_samplers",
        type=int, default=2,
        help="The number of sampler processes per trainer process",
    )
    parser.add_argument(
        "--num-servers", "--num_servers",
        type=int, default=1,
        help="The number of server processes per machine",
    )
    parser.add_argument(
        "--num-server-threads", "--num_server_threads",
        type=int, default=1,
        help="The number of OMP threads in the server process. "
             "It should be small if server processes and trainer processes run "
             "on the same machine. By default, it is 1.",
    )
    parser.add_argument(
        "--num-omp-threads", "--num_omp_threads",
        type=int,
        help="The number of OMP threads per trainer",
    )

    # These defaults are set for distributed training
    parser.add_argument("--num-epochs", "--num_epochs",
                        type=int, default=10)
    parser.add_argument("--num-hidden", "--num_hidden",
                        type=int, default=64)
    parser.add_argument("--num-layers", "--num_layers",
                        type=int, default=2)
    parser.add_argument("--fan-out", "--fan_out",
                        type=str, default="55,65")
    parser.add_argument("--batch-size", "--batch_size",
                        type=int, default=2048)
    parser.add_argument("--batch-size-eval", "--batch_size_eval",
                        type=int, default=1000000)
    parser.add_argument("--eval-every", "--eval_every",
                        type=int, default=1)
    parser.add_argument("--lr",
                        type=float, default=0.0005)

    # distributed only
    parser.add_argument("--log-every", "--log_every",
                        type=int, default=20)

    # single only
    parser.add_argument("--num-dl-workers", "--num_dl_workers",
                        type=int, default=4)

    args = parser.parse_args()
    print(args)

    run(args)
