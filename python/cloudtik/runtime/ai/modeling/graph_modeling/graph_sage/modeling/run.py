import argparse
import os
import sys
import tempfile

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.data.process \
    import process_data
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.transductive.predictor import predict
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.utils import \
    existing_path
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.build_graph import \
    build_graph
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.partition_graph import \
    partition_graph
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.embeddings import \
    apply_embeddings, _map_node_embeddings
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


def _get_data_with_embeddings_file(output_dir, data_with_embeddings_name):
    return os.path.join(output_dir, data_with_embeddings_name)


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


def _check_temp_dir(args):
    if args.single_node:
        if not args.temp_dir:
            # for single node, get get a default temp dir from /tmp
            args.temp_dir = tempfile.mkdtemp()
            print("temp-dir is not specified. Default to: {}".format(
                args.temp_dir))
    else:
        if not args.temp_dir:
            raise ValueError(
                "Must specify the temp-dir for storing the shared intermediate data")


def _check_tabular2graph(args):
    if not args.tabular2graph:
        # default to the built-in tabular2graph.yaml if not specified
        args.tabular2graph = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config/tabular2graph.yaml")
        print("tabular2graph is not specified. Default to: {}".format(
            args.tabular2graph))


def _check_model_file(args):
    if not args.model_file:
        args.model_file = _get_model_file(args.output_dir)
        print("model-file is not specified. Default to: {}".format(
            args.model_file))


def _check_train_output(args):
    if not args.train_output:
        args.train_output = os.path.join(
            args.output_dir, "train_node_embeddings.pt")
        print("train-output is not specified. Default to: {}".format(
            args.train_output))


def _check_predict_output(args):
    if not args.predict_output:
        args.predict_output = os.path.join(
            args.output_dir, "predict_node_embeddings.pt")
        print("predict-output is not specified. Default to: {}".format(
            args.predict_output))


def get_data_with_embeddings_path(args):
    return _get_data_with_embeddings_file(
        args.output_dir, args.data_with_embeddings_name)


def _process_data(args):
    if not args.raw_data_path:
        raise ValueError(
            "Must specify the raw data file which contains raw data to be processed.")
    if not args.processed_data_path:
        raise RuntimeError("Please specify the processed-data-path for storing of processed data.")
    process_data(
        raw_data_path=args.raw_data_path,
        output_file=args.processed_data_path,
    )


def _build_graph(args):
    if not args.processed_data_path:
        raise ValueError(
            "Must specify the input file which contains the processed data.")
    _check_temp_dir(args)
    _check_tabular2graph(args)

    dataset_output_dir = _get_dataset_output_dir(args.temp_dir)
    build_graph(
        input_file=args.processed_data_path,
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


def _train_local(args):
    if not args.temp_dir:
        raise ValueError(
            "Must specify the temp dir which stored the intermediate data.")
    # make sure the output dir exists
    if not args.output_dir:
        raise ValueError(
            "Must specify the output dir for storing results.")
    os.makedirs(args.output_dir, exist_ok=True)

    _check_model_file(args)
    _check_train_output(args)

    # Call launch which run a single local training processes
    dataset_dir = _get_dataset_dir(
        args.temp_dir, args.dataset_name)

    workspace = GNN_HOME_PATH
    exec_script = os.path.join(GNN_HOME_PATH, "model",
                               "homogeneous", "transductive", "train.py")
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
            model_file=args.model_file,
            node_embeddings_file=args.train_output,
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

    _check_model_file(args)
    _check_train_output(args)

    # Call launch which run the distributed training processes
    dataset_dir = _get_dataset_dir(
        args.temp_dir, args.dataset_name)
    part_config = _get_partition_config(args.temp_dir, args.graph_name)
    ip_config = _get_ip_config(args.temp_dir)

    # Save IP config to shared ip config file
    _save_ip_config(ip_config, args.hosts)

    node_embeddings_file = os.path.join(
        args.output_dir, "node_embeddings.pt")

    workspace = GNN_HOME_PATH
    exec_script = os.path.join(GNN_HOME_PATH, "model",
                               "homogeneous", "transductive", "distributed", "train.py")
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
            model_file=args.model_file,
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

    # the train process finished, map the partitioned node embeddings to global
    partition_dir = _get_partition_dir(args.temp_dir)
    _map_node_embeddings(
        node_embeddings_file,
        partition_dir=partition_dir,
        output_file=args.train_output)


def _train(args):
    if args.single_node:
        _train_local(args)
    else:
        _train_distributed(args)


def _predict_node_embeddings(args):
    # make sure the output dir exists
    if not args.output_dir:
        raise ValueError(
            "Must specify the output dir for storing results.")
    os.makedirs(args.output_dir, exist_ok=True)

    _check_model_file(args)
    if not os.path.exists(args.model_file):
        raise ValueError(
            "The model file doesn't exist: {}.".format(args.model_file))
    _check_predict_output(args)

    dataset_dir = _get_dataset_dir(
        args.temp_dir, args.dataset_name)
    predict(dataset_dir,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            model_file=args.model_file,
            predict_output=args.predict_output,
            batch_size=args.batch_size_eval)


def _predict(args):
    # The training node embeddings are not enough
    # predict to get the node embeddings (cases such as new node added)
    # TODO: optimize if there is no new node, we can use the trained node embeddings
    # The current impl path assumes transductive learning
    _predict_node_embeddings(args)
    _apply_embeddings_to_data(args)


def _apply_embeddings_to_data(args):
    if not args.processed_data_path:
        raise ValueError(
            "Must specify the processed data file which contains the processed data.")
    if not args.output_dir:
        raise ValueError(
            "Must specify the output dir to store data with node embeddings.")
    _check_tabular2graph(args)
    _check_predict_output(args)

    node_embeddings_file = args.predict_output
    if not os.path.exists(node_embeddings_file):
        raise ValueError(
            "The node embeddings file doesn't exist: {}."
            "This file is generated by predicting on a graph.".format(node_embeddings_file))

    output_file = _get_data_with_embeddings_file(
        args.output_dir, args.data_with_embeddings_name)
    apply_embeddings(
        processed_data_path=args.processed_data_path,
        node_embeddings_file=node_embeddings_file,
        output_file=output_file,
        tabular2graph=args.tabular2graph
    )


def run(args):
    if not args.no_process_data:
        _process_data(args)

    # run build the graph
    if not args.no_build_graph:
        _build_graph(args)

    if not args.single_node and not args.no_partition_graph:
        _partition_graph(args)

    if not args.no_train:
        _train(args)

    if not args.no_predict:
        _predict(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN Training")
    parser.add_argument(
        "--single-node", "--single_node",
        default=False, action="store_true",
        help="To do single node training")
    parser.add_argument(
        "--no-process-data", "--no_process_data",
        default=False, action="store_true",
        help="whether to do data process")
    parser.add_argument(
        "--no-build-graph", "--no_build_graph",
        default=False, action="store_true",
        help="whether to build graph")
    parser.add_argument(
        "--no-partition-graph", "--no_partition_graph",
        default=False, action="store_true",
        help="whether to partition graph")
    parser.add_argument(
        "--no-train", "--no_train",
        default=False, action="store_true",
        help="whether to do training")
    parser.add_argument(
        "--no-predict", "--no_predict",
        default=False, action="store_true",
        help="whether to do predict")

    parser.add_argument(
        "--raw-data-path", "--raw_data_path",
        type=existing_path,
        help="The path to the raw transaction data")
    parser.add_argument(
        "--processed-data-path", "--processed_data_path",
        type=str,
        help="The path to the output processed data")
    parser.add_argument(
        "--model-file", "--model_file",
        type=str,
        help="The path to the output model file")
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
        type=str, default="tabformer",
        help="The dataset name")
    parser.add_argument(
        "--tabular2graph",
        type=str,
        help="The path to the tabular2graph.yaml")

    # Train
    parser.add_argument(
        "--train-output", "--train_output",
        type=str, default="node_embeddings",
        help="The path to the train output node embeddings file")

    # Predict
    parser.add_argument(
        "--predict-output", "--predict_output",
        type=str, default="node_embeddings",
        help="The path to the predict output node embeddings file")
    parser.add_argument(
        "--data-with-embeddings-name", "--data_with_embeddings_name",
        type=str, default="data_with_embeddings.csv",
        help="The path to save the data with embeddings file")

    # Distributed training
    parser.add_argument(
        "--hosts",
        type=str,
        help="List of hosts separated with comma for launching tasks. ")
    # Partition graph parameters
    parser.add_argument(
        "--graph-name", "--graph_name",
        type=str, default="tabformer_graph",
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
