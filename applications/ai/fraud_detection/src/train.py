import argparse
import os

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling \
    import (run as graph_run, ModelingArgs as GraphModelingArgs, get_data_with_embeddings_path)
from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling \
    import (run as xgboost_run, ModelingArgs as XGBoostModelingArgs)


def _process_data(args, single_node):
    xgboost_args = XGBoostModelingArgs()
    xgboost_args.single_node = single_node
    xgboost_args.no_train = True
    xgboost_args.no_predict = True
    xgboost_args.raw_data_path = args.raw_data_path
    xgboost_args.processed_data_path = args.processed_data_path
    xgboost_run(xgboost_args)


def _get_data_with_embeddings(args):
    graph_args = GraphModelingArgs()
    graph_args.output_dir = args.output_dir
    return get_data_with_embeddings_path(graph_args)


def _run_graph(args, single_node):
    # train embeddings with GraphSAGE
    graph_args = GraphModelingArgs()
    graph_args.single_node = single_node
    graph_args.hosts = args.hosts

    graph_args.no_process_data = True
    graph_args.no_train = args.no_train
    graph_args.no_predict = args.no_predict

    graph_args.processed_data_path = args.processed_data_path
    graph_args.temp_dir = args.temp_dir
    graph_args.output_dir = args.output_dir
    graph_args.tabular2graph = args.tabular2graph

    # other possible parameters user want to pass
    graph_run(graph_args)


def _run_xgboost(args, single_node):
    # train XGBoost
    xgboost_args = XGBoostModelingArgs()
    xgboost_args.single_node = single_node

    xgboost_args.no_process_data = True
    xgboost_args.no_train = args.no_train
    xgboost_args.no_predict = args.no_predict

    xgboost_args.processed_data_path = _get_data_with_embeddings(args)
    xgboost_args.temp_dir = args.temp_dir
    xgboost_args.output_dir = args.output_dir
    xgboost_args.model_file = args.model_file

    # other possible parameters user want to pass
    xgboost_run(xgboost_args)


def run(args):
    # if processed data path not specified, default to a file name to output dir.
    if not args.processed_data_path and args.temp_dir:
        args.processed_data_path = os.path.join(
            args.temp_dir, "processed_data.csv")
        print("processed-ata-path is not specified. Default to: {}".format(
            args.processed_data_path))

    if not args.model_file and args.output_dir:
        args.model_file = os.path.join(
            args.output_dir, "xgboost_model.json")
        print("Output model-file is not specified. Default to: {}".format(
            args.model_file))

    single_node = False if args.hosts else True

    if args.raw_data_path:
        # process data
        _process_data(args, single_node)

    if not args.no_graph:
        _run_graph(
            args, single_node)

    if not args.no_xgboost:
        _run_xgboost(
            args, single_node)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN Training")

    parser.add_argument(
        "--no-graph", "--no_graph",
        default=False, action="store_true",
        help="whether to run neural graph.")
    parser.add_argument(
        "--no-xgboost", "--no_xgboost",
        default=False, action="store_true",
        help="whether to run xgboost.")
    parser.add_argument(
        "--no-train", "--no_train",
        default=False, action="store_true",
        help="whether to train the data.")
    parser.add_argument(
        "--no-predict", "--no_predict",
        default=False, action="store_true",
        help="whether to predict on data.")

    parser.add_argument(
        "--raw-data-path", "--raw_data_path",
        type=str,
        help="The path to the raw transaction data")
    parser.add_argument(
        "--processed-data-path", "--processed_data_path",
        type=str,
        help="The path to the output processed data")

    parser.add_argument(
        "--temp-dir", "--temp_dir",
        type=str,
        help="The path to the intermediate data")
    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        help="The path to the output")

    parser.add_argument(
        "--model-file", "--model_file",
        type=str,
        help="The path to the output model file")

    parser.add_argument(
        "--tabular2graph",
        type=str,
        help="The path to the tabular2graph.yaml")
    parser.add_argument(
        "--hosts",
        type=str,
        help="List of hosts separated with comma for launching tasks. ")

    args = parser.parse_args()
    print(args)

    run(args)
