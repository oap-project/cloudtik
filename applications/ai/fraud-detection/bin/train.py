import argparse
import os

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling \
    import (run as graph_run, ModelingArgs as GraphModelingArgs, get_mapped_embeddings_path)
from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling \
    import (run as xgboost_run, ModelingArgs as XGBoostModelingArgs)


def run(args):
    # if processed data path not specified, default to a file name to output dir.
    if not args.processed_data_path and args.temp_dir:
        args.processed_data_path = os.path.join(
            args.temp_dir, "processed_data.csv")
        print("processed-ata-path is not specified. Default to: {}".format(
            args.processed_data_path))

    if not args.model_file and args.output_dir:
        args.model_file = os.path.join(
            args.output_dir, "output_model")
        print("Output model-file is not specified. Default to: {}".format(
            args.model_file))

    single_node = True
    if args.hosts:
        single_node = False

    if args.raw_data_path:
        # process data
        xgboost_args = XGBoostModelingArgs()
        xgboost_args.single_node = single_node
        xgboost_args.no_train = True
        xgboost_args.raw_data_path = args.raw_data_path
        xgboost_args.processed_data_path = args.processed_data_path
        xgboost_run(xgboost_args)

    # train embeddings with GraphSAGE
    graph_args = GraphModelingArgs()
    graph_args.single_node = single_node
    graph_args.hosts = args.hosts
    graph_args.no_process_data = True
    graph_args.processed_data_path = args.processed_data_path
    graph_args.temp_dir = args.temp_dir
    graph_args.output_dir = args.output_dir
    graph_args.tabular2graph = args.tabular2graph
    # other possible parameters user want to pass
    graph_run(graph_args)

    mapped_embeddings_path = get_mapped_embeddings_path(graph_args)

    # train XGBoost
    xgboost_args = XGBoostModelingArgs()
    xgboost_args.single_node = single_node
    xgboost_args.no_process_data = True
    xgboost_args.processed_data_path = mapped_embeddings_path
    xgboost_args.temp_dir = args.temp_dir
    xgboost_args.model_file = args.model_file
    # other possible parameters user want to pass
    xgboost_run(xgboost_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN Training")
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
