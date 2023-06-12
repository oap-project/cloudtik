import argparse
import os
import tempfile

from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.data.process \
    import process_data
from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.utils import \
    existing_file, existing_path, load_config, read_csv_files, DATA_ENGINE_PANDAS, DATA_ENGINE_MODIN


def _process_data(args, data_engine):
    if not args.raw_data_path:
        raise ValueError(
            "Must specify the raw data path which contains data to be processed.")

    if not args.data_processing_config:
        # default to the built-in data_processing_config.yaml if not specified
        args.data_processing_config = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config/data-processing-config.yaml")
        print("data-processing-config is not specified. Default to: {}".format(
            args.data_processing_config))

    if args.in_memory or args.no_save:
        processed_data_file = None
    else:
        processed_data_file = args.processed_data_file
        if not processed_data_file:
            raise RuntimeError("Please specify the file path of processed data.")
    train_data, test_data = process_data(
        raw_data_path=args.raw_data_path,
        data_engine=data_engine,
        data_processing_config=args.data_processing_config,
        output_file=processed_data_file
    )
    return train_data, test_data


def _train_on_data(
        df, training_config,
        temp_dir, model_file, in_memory):
    print('start training model...')
    config = load_config(training_config)
    train_data_spec = config['data_spec']
    hpo_spec = config.get('hpo_spec')
    train_model_spec = config.get('model_spec')
    if hpo_spec is None and train_model_spec is None:
        raise RuntimeError("Must specify either hpo_spec or model_spec.")

    if hpo_spec is not None:
        print("Do training with hyper parameter optimization (HPO).")
    else:
        print("Do training without hyper parameter optimization (HPO).")

    on_ray = True if not args.single_node else False

    if on_ray:
        ray_params = get_ray_params(args)
        from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.\
            xgboost.modeling.model.ray.trainer import train
        train(
            train_data_spec, df,
            train_model_spec, in_memory,
            tmp_path=temp_dir, model_file=model_file,
            hpo_spec=hpo_spec,
            ray_params=ray_params)
    else:
        from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.\
            xgboost.modeling.model.trainer import train
        train(
            train_data_spec, df,
            train_model_spec, in_memory,
            tmp_path=temp_dir, model_file=model_file,
            hpo_spec=hpo_spec)


def get_ray_params(args):
    ray_params = {}
    if args.num_actors:
        ray_params["num_actors"] = args.num_actors
    if args.cpus_per_actor:
        ray_params["cpus_per_actor"] = args.cpus_per_actor
    if args.gpus_per_actor:
        ray_params["gpus_per_actor"] = args.gpus_per_actor
    if args.elastic_training is not None:
        ray_params["elastic_training"] = args.elastic_training
    if args.max_failed_actors:
        ray_params["max_failed_actors"] = args.max_failed_actors
    if args.max_actor_restarts:
        ray_params["max_actor_restarts"] = args.max_actor_restarts
    if args.checkpoint_frequency:
        ray_params["checkpoint_frequency"] = args.checkpoint_frequency
    return ray_params


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


def _train(args, data_engine, train_data, test_data):
    if not args.in_memory and not args.processed_data_file:
        raise ValueError(
            "Must specify the processed-data-file which contains processed data to be trained.")
    _check_temp_dir(args)

    if not args.training_config:
        # default to the built-in training_config.yaml if not specified
        args.training_config = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config/training-config.yaml")
        print("training-config is not specified. Default to: {}".format(
            args.training_config))

    if args.in_memory:
        if data_engine == DATA_ENGINE_PANDAS:
            import pandas as pd
        else:
            import modin.pandas as pd
        data = pd.concat([train_data, test_data])
    else:
        print(f"loading data from: {args.processed_data_file}")
        data = read_csv_files(
            args.processed_data_file, engine=DATA_ENGINE_PANDAS)

    _train_on_data(
        data,
        training_config=args.training_config,
        temp_dir=args.temp_dir,
        model_file=args.model_file,
        in_memory=args.in_memory
    )


def run(args):
    data_engine = DATA_ENGINE_PANDAS
    if not args.single_node:
        data_engine = DATA_ENGINE_MODIN
    train_data, test_data = (None, None)
    if not args.no_process_data or args.in_memory:
        train_data, test_data = _process_data(
            args, data_engine)

    if not args.no_train:
        _train(
            args, data_engine, train_data, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument(
        "--single-node", "--single_node",
        default=False, action="store_true",
        help="To do single node training")
    parser.add_argument(
        "--no-process-data", "--no_process_data",
        default=False, action="store_true",
        help="whether to do data process")
    parser.add_argument(
        "--no-train", "--no_train",
        default=False, action="store_true",
        help="whether to do training")

    parser.add_argument(
        "--in-memory", "--in_memory",
        default=False, action="store_true",
        help="whether do in memory data processing and training without save processed data to file.")

    parser.add_argument(
        "--raw-data-path", "--raw_data_path",
        type=existing_path,
        help="The path contains the raw data files or the file.")
    parser.add_argument(
        "--data-processing-config", "--data_processing_config",
        type=existing_file, help="The path to the data processing config file")
    parser.add_argument(
        "--training-config", "--training_config",
        type=existing_file,
        help="The path to the training config file")
    parser.add_argument(
        "--no-save", "--no_save",
        default=False, action="store_true",
        help="whether to save the processed data file")
    parser.add_argument(
        "--processed-data-file", "--processed_data_file",
        type=str,
        help="The path to the output processed data file")
    parser.add_argument(
        "--temp-dir", "--temp_dir",
        type=str,
        help="The path to the shared intermediate data")
    parser.add_argument(
        "--model-file", "--model_file",
        type=str,
        help="The path to the output model file")

    # Ray params
    #     num_actors: 5
    #     cpus_per_actor: 15
    #     elastic_training: True
    #     max_failed_actors: 4
    #     max_actor_restarts: 8
    parser.add_argument(
        "--num-actors", "--num_actors",
        type=int, default=5,
        help="The number of actors")
    parser.add_argument(
        "--cpus-per-actor", "--cpus_per_actor",
        type=int, default=15,
        help="The number of cpus per actor")
    parser.add_argument(
        "--gpus-per-actor", "--gpus_per_actor",
        type=int, default=-1,
        help="The number of gpus per actor")
    parser.add_argument(
        "--elastic-training", "--elastic_training",
        action="store_true", default=False,
        help="whether to use elastic training")
    parser.add_argument(
        "--max-failed-actors", "--max_failed_actors",
        type=int, default=4,
        help="The max number of failed actors")
    parser.add_argument(
        "--max-actor-restarts", "--max_actor_restarts",
        type=int, default=8,
        help="The max number of actor restarts")
    parser.add_argument(
        "--checkpoint-frequency", "--checkpoint_frequency",
        type=int, default=5,
        help="The checkpoint frequency")

    args = parser.parse_args()
    print(args)

    run(args)
