import argparse

from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.process_data import \
    process_data
from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.utils import \
    existing_file, existing_directory, load_config, read_csv_files, DATA_ENGINE_PANDAS, DATA_ENGINE_MODIN


def train_on_ray(
        df, training_config, temp_dir, model_file,
        in_memory, on_ray=False, ray_params=None):
    print('start training models on ray...')
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

    from .train_ray import train_ray
    train_ray(
        train_data_spec, df,
        train_model_spec, in_memory,
        tmp_path=temp_dir, model_file=model_file,
        on_ray=on_ray, ray_params=ray_params, hpo_spec=hpo_spec)


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


def run(args):
    data_engine = DATA_ENGINE_PANDAS
    if not args.is_single_node:
        data_engine = DATA_ENGINE_MODIN

    if not args.no_process_data or args.in_memory:
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

    if not args.no_training:
        if args.in_memory:
            if data_engine == DATA_ENGINE_PANDAS:
                import pandas as pd
            else:
                import modin.pandas as pd
            data = pd.concat([train_data, test_data])
        else:
            data = read_csv_files(
                args.processed_data_file, engine=DATA_ENGINE_PANDAS)

        on_ray = False
        if not args.single_node:
            on_ray = True

        ray_params = get_ray_params(args)
        train_on_ray(
            data,
            training_config=args.training_config,
            temp_dir=args.temp_dir,
            model_file=args.model_file,
            in_memory=args.in_memory,
            on_ray=on_ray,
            ray_params=ray_params,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument(
        "--single_node", default=False, action="store_true",
        help="To do single node training")
    parser.add_argument(
        "--no_process_data", default=False, action="store_true",
        help="whether to do data process")
    parser.add_argument(
        "--no_training", default=False, action="store_true",
        help="whether to do training")

    parser.add_argument(
        "--in_memory", default=False, action="store_true",
        help="whether do in memory data processing and training without save processed data to file.")

    parser.add_argument(
        "--raw_data_path", type=existing_directory, help="The path contains the raw data files")
    parser.add_argument(
        "--data_processing_config", type=existing_file, help="The path to the data processing config file")
    parser.add_argument(
        "--training_config", type=existing_file, help="The path to the training config file")
    parser.add_argument(
        "--no_save", default=False, action="store_true",
        help="whether to save the processed data file")
    parser.add_argument(
        "--processed_data_file",
        type=str,
        help="The path to the output processed data file")
    parser.add_argument(
        "--model_file",
        type=str,
        help="The path to the output model file")

    # Ray params
    #     num_actors: 5
    #     cpus_per_actor: 15
    #     elastic_training: True
    #     max_failed_actors: 4
    #     max_actor_restarts: 8
    parser.add_argument(
        "--num_actors", type=int, default=5,
        help="The number of actors")
    parser.add_argument(
        "--cpus_per_actor", type=int, default=15,
        help="The number of cpus per actor")
    parser.add_argument(
        "--gpus_per_actor", type=int,
        help="The number of gpus per actor")
    parser.add_argument(
        "--elastic_training", action="store_true",
        help="whether to use elastic training")
    parser.add_argument(
        "--max_failed_actors", type=int, default=4,
        help="The max number of failed actors")
    parser.add_argument(
        "--max_actor_restarts", type=int, default=8,
        help="The max number of actor restarts")
    parser.add_argument(
        "--checkpoint_frequency", type=int,
        help="The checkpoint frequency")

    args = parser.parse_args()
    print(args)

    run(args)
