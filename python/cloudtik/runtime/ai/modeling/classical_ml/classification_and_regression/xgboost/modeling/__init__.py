from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.run \
    import run as _run


class ModelingArgs(object):
    def __init__(self):
        self.single_node = False
        self.no_process_data = False
        self.no_training = False

        self.in_memory = False
        self.raw_data_path = None
        self.data_processing_config = None
        self.no_save = False
        self.processed_data_file = None

        self.training_config = None
        self.temp_dir = None
        self.model_file = None

        # ray params
        self.num_actors = 5
        self.cpus_per_actor = 15
        self.gpus_per_actor = -1
        self.elastic_training = False
        self.max_failed_actors = 4
        self.max_actor_restarts = 8
        self.checkpoint_frequency = 5


def run(args: ModelingArgs):
    return _run(args)
