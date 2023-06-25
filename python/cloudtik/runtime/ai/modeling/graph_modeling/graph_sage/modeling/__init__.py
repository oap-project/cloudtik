from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.run \
    import (run as _run, get_data_with_embeddings_path)


class ModelingArgs(object):
    def __init__(self):
        self.single_node = True
        self.no_process_data = False
        self.no_build_graph = False
        self.no_partition_graph = False
        self.no_train = False
        self.no_predict = False

        self.raw_data_path = None
        self.processed_data_path = None

        self.temp_dir = None
        self.output_dir = None
        self.model_file = None
        self.dataset_name = "tabformer"
        self.tabular2graph = None
        self.train_output = None
        self.predict_output = None

        self.data_with_embeddings_name = "data_with_embeddings.csv"

        self.hosts = None
        self.graph_name = "tabformer_graph"
        self.num_parts = 0
        self.num_hops = 1
        self.num_trainers = 1
        self.num_samplers = 2
        self.num_servers = 1

        self.num_server_threads = 1
        self.num_omp_threads = None

        # These defaults are set for distributed training
        self.num_epochs = 10
        self.num_hidden = 64
        self.num_layers = 2
        self.fan_out = "55,65"
        self.batch_size = 2048
        self.batch_size_eval = 1000000
        self.eval_every = 1
        self.lr = 0.0005

        self.log_every = 20
        self.num_dl_workers = 4

        self.heterogeneous = False
        self.relations = None
        self.inductive = False
        self.node_feature = None


def run(args: ModelingArgs):
    return _run(args)

