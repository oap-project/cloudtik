import argparse
import os
import shutil
import tempfile

from cloudtik.runtime.ai.modeling.transfer_learning import dataset_factory, model_factory


FRAMEWORK_PYTORCH = 'pytorch'

# Training settings
parser = argparse.ArgumentParser(description='Text Classification Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')
parser.add_argument('--model-name', default="bert-base-cased",
                    help='the model name to use.')
parser.add_argument('--dataset-name', default="imdb",
                    help='the dataset name to use.')
parser.add_argument('--shared-dir', default="/cloudtik/fs",
                    help='the shared file system dir.')
parser.add_argument('--train-local', action='store_true', default=False,
                    help='Do the train locally on head.')

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
        total_workers = cluster_info.get("total-workers-ready")
        if total_workers:
            args.num_proc = total_workers
        if not args.num_proc:
            args.num_proc = 1

    worker_ips = cluster.get_worker_node_ips(node_status="up-to-date")

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

    # Get the dataset
    output_dir = tempfile.mkdtemp()

    category = 'text_classification'
    framework = FRAMEWORK_PYTORCH
    dataset_name = args.dataset_name
    model_name = args.model_name

    print("Model name:", model_name)
    print("Dataset name:", dataset_name)

    if args.train_local:
        dataset_dir = output_dir
    else:
        if not args.shared_dir:
            raise ValueError("Must specify shared dir for distributed training.")
        dataset_dir = args.shared_dir

    # Get the model
    model = model_factory.get_model(
        model_name, category=category, framework=framework)

    # Get the dataset
    dataset = dataset_factory.get_dataset(
        dataset_dir, category, framework, dataset_name,
        source='hugging_face', split=["train"], shuffle_files=False)

    # Preprocess the dataset
    dataset.preprocess(model_name, batch_size=32)
    dataset.shuffle_split(train_pct=0.01, val_pct=0.01, seed=10)
    assert dataset._validation_type == 'shuffle_split'

    if args.train_local:
        # Evaluate before training
        pretrained_metrics = model.evaluate(dataset)
        assert len(pretrained_metrics) > 0

        print("Training locally...")
        model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False)
        classifier_layer = getattr(model._model, "classifier")
        try:
            # If extra_layers given, the classifier is a Sequential layer with given input
            n_layers = len(classifier_layer)
        except TypeError:
            # If not given, the classifier is just a single Linear layer
            n_layers = 1
        print("Number of classifier layers:", n_layers)

        # Evaluate
        trained_metrics = model.evaluate(dataset)
        assert trained_metrics[0] <= pretrained_metrics[0]  # loss
        assert trained_metrics[1] >= pretrained_metrics[1]  # accuracy

        # Export the saved model
        saved_model_dir = model.export(output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "model.pt"))

        # Reload the saved model
        reload_model = model_factory.get_model(model_name, framework)
        reload_model.load_from_directory(saved_model_dir, num_classes=len(dataset.class_names))

        # Evaluate
        reload_metrics = reload_model.evaluate(dataset)
        assert reload_metrics == trained_metrics
    else:
        print("Training distributed...")
        nnodes = len(worker_ips)
        nproc_per_node = worker_num_proc
        shared_dir = args.shared_dir
        model.train(dataset, output_dir=output_dir, epochs=1, do_eval=False,
                    distributed=True, nnodes=nnodes, nproc_per_node=nproc_per_node,
                    hosts=hosts, shared_dir=shared_dir)

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
