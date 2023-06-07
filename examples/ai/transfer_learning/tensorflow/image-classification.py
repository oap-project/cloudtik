import argparse
import os
import shutil
import tempfile

import numpy as np

from cloudtik.runtime.ai.modeling.transfer_learning import dataset_factory, model_factory

FRAMEWORK_TENSORFLOW = 'tensorflow'

# Training settings
parser = argparse.ArgumentParser(description='Image Classification Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')
parser.add_argument('--model-name', default="efficientnet_b0",
                    help='the model name to use.')
parser.add_argument('--dataset-name', default="tf_flowers",
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

    category = 'image_classification'
    framework = FRAMEWORK_TENSORFLOW
    dataset_name = args.dataset_name
    model_name = args.model_name

    print("Model name:", model_name)
    print("Dataset name:", dataset_name)

    try:
        # Get the model
        model = model_factory.get_model(
            model_name, category=category, framework=framework)

        # Get the dataset
        dataset = dataset_factory.get_dataset(
            output_dir, category, framework, dataset_name,
            source='tfds', split=["train[:5%]"], shuffle_files=False, seed=10)

        # Preprocess the dataset
        dataset.preprocess(model.image_size, 32)
        dataset.shuffle_split(shuffle_files=False)

        if args.train_local:
            print("Training locally...")
            # Evaluate before training
            pretrained_metrics = model.evaluate(dataset)
            assert len(pretrained_metrics) > 0

            # Train
            history = model.train(
                dataset, output_dir=output_dir, epochs=1,
                shuffle_files=False, seed=10, do_eval=False)
            assert history is not None

            # Verify that checkpoints were generated
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(model_name))
            assert os.path.isdir(checkpoint_dir)
            assert len(os.listdir(checkpoint_dir))

            # Evaluate
            trained_metrics = model.evaluate(dataset)
            assert trained_metrics[0] <= pretrained_metrics[0]  # loss
            assert trained_metrics[1] >= pretrained_metrics[1]  # accuracy

            # Predict with a batch
            images, labels = dataset.get_batch()
            predictions = model.predict(images)
            assert len(predictions) == 32
            probabilities = model.predict(images, return_type='probabilities')
            assert probabilities.shape == (32, 5)  # tf_flowers has 5 classes
            np.testing.assert_almost_equal(np.sum(probabilities), np.float32(32), decimal=4)

            # Export the saved model
            saved_model_dir = model.export(output_dir)
            assert os.path.isdir(saved_model_dir)
            assert os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb"))

            # Reload the saved model
            reload_model = model_factory.get_model(model_name, framework)
            reload_model.load_from_directory(saved_model_dir)

            # Evaluate
            reload_metrics = reload_model.evaluate(dataset)
            np.testing.assert_almost_equal(reload_metrics, trained_metrics)

            # Retrain from checkpoints and verify that we have better accuracy than the original training
            retrain_model = model_factory.load_model(
                model_name, saved_model_dir, category=category, framework=framework)
            retrain_history = retrain_model.train(dataset, output_dir=output_dir, epochs=1,
                                                  initial_checkpoints=checkpoint_dir,
                                                  shuffle_files=False, seed=10, do_eval=False)
            print(retrain_history['acc'])
        else:
            print("Training distributed...")
            nnodes = len(worker_ips)
            nproc_per_node = worker_num_proc
            shared_dir = args.shared_dir
            model.train(dataset, output_dir=output_dir, epochs=1,
                        shuffle_files=False, seed=10, do_eval=False,
                        distributed=True, nnodes=nnodes, nproc_per_node=nproc_per_node,
                        hosts=hosts, shared_dir=shared_dir)
    finally:
        # Delete the temp output directory
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
