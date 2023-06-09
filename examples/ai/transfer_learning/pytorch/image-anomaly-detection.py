import argparse
import os
import shutil
import tempfile

from cloudtik.runtime.ai.modeling.transfer_learning import dataset_factory, model_factory
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import download_and_extract_tar_file
from cloudtik.runtime.ai.modeling.transfer_learning.image_anomaly_detection.pytorch.image_anomaly_detection_model import \
    extract_features

FRAMEWORK_PYTORCH = 'pytorch'

# Training settings
parser = argparse.ArgumentParser(description='Image Anomaly Detection Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training')
parser.add_argument('--model-name', default="resnet18",
                    help='the model name to use.')
parser.add_argument('--dataset-name', default="custom",
                    help='the dataset name to use.')
parser.add_argument('--feature-extractor', default="default",
                    help='the feature-extractor to use: default, simsiam, cutpaste.')
parser.add_argument('--eval-use-test-set', action='store_true', default=False,
                    help='Whether evaluate using test set.')


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

    category = 'image_anomaly_detection'
    framework = FRAMEWORK_PYTORCH
    dataset_name = args.dataset_name
    model_name = args.model_name

    print("Model name:", model_name)
    print("Dataset name:", dataset_name)

    # Get the model
    model = model_factory.get_model(
        model_name, category=category, framework=framework)

    # prepare the custom dataset
    temp_dir = tempfile.mkdtemp()
    custom_dataset_path = os.path.join(temp_dir, "flower_photos")
    if not os.path.exists(custom_dataset_path):
        download_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        download_and_extract_tar_file(download_url, temp_dir)
        # Rename daisy to "good" and delete all but one other kind to make the dataset small
        os.rename(os.path.join(custom_dataset_path, 'daisy'), os.path.join(custom_dataset_path, 'good'))
        for flower in ['dandelion', 'roses', 'sunflowers']:
            shutil.rmtree(os.path.join(custom_dataset_path, flower))

    dataset_dir = custom_dataset_path

    # Get the dataset
    dataset = dataset_factory.load_dataset(
        dataset_dir, category, framework, dataset_name,
        shuffle_files=False)
    assert ['tulips'] == dataset.defect_names
    assert ['bad', 'good'] == dataset.class_names

    # Preprocess the dataset and split to get small subsets for training and validation
    dataset.preprocess(model.image_size, 32)

    if args.eval_use_test_set:
        dataset.shuffle_split(
            train_pct=0.5, val_pct=0.25, test_pct=0.25, seed=10)
    else:
        dataset.shuffle_split(
            train_pct=0.5, val_pct=0.5, seed=10)

    print("Training locally...")
    if args.feature_extractor == "simsiam":
        # Train for 1 epoch
        pca_components, trained_model = model.train(
            dataset, output_dir,
            epochs=1, seed=10,
            layer_name='layer3', feature_dim=1000, pred_dim=250,
            simsiam=True)

        # Evaluate
        threshold, auroc = model.evaluate(
            dataset, pca_components,
            use_test_set=args.eval_use_test_set)
        assert isinstance(auroc, float)

        # Predict with a batch
        images, labels = dataset.get_batch(subset='validation')
        predictions = model.predict(images, pca_components)
        assert len(predictions) == 32
    elif args.feature_extractor == "cutpaste":
        # Train for 1 epoch
        pca_components, trained_model = model.train(
            dataset, output_dir,
            epochs=1, seed=10,
            layer_name='layer3', optim='sgd', freeze_resnet=20,
            head_layer=2, cutpaste_type='normal',
            cutpaste=True)

        # Evaluate
        threshold, auroc = model.evaluate(
            dataset, pca_components,
            use_test_set=args.eval_use_test_set)
        assert isinstance(auroc, float)

        # Predict with a batch
        images, labels = dataset.get_batch(subset='test')
        predictions = model.predict(images, pca_components)
        assert len(predictions) == 32
    else:
        # default
        # Train for 1 epoch
        pca_components, trained_model = model.train(
            dataset, output_dir,
            layer_name='layer3', seed=10)

        # Extract features
        images, labels = dataset.get_batch(subset='validation')
        features = extract_features(
            trained_model, images, layer_name='layer3', pooling=['avg', 2])
        assert len(features) == 32

        # Evaluate
        threshold, auroc = model.evaluate(
            dataset, pca_components,
            use_test_set=args.eval_use_test_set)
        assert isinstance(auroc, float)

        # Predict with a batch
        predictions = model.predict(images, pca_components)
        assert len(predictions) == 32

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
