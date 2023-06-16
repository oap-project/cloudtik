import argparse
import os
import shutil
import time
import yaml

import numpy as np
import tensorflow as tf
from PIL import Image

from cloudtik.runtime.ai.modeling.transfer_learning import dataset_factory, model_factory
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import FrameworkType

IMAGE_SIZE = 224


class TrainerArguments:
    def __init__(self):
        self.no_train = False
        self.no_predict = False
        self.data_path = None
        self.model = "resnet_v1_50"
        self.temp_dir = None
        self.output_dir = None
        self.model_dir = None
        self.output_report_file = None

        self.batch_size = 32
        self.epochs = 5
        self.int8 = False
        self.disable_auto_mixed_precision = False

        self.hosts = None


def collect_class_labels(dataset_dir):
    dataset = dataset_factory.load_dataset(dataset_dir=dataset_dir,
                                           use_case='image_classification',
                                           framework='tensorflow')
    return dataset.class_names


def clean_output_folder(output_dir, model_name):
    folder_path = os.path.join(output_dir, model_name)
    if os.path.exists(folder_path):
        shutil.rmtree(os.path.join(output_dir, model_name))


def quantize_model(model, saved_model_dir, quantization_output_dir):
    # TODO: the path of neural compressor config file
    root_folder = os.path.dirname(os.path.abspath(__file__))
    neural_compressor_config = os.path.join(root_folder,
                                            "neural-compressor-config.yaml")

    model.quantize(
        saved_model_dir, quantization_output_dir, neural_compressor_config)


def train(
        dataset_dir, output_dir,
        model="resnet_v1_50", batch_size=32, epochs=5,
        save_model=True, model_dir=None,
        quantization=False, enable_auto_mixed_precision=True):
    # Clean the output folder first
    clean_output_folder(output_dir, model)

    dict_metrics = {}

    #  Loading the model
    tstart = time.time()
    model = model_factory.get_model(
        model_name=model,
        framework=str(FrameworkType.TENSORFLOW))
    tend = time.time()
    print("\nModel Loading time (s): ", tend - tstart)

    # Load the dataset from the custom dataset path
    # Data loading and preprocessing #
    dataset = dataset_factory.load_dataset(
        dataset_dir=dataset_dir,
        category='image_classification',
        framework=str(FrameworkType.TENSORFLOW),
        shuffle_files=True)

    print("Class names:", str(dataset.class_names))
    dataset.preprocess(
        model.image_size, batch_size=batch_size,
        add_aug=['hvflip', 'rotate'])
    dataset.shuffle_split(train_pct=.80, val_pct=.20)

    # Fine tuning
    tstart = time.time()
    history = model.train(
        dataset, output_dir=output_dir, epochs=epochs,
        seed=10,
        enable_auto_mixed_precision=enable_auto_mixed_precision,
        extra_layers=[1024, 512])
    tend = time.time()
    print("\nTotal fine tuning time (s): ", tend - tstart)
    dict_metrics['training_time'] = tend - tstart

    metrics = model.evaluate(dataset)
    for metric_name, metric_value in zip(
            model._model.metrics_names, metrics):
        print("{}: {}".format(metric_name, metric_value))
        dict_metrics[metric_name] = metric_value
    print('dict_metrics:', dict_metrics)
    print('Finished fine tuning the model...')

    saved_model_dir = None
    if save_model:
        saved_model_dir = _save_model(
            model, model_dir, output_dir, quantization)

    print("Done fine tuning the model ............")
    return model, history, dict_metrics, saved_model_dir


def _save_model(model, model_dir, output_dir, quantization):
    saved_model_dir = model.export(output_dir)
    if quantization:
        print('Doing quantization on the model')
        if model_dir:
            quantization_output_dir = model_dir
        else:
            clean_output_folder(output_dir, 'quantized_models')
            quantization_output_dir = os.path.join(
                output_dir, 'quantized_models',
                os.path.basename(saved_model_dir))
        quantize_model(model, saved_model_dir, quantization_output_dir)
        return quantization_output_dir
    else:
        if model_dir:
            # move the contents of saved_model_dir to model_dir
            shutil.move(saved_model_dir, model_dir)
            saved_model_dir = model_dir
    return saved_model_dir


def _predict(model, image_location):
    image_shape = (model.image_size, model.image_size)
    image = Image.open(image_location).resize(image_shape)
    # Get the image as a np array and call predict while adding a batch
    # dimension (with np.newaxis)
    image = np.array(image)/255.0
    result = model.predict(image[np.newaxis, ...], 'probabilities')[0]
    return result


def _predict_int8(model, image_location):
    image_shape = (IMAGE_SIZE, IMAGE_SIZE)
    image = Image.open(image_location).resize(image_shape)
    # Get the image as a np array and call predict while
    # adding a batch dimension (with np.newaxis)
    image = np.array(image)/255.0
    image = image[np.newaxis, ...].astype('float32')
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # result = model.predict(image[np.newaxis, ...])
    # result=model.predict(image[np.newaxis, ...], 'probabilities')[0]
    output_name = list(infer.structured_outputs.keys())
    result = infer(tf.constant(image))[output_name[0]][0]
    return result


def preprocess_dataset(dataset_dir, image_size, batch_size):
    """
    Load and preprocess dataset
    """
    dataset = dataset_factory.load_dataset(
        dataset_dir=dataset_dir,
        use_case='image_classification',
        framework=str(FrameworkType.TENSORFLOW),
        shuffle_files=False)
    dataset.preprocess(image_size, batch_size)
    class_dict = reverse_map(dataset.class_names)
    return dataset, class_dict


def reverse_map(class_names):
    class_dict = {}
    i = 0
    for c in class_names:
        class_dict[i] = c
        i = i + 1
    return class_dict


def predict(
        test_data_dir, saved_model_dir, class_labels,
        model_name="resnet_v1_50", int8=False,
        output_file="output.yaml"):
    # Load the model
    tstart = time.time()
    model_dir = saved_model_dir
    test_dir = test_data_dir
    labels = class_labels
    predictions_report_save_file = output_file
    predictions_report = {"metric": {}, "results": {}}

    # Load model
    model = model_factory.load_model(
        model_name, model_dir,
        category="image_classification",
        framework=str(FrameworkType.TENSORFLOW))

    if int8:
        model_int8 = tf.saved_model.load(model_dir)

    tend = time.time()
    print("\n Vision Model Loading time: ", tend - tstart)
    # Load dataset for metric evaluation
    dataset, class_dict = preprocess_dataset(test_data_dir,
                                             model.image_size, 32)
    metrics = model.evaluate(dataset)
    for metric_name, metric_value in zip(model._model.metrics_names,
                                         metrics):
        print("{}: {}".format(metric_name, metric_value))
        predictions_report["metric"][metric_name] = metric_value

    tstart = time.time()
    for label in os.listdir(test_dir):
        print("Infering data in folder: ", label)
        fns = os.listdir(os.path.join(test_dir, label))
        for fn in fns:
            patient_id = fn
            fn = os.path.join(os.path.join(test_dir, label, fn))
            if int8:
                result = _predict_int8(model_int8, fn)
            else:
                result = _predict(model, fn)
            pred_prob = result.numpy().tolist()
            infer_result_patient = [
                {
                    "label": label,
                    "pred": class_dict[np.argmax(pred_prob).tolist()],
                    "pred_prob": pred_prob
                }
            ]
            predictions_report["label"] = labels
            predictions_report["label_id"] = list(class_dict.keys())
            predictions_report["results"][patient_id] = infer_result_patient

    with open(predictions_report_save_file, 'w') as file:
        _ = yaml.dump(predictions_report, file, )
    print("Prediction time: ", time.time() - tstart)


def run(args):
    dataset_dir = args.data_path
    if not dataset_dir:
        raise ValueError("Must specify the data path which contains the train and test images.")

    train_dataset_dir = os.path.join(dataset_dir, "train")
    test_dataset_dir = os.path.join(dataset_dir, "test")
    enable_auto_mixed_precision = not args.disable_auto_mixed_precision

    if not args.no_train:
        if not args.output_dir:
            raise ValueError("Must specify the output dir for storing the output model and result.")

        model, history, dict_metrics, saved_model_dir = train(
            train_dataset_dir,
            args.output_dir, args.model,
            args.batch_size, args.epochs,
            args.model_dir, enable_auto_mixed_precision=enable_auto_mixed_precision)
        class_labels = collect_class_labels(train_dataset_dir)

        if not args.model_dir:
            args.model_dir = saved_model_dir
        train_output = os.path.join(args.output_dir, "train_output.yaml")
        predict(
            train_dataset_dir, args.model_dir, class_labels,
            args.model, args.int8, train_output)

        if not args.output_report_file:
            args.output_report_file = os.path.join(args.output_dir, "output.yaml")

    if not args.no_predict:
        if not args.model_dir:
            raise ValueError("Must specify the model dir stored the output model.")
        if not args.output_report_file:
            raise ValueError("Must specify the output report file for storing test results.")
        class_labels = collect_class_labels(train_dataset_dir)
        predict(
            test_dataset_dir, args.model_dir, class_labels,
            args.model, args.int8, args.output_report_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Vision Model Training")

    parser.add_argument(
        "--no-train", "--no_train",
        default=False, action="store_true",
        help="whether to do train (fine tuning)")
    parser.add_argument(
        "--no-predict", "--no_predict",
        default=False, action="store_true",
        help="whether to do prediction on test data")

    parser.add_argument(
        "--data-path", "--data_path",
        type=str,
        help="The path to the input data")
    parser.add_argument(
        "--model",
        type=str, default="resnet_v1_50",
        help="The model name")

    parser.add_argument(
        "--temp-dir", "--temp_dir",
        type=str,
        help="The path to the intermediate data")
    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        help="The path to the output")

    parser.add_argument(
        "--model-dir", "--model_dir",
        type=str,
        help="The path to the output model file")
    parser.add_argument(
        "--output-report-file", "--output_report_file",
        type=str,
        help="The path to the output report file")

    parser.add_argument(
        "--batch-size", "--batch_size",
        type=int, default=32)
    parser.add_argument(
        "--epochs",
        type=int, default=5)
    parser.add_argument(
        "--int8",
        default=False, action="store_true")
    parser.add_argument(
        "--disable-auto-mixed-precision", "--disable_auto_mixed_precision",
        default=False, action="store_true")

    parser.add_argument(
        "--hosts",
        type=str,
        help="List of hosts separated with comma for launching tasks. ")

    args = parser.parse_args()
    print(args)

    run(args)
