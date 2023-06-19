import os
import shutil
import time
import yaml

import numpy as np
import tensorflow as tf
from PIL import Image

from cloudtik.runtime.ai.modeling.transfer_learning import dataset_factory, model_factory
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import FrameworkType
from cloudtik.runtime.ai.util.utils import move_dir_contents
from disease_prediction.utils import id_to_label_mapping, is_labeled_dataset, read_yaml_file

IMAGE_SIZE = 224

TRAIN_FRAMEWORK = str(FrameworkType.TENSORFLOW)
TRAIN_CATEGORY = "image_classification"


def collect_class_labels(dataset_dir):
    dataset = dataset_factory.load_dataset(dataset_dir=dataset_dir,
                                           category=TRAIN_CATEGORY,
                                           framework=TRAIN_FRAMEWORK)
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
        framework=TRAIN_FRAMEWORK)
    tend = time.time()
    print("\nModel Loading time (s): ", tend - tstart)

    # Load the dataset from the custom dataset path
    # Data loading and preprocessing #
    dataset = dataset_factory.load_dataset(
        dataset_dir=dataset_dir,
        category=TRAIN_CATEGORY,
        framework=TRAIN_FRAMEWORK,
        shuffle_files=True)
    class_names = dataset.class_names

    print("Class names:", str(class_names))
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
        # save also the classnames as meta information
        _save_model_meta(class_names, saved_model_dir)

    print("Done fine tuning the model ............")
    return model, class_names, history, dict_metrics, saved_model_dir


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
            move_dir_contents(
                source=saved_model_dir, target=model_dir, overwrite=True)
            saved_model_dir = model_dir
    return saved_model_dir


def _save_model_meta(class_names, model_dir):
    meta = {"class_names": class_names}
    meta_file = os.path.join(model_dir, "model-meta.yaml")
    with open(meta_file, 'w') as file:
        yaml.dump(meta, file)


def load_model_meta(model_dir):
    meta_file = os.path.join(model_dir, "model-meta.yaml")
    return read_yaml_file(meta_file)


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
    predict_model = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # result = model.predict(image[np.newaxis, ...])
    # result=model.predict(image[np.newaxis, ...], 'probabilities')[0]
    output_name = list(predict_model.structured_outputs.keys())
    result = predict_model(tf.constant(image))[output_name[0]][0]
    return result


def load_and_preprocess_dataset(dataset_dir, image_size, batch_size):
    """
    Load and preprocess dataset
    """
    dataset = dataset_factory.load_dataset(
        dataset_dir=dataset_dir,
        category=TRAIN_CATEGORY,
        framework=TRAIN_FRAMEWORK,
        shuffle_files=False)
    dataset.preprocess(image_size, batch_size)
    return dataset


def predict_image_dir(
        model, image_dir, id_to_label,
        predictions_report,
        label=None,
        int8=False):
    fns = os.listdir(image_dir)
    for fn in fns:
        patient_id = fn
        fn = os.path.join(os.path.join(image_dir, fn))
        if int8:
            result = _predict_int8(model, fn)
        else:
            result = _predict(model, fn)
        pred_prob = result.numpy().tolist()

        result_dict = {
            "pred": id_to_label[np.argmax(pred_prob).tolist()],
            "pred_prob": pred_prob
        }
        if label:
            result_dict["label"] = label
        predictions_report["results"][patient_id] = [result_dict]


def predict(
        test_data_dir, saved_model_dir, class_names,
        model_name="resnet_v1_50", int8=False,
        output_file="output.yaml"):
    # Load the model
    tstart = time.time()
    model_dir = saved_model_dir

    id_to_label = id_to_label_mapping(class_names)
    is_labeled = is_labeled_dataset(test_data_dir)

    predictions_report_save_file = output_file
    predictions_report = {"metric": {}, "results": {}}

    # Load model
    model = model_factory.load_model(
        model_name, model_dir,
        category=TRAIN_CATEGORY,
        framework=TRAIN_FRAMEWORK)

    if int8:
        model_int8 = tf.saved_model.load(model_dir)
        predict_model = model_int8
    else:
        predict_model = model

    tend = time.time()
    print("\n Vision Model Loading time: ", tend - tstart)

    # This should only be done on a dataset that has labels for testing
    if is_labeled:
        # Load dataset for metric evaluation
        dataset = load_and_preprocess_dataset(
            test_data_dir, model.image_size, 32)
        metrics = model.evaluate(dataset)
        for metric_name, metric_value in zip(model._model.metrics_names,
                                             metrics):
            print("{}: {}".format(metric_name, metric_value))
            predictions_report["metric"][metric_name] = metric_value

    tstart = time.time()
    if is_labeled:
        for label in os.listdir(test_data_dir):
            print("Predicting data in labeled folder: ", label)
            image_dir = os.path.join(test_data_dir, label)
            predict_image_dir(
                predict_model, image_dir, id_to_label,
                predictions_report,
                label=label, int8=int8
            )
    else:
        # It's a prediction dataset without label
        predict_image_dir(
            predict_model, test_data_dir, id_to_label,
            predictions_report,
            label=None, int8=int8
        )

    predictions_report["label"] = class_names
    predictions_report["label_id"] = list(id_to_label.keys())

    with open(predictions_report_save_file, 'w') as file:
        _ = yaml.dump(predictions_report, file, )
    print("Prediction time: ", time.time() - tstart)
