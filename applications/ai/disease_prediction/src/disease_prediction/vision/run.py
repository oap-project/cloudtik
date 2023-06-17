import argparse
import os

from disease_prediction.vision.trainer import train, collect_class_labels, predict


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
