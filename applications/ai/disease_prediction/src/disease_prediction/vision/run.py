import argparse
import os

from disease_prediction.vision.trainer import train, predict, load_model_meta

from disease_prediction.utils import DEFAULT_TRAIN_OUTPUT, DEFAULT_PREDICT_OUTPUT


class TrainerArguments:
    def __init__(self):
        self.no_train = False
        self.no_predict = False
        self.data_path = None
        self.model = "resnet_v1_50"
        self.temp_dir = None
        self.output_dir = None
        self.model_dir = None
        self.train_output = None
        self.predict_output = None

        self.batch_size = 32
        self.epochs = 5
        self.int8 = False
        self.disable_auto_mixed_precision = False

        self.hosts = None


def run(args):
    dataset_dir = args.data_path
    if not dataset_dir:
        raise ValueError(
            "Must specify the data path which contains the train and test images.")

    train_dataset_dir = os.path.join(dataset_dir, "train")
    test_dataset_dir = os.path.join(dataset_dir, "test")
    enable_auto_mixed_precision = not args.disable_auto_mixed_precision

    if not args.no_train:
        if not args.output_dir:
            raise ValueError(
                "Must specify the output dir for storing the output model and result.")

        print("Start Vision training...")
        model, class_names, history, dict_metrics, saved_model_dir = train(
            dataset_dir=train_dataset_dir,
            output_dir=args.output_dir,
            model=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_dir=args.model_dir,
            enable_auto_mixed_precision=enable_auto_mixed_precision)

        if not args.model_dir:
            args.model_dir = saved_model_dir

        if not args.train_output:
            args.train_output = os.path.join(
                args.output_dir, DEFAULT_TRAIN_OUTPUT)
            print("train-output is not specified. Default to: {}".format(
                args.output_dir))

        predict(
            train_dataset_dir, args.model_dir,
            class_names=class_names,
            model_name=args.model,
            int8=args.int8,
            output_file=args.train_output)
        print("End Vision training.")

    if not args.no_predict:
        if not args.predict_output and args.output_dir:
            args.predict_output = os.path.join(
                args.output_dir, DEFAULT_PREDICT_OUTPUT)
            print("predict-output is not specified. Default to: {}".format(
                args.output_dir))

        if not args.model_dir:
            raise ValueError(
                "Must specify the model dir stored the output model.")
        if not args.predict_output:
            raise ValueError(
                "Must specify the predict output for storing test results.")

        # The class names (same order as training) stored in the model meta
        model_meta = load_model_meta(args.model_dir)

        print("Start Vision predicting...")
        predict(
            test_dataset_dir, args.model_dir,
            class_names=model_meta["class_names"],
            model_name=args.model,
            int8=args.int8,
            output_file=args.predict_output)
        print("End Vision predicting...")


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
        help="The path to the model")
    parser.add_argument(
        "--train-output", "--train_output",
        type=str,
        help="The path to the train output")
    parser.add_argument(
        "--predict-output", "--predict_output",
        type=str,
        help="The path to the predict output")

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
