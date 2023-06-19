import argparse
import os

from disease_prediction.consult.trainer import Trainer as ConsultTrainer


class TrainerArguments:
    def __init__(self):
        self.no_train = False
        self.no_predict = False

        self.dlsa_train_input = None
        self.dlsa_train_output = None
        self.vision_train_output = None

        self.dlsa_predict_input = None
        self.dlsa_predict_output = None
        self.vision_predict_output = None

        self.output_dir = None
        self.temp_dir = None
        self.model_file = None

        self.predict_output = None


def run(args):
    if not args.no_train:
        # train data for inputs (set some meaningful default if not set)
        if not args.dlsa_train_input:
            raise ValueError(
                "Must specify DLSA train input for consult train.")
        if not args.dlsa_train_output:
            raise ValueError(
                "Must specify DLSA train output for consult train.")
        if not args.vision_train_output:
            raise ValueError(
                "Must specify vision train output for consult train.")

        if not args.model_file and args.output_dir:
            args.model_file = os.path.join(args.output_dir, "consult-model.csv")
            print("model-file is not specified. Default to: {}".format(
                args.model_file))

        consult_trainer = ConsultTrainer()

        print("Start consult training...")
        consult_trainer.train(
            dlsa_train_input=args.dlsa_train_input,
            dlsa_train_output=args.dlsa_train_output,
            vision_train_output=args.vision_train_output,
            model_file=args.model_file)
        print("End consult training.")

    if not args.no_predict:
        if not args.predict_output and args.output_dir:
            args.predict_output = os.path.join(args.output_dir, "predict_output.yaml")
            print("predict-output is not specified. Default to: {}".format(
                args.output_dir))

        # predict data for inputs
        if not args.dlsa_predict_input:
            raise ValueError(
                "Must specify DLSA predict input for consult predict.")
        if not args.dlsa_predict_output:
            raise ValueError(
                "Must specify DLSA predict output for consult predict.")
        if not args.vision_predict_output:
            raise ValueError(
                "Must specify vision predict output for consult predict.")

        if not args.model_file:
            raise ValueError(
                "Must specify the model dir stored the output model.")
        if not args.predict_output:
            raise ValueError(
                "Must specify the predict output for storing test results.")

        consult_predictor = ConsultTrainer()
        # load model
        consult_predictor.load_model(
            args.model_file)

        print("Start consult predicting...")
        consult_predictor.predict(
            dlsa_predict_input=args.dlsa_predict_input,
            dlsa_predict_output=args.dlsa_predict_output,
            vision_predict_output=args.vision_predict_output,
            output_file=args.predict_output)
        print("End consult predicting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consult Model Training")

    parser.add_argument(
        "--no-train", "--no_train",
        default=False, action="store_true",
        help="whether to do train")
    parser.add_argument(
        "--no-predict", "--no_predict",
        default=False, action="store_true",
        help="whether to do prediction on test data")

    parser.add_argument(
        "--dlsa-train-input", "--dlsa_train_input",
        type=str,
        help="The path to the DLSA train input")
    parser.add_argument(
        "--dlsa-train-output", "--dlsa_train_output",
        type=str,
        help="The path to the DLSA train output")
    parser.add_argument(
        "--vision-train-output", "--vision_train_output",
        type=str,
        help="The path to the vision train output")

    parser.add_argument(
        "--dlsa-predict-input", "--dlsa_predict_input",
        type=str,
        help="The path to the DLSA predict input")
    parser.add_argument(
        "--dlsa-predict-output", "--dlsa_predict_output",
        type=str,
        help="The path to the DLSA predict output")
    parser.add_argument(
        "--vision-predict-output", "--vision_predict_output",
        type=str,
        help="The path to the vision predict output")

    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        help="The path to the output")
    parser.add_argument(
        "--temp-dir", "--temp_dir",
        type=str,
        help="The path to the intermediate data")

    parser.add_argument(
        "--model-file", "--model_file",
        type=str,
        help="The path to the model file")

    parser.add_argument(
        "--predict-output", "--predict_output",
        type=str,
        help="The path to the predict output")

    args = parser.parse_args()
    print(args)

    run(args)
