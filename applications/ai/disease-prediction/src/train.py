import argparse

from .train_vision import run as run_train_vision
from .sentiment_analysis.run import run as run_train_doc


def run(args):
    # run vision and doc training

    run_train_vision(args)

    run_train_doc(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease Prediction Training")

    args = parser.parse_args()
    print(args)

    run(args)
