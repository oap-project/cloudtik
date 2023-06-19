# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.metrics import classification_report


class Trainer(object):
    def __init__(self):
        # per_class f1_scores is the model
        self.f1_scores = None

    def train(self, train_data, model_file=None):
        """
        Get/train f1_scores from training output of other model
        Args:
            train_data: train data is the processed output of
            previous prediction results from model train
            model_file: to save the trained model to file

        """
        self.f1_scores = f1_score_per_class(train_data)
        if model_file:
            self.save_model(model_file)
        return self.f1_scores

    def predict(self, test_data) -> DataFrame:
        """
        Correct the probabilities using the f1_score of each class
        Args:
            test_data: test data is the processed output of
            previous prediction results from model predict.

        Returns: The corrected probabilities of each class using the f1_score
        """
        f1_scores = self.f1_scores

        # corrected scores based on f1-scores
        scrored_probabilities = [
            list(f1_scores * i) for i in test_data["predictions_probabilities"]
        ]

        # create a pandas DataFrame with corrected predictions, labels and IDs
        df_scored = DataFrame()
        if 'label_id' in test_data:
            df_scored["labels"] = test_data["label_id"]
        df_scored["predictions_label"] = test_data["predictions_label"]
        df_scored["predictions_probabilities"] = scrored_probabilities
        df_scored["id"] = test_data["id"]

        return df_scored

    def load_model(self, model_file):
        model_df = read_csv(
            model_file, index_col=None, header=0)
        self.f1_scores = model_df["f1-score"]

    def save_model(self, model_file):
        if self.f1_scores is None:
            raise RuntimeError("No trained or loaded model to save.")

        self.f1_scores.rename('f1-score', inplace=True)
        self.f1_scores.to_csv(
            model_file, index=False, header=True)


def f1_score_per_class(output_dict):
    """
    Compute the F1-score per class based on the predictions in a YAML dictionary.

    Args:
    - output_dict (dict): A dictionary containing the predictions for a given task.

    Returns:
    - f1_scores (pandas.Series): A Series containing the F1-score per class.
    """
    # Extract ground truth labels and predicted labels
    truth = output_dict["label_id"]
    pred = output_dict["predictions_label"]

    # Compute the classification report and convert to DataFrame
    cm_res = classification_report(truth, pred, output_dict=True)
    df_cm_res = DataFrame(cm_res).transpose().round(3)

    # Extract the F1-score per class and return as a Series
    f1_scores = df_cm_res.loc[[str(i) for i in np.unique(truth)], "f1-score"]
    return f1_scores
