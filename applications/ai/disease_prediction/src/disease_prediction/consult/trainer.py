# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from pandas import read_csv, DataFrame, concat

from disease_prediction.f1_score.data.process import \
    process_dlsa_output, process_vision_output
from disease_prediction.f1_score.trainer import Trainer as F1ScoreTrainer
from disease_prediction.utils import read_yaml_file


class Trainer(object):
    def __init__(self):
        # per_class f1_scores is the model
        self.dlsa_f1_score_trainer = F1ScoreTrainer()
        self.vision_f1_score_trainer = F1ScoreTrainer()

    def train(self,
              dlsa_train_input,
              dlsa_train_output,
              vision_train_output,
              model_file=None):
        """
        Train/get the f1-score of the predictions from DLSA and Vision models
        Args:
            dlsa_train_input: The DLSA train prediction input
            dlsa_train_output: The DLSA train prediction output
            vision_train_output: The vision train prediction output
            model_file: The model file to save. Will not save model if None
        """
        # process the original train output and return the dict
        dlsa_train_output = process_dlsa_output(
            dlsa_train_output, dlsa_train_input)
        vision_train_output = process_vision_output(
            vision_train_output)

        self.dlsa_f1_score_trainer.train(dlsa_train_output)
        self.vision_f1_score_trainer.train(vision_train_output)

        # store the 'model' data to file for reuse without train
        if model_file:
            self.save_model(model_file)

    def predict(self,
                dlsa_predict_input,
                dlsa_predict_output,
                vision_predict_output,
                output_file):
        """
        Generate the consultation predictions from DLSA and Vision models
        Args:
            dlsa_predict_input: The DLSA prediction input
            dlsa_predict_output: The DLSA prediction output
            vision_predict_output: The vision prediction output
            output_file: The predict output file to save. Will not save if None
        Returns:
            df_results (pandas.DataFrame):
                a DataFrame containing the consultation predictions and the labels
        """
        dlsa_processed_output = process_dlsa_output(
            dlsa_predict_output, dlsa_predict_input)
        vision_processed_output = process_vision_output(
            vision_predict_output)

        # Get scored prediction probabilities for DLSA and Vision models
        dlsa_scored_pred = self.dlsa_f1_score_trainer.predict(
            dlsa_processed_output)
        vision_scored_pred = self.vision_f1_score_trainer.predict(
            vision_processed_output)

        # Sort predictions by ID and align IDs between DLSA and Vision models
        dlsa_scored_pred = dlsa_scored_pred.sort_values(by="id").reset_index(drop=True)
        vision_scored_pred = vision_scored_pred.sort_values(by="id").reset_index(drop=True)
        if (len(dlsa_scored_pred) != len(vision_scored_pred)) or (
                all(dlsa_scored_pred.id != vision_scored_pred.id)
        ):
            dlsa_scored_pred, vision_scored_pred = set_diff_ids(
                dlsa_scored_pred, vision_scored_pred
            )

        # Create consultation scores and map predicted labels to their original label names
        pred = consult_predictions(
            dlsa_scored_pred["predictions_probabilities"].to_list(),
            vision_scored_pred["predictions_probabilities"].to_list(),
        )

        vision_output = read_yaml_file(vision_predict_output)
        pred_mapped = [vision_output["label"][i] for i in pred]

        # Create DataFrame with predictions and labels
        df_results = DataFrame()
        df_results["labels"] = vision_scored_pred["labels"]
        df_results["dlsa_predictions"] = dlsa_scored_pred["predictions_label"]
        df_results["vision_predictions"] = vision_scored_pred["predictions_label"]
        df_results["consultation_predictions"] = pred_mapped

        if output_file:
            # save the result to file if needed
            df_results.to_csv(
                output_file, index=False, header=True)

        return df_results

    def load_model(self, model_file):
        model_df = read_csv(
            model_file, index_col=None, header=0)

        self.dlsa_f1_score_trainer.f1_scores = model_df['dlsa-f1-score']
        self.vision_f1_score_trainer.f1_scores = model_df['vision-f1-score']

    def save_model(self, model_file):
        if self.dlsa_f1_score_trainer.f1_scores is None or (
                self.vision_f1_score_trainer.f1_scores is None):
            raise RuntimeError("No trained or loaded model to save.")

        # self.f1_scores.rename('f1-score', inplace=True)
        model_df = DataFrame()
        model_df["dlsa-f1-score"] = self.dlsa_f1_score_trainer.f1_scores
        model_df["vision-f1-score"] = self.vision_f1_score_trainer.f1_scores
        model_df.to_csv(
            model_file, index=False, header=True)


def consult_predictions(dlsa_scored_pred, vision_scored_pred):
    """
    Create consultation scores by adding the scored prediction scores for both DLSA and vision models element-wise.

    Args:
    dlsa_scored_pred: List containing scored prediction scores for DLSA model.
    vision_scored_pred: List containing scored prediction scores for vision model.

    Returns:
    List containing predicted labels based on consultation scores.
    """
    consultation_scored_pred = np.array(
        dlsa_scored_pred) + np.array(vision_scored_pred)
    return [np.argmax(i) for i in consultation_scored_pred]


def set_diff_ids(df1, df2):
    """
    Concatenates dataframes and ensures both dataframes have the same unique IDs.

    Args:
    df1: First dataframe.
    df2: Second dataframe.

    Returns:
    Two dataframes with the same unique IDs.
    """
    # find the IDs in df1 that are not in df2
    diff_ids_1 = list(set(df1.id) - set(df2.id))

    # concatenate df1 and the subset of df1 that contains the IDs not in df2
    if diff_ids_1:
        df2 = (
            concat([df2, df1[df1.id.isin(diff_ids_1)]])
            .sort_values(by="id")
            .reset_index(drop=True)
        )

    # find the IDs in df2 that are not in df1
    diff_ids_2 = list(set(df2.id) - set(df1.id))

    # concatenate df2 and the subset of df2 that contains the IDs not in df1
    if diff_ids_2:
        df1 = (
            concat([df1, df2[df2.id.isin(diff_ids_2)]])
            .sort_values(by="id")
            .reset_index(drop=True)
        )

    return df1, df2
