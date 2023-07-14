# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from disease_prediction.utils import get_subject_id, read_yaml_file


def process_dlsa_output(output_file, origin_input, pd):
    """
    Processes predictions obtained from DSLA predict.

    Args:
        output_file: The output file of a dictionary containing the predictions.
        origin_input: The original input for the predictions.
        pd: pandas API module

    Returns:
        A dictionary containing the processed predictions.
    """
    output_dict = read_yaml_file(output_file)

    # Add patient IDs to the YAML dictionary.
    output_dict["id"] = pd.read_csv(origin_input)["Patient_ID"].to_list()

    return output_dict


def process_vision_output(output_file, pd):
    """
    Processes predictions obtained from a computer vision.

    Args:
        output_file: The output file of a dictionary containing the predictions.
        pd: pandas API module

    Returns:
        A dictionary containing the processed predictions.
    """
    output_dict = read_yaml_file(output_file)

    # Get the predictions for each file.
    results_dict = output_dict["results"]

    # Create a DataFrame for each file.
    df_list = []
    for f in results_dict.keys():
        temp_dict = results_dict[f]
        temp_dict[0]["file"] = f
        temp_dict[0]["id"] = get_subject_id(f)
        df_list.append(pd.DataFrame(temp_dict))

    # Concatenate the DataFrames and group by patient ID.
    df = pd.concat(df_list)
    df_g = df.groupby("id")

    # For each patient ID, calculate the mean prediction per class.
    # Then, assign the class with the highest mean prediction as the predicted class.
    df_list = []
    for k in df_g.groups.keys():
        df_temp = df_g.get_group(k).reset_index(drop=True)
        mean_per_class = np.mean(np.array(df_temp.pred_prob.to_list()), axis=0)
        pred = output_dict["label"][np.argmax(mean_per_class)]

        df_temp.loc[0, "pred_prob"] = mean_per_class
        df_temp.loc[0, "pred"] = pred

        df_list.append(df_temp.loc[[0]])

    df_results = pd.concat(df_list)

    # Create a dictionary with the processed predictions.
    predictions_report = {}

    if "label" in df_results.columns:
        predictions_report["label_id"] = df_results.label.to_list()
    predictions_report["predictions_label"] = df_results.pred.to_list()
    predictions_report["predictions_probabilities"] = df_results.pred_prob.to_list()
    predictions_report["id"] = df_results.id.to_list()

    return predictions_report
