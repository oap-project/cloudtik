import os
from os import path
from typing import Tuple, Any
from sklearn.model_selection import train_test_split


def get_split_output_dir(output_dir):
    return os.path.join(output_dir, "annotation", "split")


def split_data(
        df, test_size: float) -> Tuple[Any, Any]:
    """Split the dataset into training and testing sets for NLP.

    Args:
        df: Pandas DataFrame containing the data.
        test_size: Proportion of the dataset to include in the test split.

    Returns:
        Tuple of Pandas DataFrames for training and testing, respectively.
    """
    return train_test_split(df, test_size=test_size)


def split(
        processed_data_dir, output_dir,
        test_size, data_api,
        overwrite=True):
    pd = data_api.pandas()
    output_annotations_file = os.path.join(
        os.path.join(processed_data_dir, "annotation"), "annotation.csv")

    split_data_path = get_split_output_dir(output_dir)
    os.makedirs(split_data_path, exist_ok=True)

    training_data_path = os.path.join(split_data_path, "train.csv")
    testing_data_path = os.path.join(split_data_path, "test.csv")

    if (path.exists(training_data_path) or path.exists(
            testing_data_path)) and not overwrite:
        print("Training or testing data already exist. Skip split.")
        return

    # create training and testing data for NLP if overwrite_training_testing_ids is True
    # read the input data
    input_data = pd.read_csv(output_annotations_file)

    # create training and testing data
    training_data, testing_data = split_data(
        input_data, test_size)

    # save training and testing data
    training_data.to_csv(training_data_path, index=False)
    testing_data.to_csv(testing_data_path, index=False)

    return training_data, testing_data
