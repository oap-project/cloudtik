import os
import shutil
from os import path, listdir
from pandas import DataFrame, read_csv
from typing import Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path


def get_vision_split_output_dir(output_dir):
    return os.path.join(output_dir, "split_images")


def get_dlsa_split_output_dir(output_dir):
    return os.path.join(output_dir, "annotation", "split")


def split_dlsa_data(df: DataFrame, test_size: float) -> Tuple[DataFrame, DataFrame]:
    """Split the dataset into training and testing sets for NLP.

    Args:
        df: Pandas DataFrame containing the data.
        test_size: Proportion of the dataset to include in the test split.

    Returns:
        Tuple of Pandas DataFrames for training and testing, respectively.
    """
    return train_test_split(df, test_size=test_size)


def get_subject_id(image_name):
    """
    Extracts the patient ID from an image filename.

    Args:
    - image_name: string representing the filename of an image

    Returns:
    - patient_id: string representing the patient ID extracted from the image filename
    """

    # Split the filename by "/"
    image_name = image_name.split("/")[-1]

    # Extract the first two substrings separated by "_", remove the first character (which is "P"), and join them
    # together to form the patient ID
    patient_id = "".join(image_name.split("_")[:2])[1:]

    return patient_id


def copy_images(patient_ids: DataFrame, source_folder: str, target_folder: str) -> None:
    """Copy images of selected patients from the source folder to the target folder.

    Args:
        patient_ids: List of patient IDs for whom the images need to be copied.
        source_folder: Path to the source folder containing the images.
        target_folder: Path to the target folder where the images need to be copied.
    """
    for f in listdir(source_folder):
        if ("_CM_" in f) and (get_subject_id(f) in patient_ids.Patient_ID.to_list()):
            full_src_path = path.join(source_folder, f)
            shutil.copy(full_src_path, target_folder)


def split_vision_data(
        train_data: DataFrame, test_data: DataFrame,
        processed_data_dir, dataset_config, output_dir) -> dict:
    """
    This function creates and organizes the train and test dataset for vision task.

    Parameters:
    train_data (pd.DataFrame): A pandas DataFrame containing the training data.
    test_data (pd.DataFrame): A pandas DataFrame containing the testing data.
    processed_data_dir (str): The root directory path for the processed path.
    dataset_config (dict): A dictionary containing the dataset information.
    output_dir (str): The output dir.

    Returns:
    dict: A dictionary containing the configuration information.
    """
    output_segmented_dir = os.path.join(processed_data_dir, "segmented_images")

    # Get the list of class labels and the column containing the class label
    label_list = dataset_config["label_list"]
    label_column = dataset_config["features"]["class_label"]

    # Create the target directory for the dataset
    split_images_dir = get_vision_split_output_dir(output_dir)
    if path.exists(split_images_dir):
        shutil.rmtree(split_images_dir)

    # Iterate over train and test data
    for cat in ["test", "train"]:
        # Get the data based on the current category
        df = test_data if cat == "test" else train_data
        
        # Iterate over each label
        for label in label_list:
            # Source folder for images
            source_folder = path.join(output_segmented_dir, label)
            
            # Create target folder
            target_folder = path.join(split_images_dir, cat, label)
            Path(target_folder).mkdir(parents=True, exist_ok=True)

            # Copy images
            patient_ids = df[df[label_column] == label]
            copy_images(patient_ids, source_folder, target_folder)


def split(
        processed_data_dir, output_dir, test_size, overwrite=True):
    output_annotations_file = os.path.join(
        os.path.join(processed_data_dir, "annotation"), "annotation.csv")

    split_data_path = get_dlsa_split_output_dir(output_dir)
    training_data_path = os.path.join(split_data_path, "train.csv")
    testing_data_path = os.path.join(split_data_path, "test.csv")

    if (path.exists(training_data_path) or path.exists(
            testing_data_path)) and not overwrite:
        print("Training or testing data already exist. Skip split.")
        return

    # create training and testing data for NLP if overwrite_training_testing_ids is True
    # read the input data
    input_data = read_csv(output_annotations_file)

    # create training and testing data
    training_data, testing_data = split_dlsa_data(
        input_data, test_size)

    # save training and testing data
    training_data.to_csv(training_data_path, index=False)
    testing_data.to_csv(testing_data_path, index=False)

    # create vision data
    split_vision_data(training_data, testing_data)
