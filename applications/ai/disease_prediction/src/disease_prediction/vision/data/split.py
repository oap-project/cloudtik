import os
import shutil
from os import path, listdir
from pandas import DataFrame
from pathlib import Path

from disease_prediction.utils import get_subject_id


def get_split_output_dir(output_dir):
    return os.path.join(output_dir, "split_images")


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


def split_data(
        train_data: DataFrame, test_data: DataFrame,
        processed_data_dir, dataset_config, output_dir):
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
    split_images_dir = get_split_output_dir(output_dir)
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

    return split_images_dir


def split(
        dlsa_training_data, dlsa_testing_data,
        processed_data_dir, dataset_config,
        output_dir):
    # create vision data
    split_data(
        dlsa_training_data, dlsa_testing_data,
        processed_data_dir=processed_data_dir,
        dataset_config=dataset_config,
        output_dir=output_dir)
