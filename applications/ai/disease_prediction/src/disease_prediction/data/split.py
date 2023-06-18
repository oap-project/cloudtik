from disease_prediction.dlsa.data.split import split as split_dlsa_data
from disease_prediction.vision.data.split import split as split_vision_data


def split(
        processed_data_dir, output_dir,
        test_size, dataset_config, overwrite=True):
    dlsa_training_data, dlsa_testing_data = split_dlsa_data(
        processed_data_dir=processed_data_dir,
        output_dir=output_dir,
        test_size=test_size, overwrite=overwrite
    )

    # create vision data
    split_vision_data(
        dlsa_training_data, dlsa_testing_data,
        processed_data_dir=processed_data_dir,
        dataset_config=dataset_config,
        output_dir=output_dir)
