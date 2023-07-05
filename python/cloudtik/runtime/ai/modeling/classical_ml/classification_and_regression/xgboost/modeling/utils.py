import argparse
import os
import numpy as np
import glob
import yaml

from cloudtik.runtime.ai.util.utils import clean_dir

DATA_ENGINE_PANDAS = 'pandas'
DATA_ENGINE_MODIN = 'modin'


def existing_directory(raw_path):
    if not os.path.isdir(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing directory'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def existing_file(raw_path):
    if not os.path.isfile(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def existing_path(raw_path):
    if not os.path.exists(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing directory or file'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def read_csv_file(file, pd, ignore_cols=None):
    csv = pd.read_csv(file)
    if ignore_cols is not None:
        print("dropping columns...")
        csv.drop(columns=ignore_cols, inplace=True)
    else:
        print("reading without dropping columns...")
    return csv


def read_csv_files(raw_data_path, engine, ignore_cols=None):
    if engine == DATA_ENGINE_PANDAS:
        import pandas as pd
    elif engine == DATA_ENGINE_MODIN:
        import modin.pandas as pd
    else:
        raise ValueError('Engine can either be pandas or modin.')

    if os.path.isfile(raw_data_path):
        # single csv file
        data = read_csv_file(raw_data_path, pd, ignore_cols)
        print(f"data has the shape {data.shape}")
        return data

    files = glob.glob(f'{raw_data_path}/*.csv')
    df = []
    for file in files:
        csv = read_csv_file(file, pd, ignore_cols)
        df.append(csv)
    data = pd.concat(df)
    print(f"data has the shape {data.shape}")
    return data


def partition_data(df, save_format, save_data_path, num_partitions):
    clean_dir(save_data_path)
    if save_format == 'csv':
        df_splits = np.array_split(df, num_partitions)
        for i, data in enumerate(df_splits):
            data.to_csv(f"{save_data_path}/partition_{i}.csv", index=False)
    else:
        print("other data format not supported")


def has_dir(data_path, folder_name):
    for fname in os.listdir(data_path):
        if fname == folder_name and os.path.isdir(os.path.join(data_path, fname)):
            return True
    return False


def read_parquet_spark(spark, data_path):
    data = spark.read.parquet(data_path)
    print(f'({data.count()}, {len(data.columns)})')
    return data


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
