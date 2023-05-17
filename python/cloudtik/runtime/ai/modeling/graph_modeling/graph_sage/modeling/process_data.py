# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
import time
import math
import argparse

from sklearn import preprocessing
from category_encoders import TargetEncoder

from .utils import existing_file


def process_data(raw_data_file, output_file):
    # Step 1: read and clean the dataframe
    tic = time.time()
    df = pd.read_csv(raw_data_file)
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    print(
        "Time to read the dataframe = {} seconds".format(math.ceil(time.time() - tic))
    )

    # Step 2: Categorical, one-hot, multi-hot encoding (independent of data split)
    tic = time.time()

    df["card_id"] = df["user"].astype("str") + df["card"].astype("str")
    df["card_id"] = df["card_id"].astype("float32")

    df["merchant_id"] = df["merchant_name"].astype("category").cat.codes

    df["amount"] = df["amount"].str.strip("$").astype("float32")

    df["merchant_city"] = df["merchant_city"].astype("category")
    df["merchant_state"] = df["merchant_state"].astype("category")
    df["zip"] = df["zip"].astype("str").astype("category")
    df["mcc"] = df["mcc"].astype("category")
    df["is_fraud?"] = df["is_fraud?"].astype("category").cat.codes

    # One hot encoding `use_chip`
    onehot_enc_cols = ["use_chip"]
    df = pd.concat([df, pd.get_dummies(df[onehot_enc_cols])], axis=1)
    df.drop(
        columns=["use_chip"], axis=1, inplace=True
    )  # we don't need it after being encoded

    # Multi hot encoding the errors
    exploded = df["errors?"].str.strip(",").str.split(",").explode()
    raw_one_hot = pd.get_dummies(exploded, columns=["errors?"])
    errs = raw_one_hot.groupby(raw_one_hot.index).sum()
    df = pd.concat([df, errs], axis=1)
    df.drop(
        columns=["errors?"], axis=1, inplace=True
    )  # we dont need it after being encoded

    # time encoding
    df["time"] = (
        df["time"]
        .str.split(":")
        .apply(lambda x: int(x[0]) * 60 + int(x[1]))
        .astype("uint8")
    )
    df["time"] = (df["time"] - df["time"].min()) / (df["time"].max() - df["time"].min())

    # Step 3 : Data Splitting for target encoding
    df["split"] = pd.Series(np.zeros(df["year"].size), dtype=np.int8)
    df.loc[df["year"] == 2018, "split"] = 1
    df.loc[df["year"] > 2018, "split"] = 2

    # Keep card_id and merchant_id in the validation and test datasets
    # only if they are included in the train datasets.
    train_card_ids = df.loc[df["split"] == 0, "card_id"]
    train_merch_ids = df.loc[df["split"] == 0, "merchant_id"]

    df.loc[(df["split"] != 0) & ~df["card_id"].isin(train_card_ids), "split"] = 3
    df.loc[(df["split"] != 0) & ~df["merchant_id"].isin(train_merch_ids), "split"] = 3

    # step 4 : Target and label encoding using data splits to avoid information leakage
    train_df = df.loc[df["split"] == 0]
    valtest_df = df.loc[(df["split"] == 1) | (df["split"] == 2)]

    # Target encoding
    high_card_cols = ["merchant_city", "merchant_state", "zip", "mcc"]
    for col in high_card_cols:
        tgt_encoder = TargetEncoder(smoothing=0.001)
        train_df[col] = tgt_encoder.fit_transform(
            train_df[col], train_df["is_fraud?"]
        ).astype("float32")
        # display(train_df[col])
        valtest_df[col] = tgt_encoder.transform(valtest_df[col]).astype("float32")

    # Label encoding `is_fraud?`
    label_encoder = preprocessing.LabelEncoder()
    train_df["is_fraud?"] = label_encoder.fit_transform(train_df["is_fraud?"])
    valtest_df["is_fraud?"] = label_encoder.transform(valtest_df["is_fraud?"])

    # merge train, validation and test dataframes into one
    df_merge = pd.concat([train_df, valtest_df])

    print("Time for featurization = {} seconds".format(math.ceil(time.time() - tic)))

    # step 5 : write edge features to a file
    tic = time.time()
    df_merge.to_csv(output_file, index=False)
    print(
        "Writing edge features to csv file takes {} seconds".format(
            math.ceil(time.time() - tic)
        )
    )
    print(df_merge.shape)


def main(args):
    process_data(
        raw_data_file=args.raw_data_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument(
        "--raw_data_file", type=existing_file, help="The path to the raw transaction data file")
    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to the output processed edge data file",)
    args = parser.parse_args()
    print(args)

    main(args)
