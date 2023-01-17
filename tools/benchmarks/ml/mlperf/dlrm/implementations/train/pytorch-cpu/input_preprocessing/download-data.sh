#!/bin/bash

# download Criteo Terabyte dataset
# the dataset contains 24 zipped files and requires about 1 TB storage for the data and another 2 TB for immediate results
data_path=$HOME/data/dlrm/criteo/
mkdir -p $data_path
cd $data_path
curl -O -C - https://storage.googleapis.com/criteo-cail-datasets/day_{$(seq -s "," 0 23)}.gz
yes n | gunzip day_{0..23}.gz

# upload unzipped dataset to hadoop fs
hadoop fs -mkdir -p $data_path
hadoop fs -put $data_path/* $data_path
