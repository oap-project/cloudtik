#!/bin/bash

function install_libraries() {
    pip install -qq tqdm joblib pandas pyarrow fastparquet
}


install_libaries
