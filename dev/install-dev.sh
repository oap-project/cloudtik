#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# This script works for ubuntu
sudo apt-get update && sudo apt-get install -y curl unzip cmake gcc g++

bash ${SCRIPT_DIR}/install-conda.sh
