#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# This script works for ubuntu
sudo apt-get update -y && sudo apt-get install -y curl unzip sed cmake gcc g++ net-tools

bash ${SCRIPT_DIR}/install-conda.sh
