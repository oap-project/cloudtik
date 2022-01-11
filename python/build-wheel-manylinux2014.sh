#!/bin/bash

set -x

# Cause the script to exit if a single command fails.
set -euo pipefail

# Prerequisites:
# Make sure you have the latest version of pip installed:
python -m pip install --upgrade pip
# Make sure you have the latest version of PyPA’s build installed:
python -m pip install --upgrade build

python -m pip install --upgrade twine

# Build CloudTik wheel, at the same directory where pyproject.toml and setup.py locate:
python setup.py bdist_wheel


# Rename the wheels so that they can be uploaded to PyPI. TODO: This is a hack, we should use auditwheel instead.
for path in ./dist/*.whl; do
  if [ -f "${path}" ]; then
    out="${path//-linux/-manylinux2014}"
    if [ "$out" != "$path" ]; then
        mv "${path}" "${out}"
    fi
  fi
done

