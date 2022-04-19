#!/bin/bash

set -x

# Cause the script to exit if a single command fails.
set -euo pipefail

# Prerequisites:
# Make sure you have the latest version of pip installed:
python -m pip install --upgrade pip
# Make sure you have the latest version of PyPA’s build installed:
python -m pip install --upgrade build

# Update the commit sha
if [ -n "$TRAVIS_COMMIT" ]; then
    CLOUDTIK_COMMIT_SHA=$TRAVIS_COMMIT
fi

if [ ! -n "$CLOUDTIK_COMMIT_SHA" ]; then
    CLOUDTIK_COMMIT_SHA=$(which git >/dev/null && git rev-parse HEAD)
fi

if [ ! -z "$CLOUDTIK_COMMIT_SHA" ]; then
  sed -i.bak "s/__commit__ = \".*\"/__commit__ = \"$CLOUDTIK_COMMIT_SHA\"/g" ./cloudtik/__init__.py && rm ./cloudtik/__init__.py.bak
fi

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

