#!/usr/bin/env bash
set -e

bash scripts/clean.sh

python setup.py sdist -d ./wheelhouse
python scripts/code_generation.py
python setup.py bdist_wheel -d ./wheelhouse