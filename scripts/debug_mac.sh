#!/usr/bin/env bash
set -e

# enables full stack traces
export RUST_BACKTRACE=1

# reset all the files
bash scripts/clean.sh

# rebuilds the validator, runtime, protobuf, components.py and python package
python3 -m pip install -e . -v | tee scripts/debug_build.log

# run tests
#python3 -m pytest -x -v | tee debug_tests.log

# run a test application
python3 scripts/app.py | tee scripts/debug_app.log

exit
