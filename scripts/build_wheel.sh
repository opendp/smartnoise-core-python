#!/usr/bin/env bash
set -e

echo "(1) Clean, delete all temporary directories";
bash scripts/clean.sh

#echo "(1a) Build validator, to have .protos generated before next step";
#cd smartnoise-core;
#cargo build --manifest-path=validator-rust/Cargo.toml
#cd ..;

# Generate python classes from protobuf definitions as well as
#  components.py and variant_message_map.py files
#
echo "(2) Generate components.py, *_pb2.py and binaries";
python3 scripts/code_generation.py

# Build the source distribution (.tar.gz as specified in the setup.cfg)
#
#echo "(3) Build the source distribution";
#python3 setup.py sdist -d ./wheelhouse

# Build the binary distribution (wheel)
#
echo "(3) Build the final wheel";
python3 setup.py bdist_wheel -d ./wheelhouse
