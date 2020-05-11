#!/usr/bin/env bash
set -e

echo "(1) Clean, delete all temporary directories";
bash scripts/clean.sh

echo "(1a) Build validator, to have .protos generated before next step";
cd whitenoise-core;
cargo build --manifest-path=validator-rust/Cargo.toml
cd ..;

# Build the source distribution (.tar.gz as specified in the setup.cfg)
#
echo "(2) Build the source distribution (may skip sdist in future builds)";
python3 setup.py sdist -d ./wheelhouse

# Generate python classes from protobuf definitions as well as
#  components.py and variant_message_map.py files
#
echo "(3) Generate python classes from protobuf definitions";
python3 scripts/code_generation.py


# Build the binary distribution (wheel)
#
echo "(4) Build the binary distribution";
python3 setup.py bdist_wheel -d ./wheelhouse
