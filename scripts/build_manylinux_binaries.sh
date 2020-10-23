#!/bin/bash

# run from within the manylinux docker containers

# exit immediately upon failure, print commands while running
set -e -x

# Install rust inside the manylinux container
#
echo ">>> install rust if it does not exist";
if ! [ -x "$(command -v cargo)" ]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
	export PATH="${HOME}/.cargo/bin:${PATH}"
fi

export WN_USE_SYSTEM_LIBS=false;
export WN_DEBUG=false;
export WN_USE_VULNERABLE_NOISE=false;

echo ">>> build the binaries";
cargo +stable build --features use-direct-api --release --manifest-path=io/smartnoise-core/ffi-rust/Cargo.toml