#!/usr/bin/env bash
set -e

# make sure the opendp/_native* files exist
if [[ ! -d "opendp/whitenoise/core/lib" ]]; then
  python3 scripts/code_generation.py
fi

WN_VERSION=0.2.1
# destroy prior generated documentation and completely rebuild
rm -r docs || true
mkdir -p docs
sphinx-build -b html docs_temp/source/ docs
touch docs/.nojekyll

rm -r docs_temp || true
