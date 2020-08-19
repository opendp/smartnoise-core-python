#!/usr/bin/env bash
set -e

# make sure the opendp/_native* files exist
if [[ ! -d "opendp/whitenoise/core/lib" ]]; then
  python3 scripts/code_generation.py
fi

WN_VERSION=0.2.0
sphinx-apidoc -fFe -H opendp-whitenoise-core -A "Consequences of Data" -V $WN_VERSION -o docs_temp/source/ opendp opendp/whitenoise/core/*_pb2.py --templatedir templates/

# destroy prior generated documentation and completely rebuild
rm -r docs || true
mkdir -p docs
sphinx-build -b html docs_temp/source/ docs
touch docs/.nojekyll

rm -r docs_temp || true
