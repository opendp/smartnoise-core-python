#!/usr/bin/env bash
set -e

# make sure the opendp/_native* files exist
if [[ ! -f "opendp/_native_runtime.py" ]]
then
  python3 setup.py develop
fi

sphinx-apidoc --implicit-namespaces -fFe -H opendp-whitenoise-core -A "Consequences of Data" -V 0.1.0 -o docs_temp/source/ opendp opendp/whitenoise.core/*_pb2.py --templatedir templates/

# destroy prior generated documentation and completely rebuild
rm -r docs
sphinx-build -b html docs_temp/source/ docs

rm -r docs_temp
