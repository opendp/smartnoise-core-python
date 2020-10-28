#!/usr/bin/env bash
set -e

# delete all temporary/hidden directories

rm -rf build/
rm -rf dist/
rm -rf docs_temp/
rm -rf tmp_binaries/

rm -rf lib/
rm -rf opendp_smartnoise_core.egg-info
rm -rf opendp_smartnoise_core_python.egg-info
rm -rf wheelhouse/
rm -rf .eggs/
rm -rf opendp/smartnoise/core/lib/

rm -f opendp/smartnoise/core/components.py
rm -f opendp/smartnoise/core/*_pb2.py
rm -f opendp/smartnoise/core/variant_message_map.py
