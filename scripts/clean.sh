#!/usr/bin/env bash
set -e

# delete all temporary/hidden directories

rm -rf build/
rm -rf dist/
rm -rf docs_temp/
rm -rf lib/
rm -rf opendp_whitenoise_core.egg-info
rm -rf wheelhouse/
rm -rf .eggs/

rm -f opendp/whitenoise/core/components.py
rm -f opendp/whitenoise/core/*_pb2.py
rm -f opendp/whitenoise/core/variant_message_map.py
