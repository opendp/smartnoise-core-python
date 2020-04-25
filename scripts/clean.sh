#!/usr/bin/env bash
set -e

# delete all temporary/hidden directories

rm -rf build/
rm -rf dist/
rm -rf docs_temp/

rm -f opendp/_native*
rm -rf opendp_whitenoise_core.egg-info
rm -f opendp/whitenoise_core/components.py
rm -f opendp/whitenoise_core/*_pb2.py
rm -f opendp/whitenoise_core/variant_message_map.py

rm -rf wheelhouse/
rm -rf .eggs/