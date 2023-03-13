# Bad hack so package fails on import
import sys

_DEPRECATED_MSG = """
-----------------------------------
*** Notice ***
-----------------------------------

The SmartNoise Core package is deprecated. 

Please migrate to the OpenDP library:

Repository: https://github.com/opendp/opendp
Documentation: https://docs.opendp.org
PyPI package: https://pypi.org/project/opendp

(March 13, 2023)
-----------------------------------
"""
if True:
    sys.exit(_DEPRECATED_MSG)


