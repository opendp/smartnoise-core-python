from .base import *

# The full namespace is used because the file location comes from an external script
# components.py is generated from scripts/code_generation.py
# in opendifferentialprivacy/smartnoise-core-python
from opendp.smartnoise.core.components import *

import os
if os.environ.get('SN_ACKNOWLEDGE_DEPRECATION', 'false') == 'false':
    import warnings
    warnings.warn('SmartNoise-Core is deprecated. Please migrate to the OpenDP library instead: https://docs.opendp.org', DeprecationWarning)
