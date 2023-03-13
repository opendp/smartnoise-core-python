__path__ = __import__('pkgutil').extend_path(__path__, __name__)

import sys
deprecated_msg = '''
-----------------------------------
*** Deprecated ***
-----------------------------------

The SmartNoise Core package is deprecated. 

Please migrate to the OpenDP library:

Repository: https://pypi.org/project/opendp
Documentation: https://docs.opendp.org
PyPI package: https://pypi.org/project/opendp

-----------------------------------
'''

# Bad hack so package fails on install
from datetime import datetime
deprecated_day = datetime(year=2023, month=3, day=13, hour=15, minute=25)
if datetime.now() > deprecated_day:
    sys.exit(deprecated_msg)
