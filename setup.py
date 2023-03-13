from setuptools import setup
import sys
import warnings

warnings.warn('SmartNoise-Core is deprecated. Please migrate to the OpenDP library: https://github.com/opendp/opendp, https://docs.opendp.org', DeprecationWarning)


setup()



"""
setup(
    extras_require={
        "plotting": [
            "networkx",
            "matplotlib"
        ],
        "test": [
            "pytest>=4.4.2",
            "pandas>=1.0.3"
        ]
    },
    package_data={
        "opendp.smartnoise": [
            os.path.join("core", "lib", filename) for filename in [
                "smartnoise_ffi.dll",
                "libsmartnoise_ffi.so",
                "libsmartnoise_ffi.dylib",
            ]
        ]
    }
)
"""
