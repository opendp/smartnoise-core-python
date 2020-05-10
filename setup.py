from setuptools import setup, find_namespace_packages
import os

setup(
    packages=find_namespace_packages(include=["opendp.*"]),
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
        "opendp.whitenoise": [
            os.path.join("core", "lib", filename) for filename in [
                "libwhitenoise_ffi.dll",
                "libwhitenoise_ffi.so",
                "libwhitenoise_ffi.dylib",
            ]
        ]
    }
)
