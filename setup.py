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
            "pytest>=4.4.2"
        ]
    },
    package_data={
        "opendp.whitenoise.core": [
            os.path.join("opendp", "whitenoise", "core", "lib", filename) for filename in [
                "libwhitenoise_validator.dll",
                "libwhitenoise_validator.so",
                "libwhitenoise_validator.dylib",

                "libwhitenoise_runtime.dll",
                "libwhitenoise_runtime.so",
                "libwhitenoise_runtime.dylib",
            ]
        ]
    }
)
