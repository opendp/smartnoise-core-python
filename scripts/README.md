# Notes about scripts

## Creating a PyPI package

This process takes several steps which involve changes in both this repository as well as the `whitenoise-core` submodule. 

Notes:
  - Make sure you have the python requirements installed.
      ``` 
      pip install -r requirements/dev.txt
      ```
  - For publishing to PyPI, you'll need PyPI credentials which allow uploading to: https://pypi.org/project/opendp-whitenoise-core/
  - All example commands run from the Terminal and start within the `whitenoise-core-python` directory--the top of this repository.

---

1. Make sure the `whitenoise-core` submodule is pointing to the correct branch, usually `develop`
    ```
    cd whitenoise-core
    git branch
    * develop
    ```
1. Edit the file `whitenoise-core/scripts/update_version.toml`, updating:
    - version numbers
    - paths to this repository as well as the `whitenoise-core` submodule
    ```
    # vim or your editor of choice
    vim whitenoise-core/scripts/update_version.toml
    ```
1. Run the `update_version.py` script
    ```
    cd whitenoise-core
    python scripts/update_version.py 
    ```
    This command runs quickly and may generate no output.
1. Run the command to build the wheel. 
    ```
    # return to the directory `whitenoise-core-python`
    #
    ./scripts/build_production_wheel.sh
    ```
    This can take a **considerable amount of time, up to 30 minutes**.
    - Note, this script has several commented out sections: (1), (2), and (3). Uncommenting them allows further manual error checking--for example making sure that the correct GLIBC libraries are being used. General users can leave these sections commented.
1. In the terminal output, look for the text `[optimized]`. It should appear twice, once for building the linux wheel and another time for the OS X wheel. An example of a line indicating a successful wheel build is:
    ```
    Finished release [optimized] target(s) in 23m 44s
    ```
    This line ^ does not appear at the end of the output--use a search to find it.
1. A wheel should have been created. For example:
    ```
    ls wheelhouse/
    opendp_whitenoise_core-0.2.1-py3-none-any.whl
    ```
1. Double-check that the wheel includes the two files: `libwhitenoise_ffi.so` and `libwhitenoise_ffi.dylib`
    ```
    ls wheelhouse/
    # copy the .whl file, renaming it as a zip
    #
    cp opendp_whitenoise_core-[x.y.z]-py3-none-any.whl /tmp/check_wheel.zip
    #
    # Unzip it and examine the listing for:
    # - opendp/whitenoise/core/lib/libwhitenoise_ffi.so
    # - opendp/whitenoise/core/lib/libwhitenoise_ffi.dylib
    #
    cd /tmp
    unzip check_wheel.zip
    Archive:  check_wheel.zip
        inflating: opendp/whitenoise/__init__.py  
        inflating: opendp/whitenoise/core/__init__.py  
        inflating: opendp/whitenoise/core/api.py  
        inflating: opendp/whitenoise/core/api_pb2.py  
        inflating: opendp/whitenoise/core/base.py  
        inflating: opendp/whitenoise/core/base_pb2.py  
        inflating: opendp/whitenoise/core/components.py  
        inflating: opendp/whitenoise/core/components_pb2.py  
        inflating: opendp/whitenoise/core/value.py  
        inflating: opendp/whitenoise/core/value_pb2.py  
        inflating: opendp/whitenoise/core/variant_message_map.py  
        inflating: opendp/whitenoise/core/lib/libwhitenoise_ffi.dylib  
        inflating: opendp/whitenoise/core/lib/libwhitenoise_ffi.so  
        inflating: opendp_whitenoise_core-0.2.1.dist-info/LICENSE  
        inflating: opendp_whitenoise_core-0.2.1.dist-info/METADATA  
        inflating: opendp_whitenoise_core-0.2.1.dist-info/WHEEL  
        inflating: opendp_whitenoise_core-0.2.1.dist-info/top_level.txt  
        inflating: opendp_whitenoise_core-0.2.1.dist-info/RECORD 
    ```
    Looks good! The two files appear ^
1. Delete the /tmp files
    ```
    rm /tmp/check_wheel.zip
    rm -rf /tmp/opendp
    rm -rf opendp_whitenoise_core*
    ```
1. Upload the wheel to PyPI!
    ```
    # You will be prompted for your credentials (unless they're set by env. variables, etc)
    #
    python -m twine upload --verbose --skip-existing wheelhouse/*
    ```
1. Check the url for your release: https://pypi.org/project/opendp-whitenoise-core/
