# Notes about scripts

## Creating a PyPI package

This process takes several steps which involve changes in both this repository as well as the `smartnoise-core` submodule. 

Notes:
  - Make sure you have the python requirements installed.
      ``` 
      pip install -r requirements/dev.txt
      ```
  - For publishing to PyPI, you'll need PyPI credentials which allow uploading to: https://pypi.org/project/opendp-smartnoise-core/
  - All example commands run from the Terminal and start within the `smartnoise-core-python` directory--the top of this repository.

---

1. Make sure the `smartnoise-core` submodule is pointing to the correct branch, usually `develop`
    ```
    cd smartnoise-core
    git branch
    * develop
    ```
1. Edit the file `smartnoise-core/scripts/update_version.toml`, updating:
    - version numbers
    - paths to this repository as well as the `smartnoise-core` submodule
    ```
    # vim or your editor of choice
    vim smartnoise-core/scripts/update_version.toml
    ```
1. Run the `update_version.py` script
    ```
    cd smartnoise-core
    python scripts/update_version.py 
    ```
    This command runs quickly and may generate no output.
1. Run the command to build the wheel. 
    ```
    # return to the directory `smartnoise-core-python`
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
    opendp_smartnoise_core-0.2.1-py3-none-any.whl
    ```
1. Double-check that the wheel includes the two files: `libsmartnoise_ffi.so` and `libsmartnoise_ffi.dylib`
    ```
    ls wheelhouse/
    # copy the .whl file, renaming it as a zip
    #
    cp opendp_smartnoise_core-[x.y.z]-py3-none-any.whl /tmp/check_wheel.zip
    #
    # Unzip it and examine the listing for:
    # - opendp/smartnoise/core/lib/libsmartnoise_ffi.so
    # - opendp/smartnoise/core/lib/libsmartnoise_ffi.dylib
    #
    cd /tmp
    unzip check_wheel.zip
    Archive:  check_wheel.zip
        inflating: opendp/smartnoise/__init__.py  
        inflating: opendp/smartnoise/core/__init__.py  
        inflating: opendp/smartnoise/core/api.py  
        inflating: opendp/smartnoise/core/api_pb2.py  
        inflating: opendp/smartnoise/core/base.py  
        inflating: opendp/smartnoise/core/base_pb2.py  
        inflating: opendp/smartnoise/core/components.py  
        inflating: opendp/smartnoise/core/components_pb2.py  
        inflating: opendp/smartnoise/core/value.py  
        inflating: opendp/smartnoise/core/value_pb2.py  
        inflating: opendp/smartnoise/core/variant_message_map.py  
        inflating: opendp/smartnoise/core/lib/libsmartnoise_ffi.dylib  
        inflating: opendp/smartnoise/core/lib/libsmartnoise_ffi.so  
        inflating: opendp_smartnoise_core-0.2.1.dist-info/LICENSE  
        inflating: opendp_smartnoise_core-0.2.1.dist-info/METADATA  
        inflating: opendp_smartnoise_core-0.2.1.dist-info/WHEEL  
        inflating: opendp_smartnoise_core-0.2.1.dist-info/top_level.txt  
        inflating: opendp_smartnoise_core-0.2.1.dist-info/RECORD 
    ```
    Looks good! The two files appear ^
1. Delete the /tmp files
    ```
    rm /tmp/check_wheel.zip
    rm -rf /tmp/opendp
    rm -rf opendp_smartnoise_core*
    ```
1. Upload the wheel to PyPI!
    ```
    # You will be prompted for your credentials (unless they're set by env. variables, etc)
    #
    python -m twine upload --verbose --skip-existing wheelhouse/*
    ```
1. Check the url for your release: https://pypi.org/project/opendp-smartnoise-core/
