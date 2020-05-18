## Whitenoise-Core Python Contributing

Please use the Github issue board to track and take ownership of issues.  
The python library is a wrapper for the core library. 
The core library is in a submodule that points to a specific commit hash.  

To install the library locally without any addons:

    pip3 install -e .

This will create a developer install of the package. A developer installation of a package updates itself automatically as you edit the code.

A significant portion of the python library is auto-generated based on resources in the core library. 
This includes the protobuf files, components.py, and native binaries. [More information here.](#non-editable-files)
To update the non-editable code:

    python3 scripts/code_generation.py

It is not necessary to re-install the package to observe code changes in new interpreter instances.

## Editable files

### opendp/whitenoise/core/base.py
This file contains the public-facing api for the library.
 
### opendp/whitenoise/core/value.py
This file contains helpers for converting between python data structures and protobuf data structures defined in the `*_pb2.py` files.
Edits are only necessary here if you need python representations of protobuf messages. 
For example, converting between the protobuf Array and numpy.ndarray. 
 
### opendp/whitenoise/core/api.py
This file handles protobuf serialization/deserization and ffi.
Python communicates with the core rust library via protobuf serialized over ffi.
When new endpoints are made available from the rust library, respective protobuf messages are added to `api.proto`, and `api.py` is updated respectively.
These functions have protobuf inputs and outputs represented by `_pb2.py`. 

## Non-editable files

### opendp/whitenoise/core/components.py
This file is auto-generated from the json files in `whitenoise-core/validator-rust/prototypes/components`.
To add a new argument to a component, edit the respective json file. 
To add a new component, add a new json file. 
The changes will appear in `components.py` when you rerun `code_generation.py`.

### opendp/whitenoise/core/*_pb2.py
These files are a one-to-one translation of `whitenoise-core/validator-rust/prototypes/*.proto` into a python api.
These files provide an intermediate-level api for constructing serializable messages in protobuf.
Additional high-level apis for building protobuf messages are in `value.py`.
Changes in the `.proto` files will appear in the `*_pb2.py` files when you rerun `code_generation.py`.

[Tutorial for using `_pb2.py` files](https://developers.google.com/protocol-buffers/docs/pythontutorial) 

### opendp/whitenoise/core/lib/.*
The `whitenoise-core/target/(release|debug)/` folder contains library binaries after running `cargo build` on the core library.
Since binaries are operating-system-specific, the relevant files are `*.so` for linux, `*.dylib` for osx, and `*.dll` for windows.
`code_generation.py` crafts the relevant cargo commands based on environment variables. 
You can use these to customize the cargo command.  
    
Disable compile-time optimizations for quick builds:
    
    export WN_DEBUG=true

Use alternative installations of GMP and MPFR (high precision floating point libraries):

    export WN_USE_SYSTEM_LIBS=true
    
Changes in the rust core library or from environment variables will appear in the `/lib` folder when you rerun `code_generation.py`. 
