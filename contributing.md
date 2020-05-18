Contributing to WhiteNoise Python
=============================

Contributions to WhiteNoise are welcome from all members of the community. This document is here to simplify the 
on-boarding experience for contributors, contributions to this document are also welcome. 
Please use the [GitHub issue board](https://github.com/orgs/opendifferentialprivacy/projects/1) to track and take ownership of issues.  

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendifferentialprivacy/whitenoise-core-python/issues).
We appreciate all contributions. We welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new functionality to the python bindings, please first open an issue and 
discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might
 be taking the core in a different direction than you might be aware of.

For a description of the library architecture and installation instructions, see [whitenoise-core-python/README.md](./README.md). 


Contributing to the Bindings
=============================

The python library is a wrapper for the core library. 
The core library is in a submodule that points to a specific commit hash.  
The python library sources are located in [`opendp/whitenoise/core/`](./opendp/whitenoise/core/).

To install the library locally without any addons:

    pip3 install -e .

This will create a developer install of the package. A developer installation of a package updates itself automatically as you edit the code.

A significant portion of the python library is auto-generated based on resources in the core library. 
This includes the protobuf files, components.py, and native binaries. [More information on these files may be found below.](#non-editable-files) To update the non-editable code:

    python3 scripts/code_generation.py

It is not necessary to re-install the package to observe code changes (in either editable or non-editable files) in new interpreter instances.

## Editable files

These files may be found in [`opendp/whitenoise/core/`](./opendp/whitenoise/core/)

### [base.py](./opendp/whitenoise/core/base.py)
This file contains the public-facing api for the library. 
The two important pieces here are the Component class and Analysis class. 
When a Component is initialized, references are added to and from the component and analysis.
Each new component is assigned a component id. 
All releases and properties are centrally stored in the analysis. 
Helpers on components (like `.value`, `.releasable`, etc) reference data stored in the analysis.

### [value.py](./opendp/whitenoise/core/value.py)
This file contains helpers for converting between python data structures and protobuf data structures defined in the `*_pb2.py` files.
Edits are only necessary here if you need to convert between python representations of protobuf messages. 
For example, converting between the protobuf Array and `numpy.ndarray`. 
 
### [api.py](./opendp/whitenoise/core/api.py)
This file handles protobuf serialization/deserialization and ffi.
Python communicates with the core rust library via serialized protobuf over ffi.
When new endpoints are made available from the rust library, respective protobuf messages are added to `api.proto`, and `api.py` is updated respectively.
These functions have protobuf inputs and outputs represented by `_pb2.py`. 

## Non-editable files

Also found in [`opendp/whitenoise/core/`](./opendp/whitenoise/core/), these files should *not* be edited as they are auto-generated and manual changes will be overwritten.

### components.py

This file is a collection of functions for building instances of the Component class. 
  - It is auto-generated from the JSON files in `whitenoise-core/validator-rust/prototypes/components`.
  - To add a __new argument__ to a component, edit the respective JSON file. 
  - To add a __new component__, add a new JSON file, and follow the guidelines in [`whitenoise-core/contributing.md`](https://github.com/opendifferentialprivacy/whitenoise-core/blob/develop/contributing.md).
   - Changes will appear in [`components.py`](./opendp/whitenoise/core/components.py) when you rerun [`scripts/code_generation.py`](./scripts/code_generation.py).

### *_pb2.py

These files are a one-to-one translation of `whitenoise-core/validator-rust/prototypes/*.proto` to a python interface.
  - They provide an intermediate-level interface for constructing protobuf messages.
  - Higher-level tools for building protobuf messages are in `value.py`.
  - Changes in the `.proto` files will appear in the `*_pb2.py` files when you rerun `code_generation.py`.
  - [Tutorial for using `_pb2.py` files](https://developers.google.com/protocol-buffers/docs/pythontutorial) 

### lib/*
The `whitenoise-core/target/(release|debug)/` folder contains library artifacts from running `cargo build` on the `whitenoise-core` library.  Running `scripts/code_generation.py` copies these artifacts into the `lib/` folder.
You can use environment variables to customize the cargo command crafted by `code_generation.py`.  
    
Disable compile-time optimizations for quick builds:
    
    export WN_DEBUG=true

Use alternative installations of GMP and MPFR (high precision floating point libraries):

    export WN_USE_SYSTEM_LIBS=true

Since binaries are operating-system-specific, the relevant files are:
  - `*.so` for Linux,
  - `*.dylib` for OS X, and
  - `*.dll` for windows.
Changes in the Rust `whitenoise-core` library or from environment variables will appear in the `/lib` folder when you rerun [`scripts/code_generation.py`](./scripts/code_generation.py). 
