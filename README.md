[**Please note that we are renaming the toolkit and will be introducing the new name in the coming weeks.**](https://projects.iq.harvard.edu/opendp/blog/building-inclusive-community)

[![Build Status](https://travis-ci.com/opendifferentialprivacy/whitenoise-core-python.svg?branch=develop)](https://travis-ci.com/opendifferentialprivacy/whitenoise-core-python)

<a href="https://opendifferentialprivacy.github.io"><img src="https://github.com/opendifferentialprivacy/whitenoise-core/blob/develop/images/WhiteNoise%20Logo/SVG/LogoMark_color.svg" align="left" height="65" vspace="8" hspace="18"></a>

## Core Differential Privacy Library Python Bindings <br/>

This repository contains python bindings to the [Core library](https://github.com/opendifferentialprivacy/whitenoise-core) and its underlying Rust binaries.

- For examples of this library in action, please see the Python notebooks in the [Samples repository](https://github.com/opendifferentialprivacy/whitenoise-samples).
- In addition, see the accompanying [System repository](https://github.com/opendifferentialprivacy/whitenoise-system) repository which includes tools for differential privacy.

---

Differential privacy is the gold standard definition of privacy protection. This project aims to connect theoretical solutions from the academic community with the practical lessons learned from real-world deployments, to make differential privacy broadly accessible to future deployments. Specifically, we provide several basic building blocks that can be used by people involved with sensitive data, with implementations based on vetted and mature differential privacy research. In the Core library, we provide a pluggable open source library of differentially private algorithms and mechanisms for releasing privacy preserving queries and statistics, as well as APIs for defining an analysis and a validator for evaluating these analyses and composing the total privacy loss on a dataset.

This library provides an easy-to-use interface for building analyses.

Differentially private computations are specified as a protobuf analysis graph that can be validated and executed to produce differentially private releases of data.


- [More about the Core Python Bindings](#more-about-core-python-bindings)
  - [Component List](#components)
  - [Architecture](#architecture)
- [Installation](#installation)
  - [Binaries](#binaries)
  - [From Source](#from-source)
- [Core Documentation](#core-documentation)
- [Communication](#communication)
- [Releases and Contributing](#releases-and-contributing)

---

## More about Core Python Bindings

### Components

For a full listing of the extensive set of components available in the library [see this documentation.](https://opendifferentialprivacy.github.io/whitenoise-core/doc/whitenoise_validator/docs/components/index.html)

### Architecture

The Core library system architecture [is described in the parent project](https://github.com/opendifferentialprivacy/whitenoise-core#Architecture).
This package is an instance of the language bindings. The purpose of the language bindings is to provide a straightforward programming interface to Python for building and releasing analyses.

Logic for determining if a component releases differentially private data, as well as the scaling of noise, property tracking, and accuracy estimates are handled by a native rust library called the Validator.
The actual execution of the components in the analysis is handled by a native Rust runtime.


## Installation
Refer to [troubleshooting.md](https://github.com/opendifferentialprivacy/whitenoise-core/blob/develop/troubleshooting.md) for install problems.

### Binaries

Initial Linux and OS X binaries are available on [pypi](https://pypi.org/project/opendp-whitenoise-core/) for Python 3.6+:
  - https://pypi.org/project/opendp-whitenoise-core/
  - ```pip3 install opendp-whitenoise```

The binaries have been used on OS X and Ubuntu and are in the process of additional testing.

### From Source

1. Clone the repository
    ```shell script
    git clone --recurse-submodules git@github.com:opendifferentialprivacy/whitenoise-core-python.git
    ```

    If you have already cloned the repository without the submodule
    ```shell script
    git submodule init
    git submodule update
    ```

2. Install the Core dependencies

    **Mac**
    ```shell script
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    xcode-select --install
    brew install protobuf python
    ```

    **Linux**
    ```shell script
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    sudo apt-get install diffutils gcc make m4 python
    # snap for protobuf 3, because apt comes with protobuf 2
    sudo snap install protobuf --classic
    ```

    **Windows**

    Install WSL and refer to the linux instructions.

3. Install live-reloading developer version of package
   ```shell script
   pip3 install -r requirements/dev.txt
   pip3 install -e .
   ```

4. Generate code (rerun anytime the Core changes)
    ```shell script
    python3 scripts/code_generation.py
    ```

5. Build documentation (optional)
    ```shell script
    ./scripts/build_docs.sh
    ```

### Core Documentation

- [Python library documentation](https://opendifferentialprivacy.github.io/whitenoise-core-python)


## Communication

- Please use [GitHub issues](https://github.com/opendifferentialprivacy/whitenoise-core-python/issues) for bug reports, feature requests, install issues, and ideas.
- [Gitter](https://gitter.im/opendifferentialprivacy/WhiteNoise) is available for general chat and online discussions.
- For other requests, please contact us at [whitenoise@opendp.io](mailto:whitenoise@opendp.io).
  - _Note: We encourage you to use [GitHub issues](https://github.com/opendifferentialprivacy/whitenoise-core-python/issues), especially for bugs._

## Releases and Contributing

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendifferentialprivacy/whitenoise-core-python/issues).

We appreciate all contributions and welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us.
  - Sending a pull request (PR) without discussion might end up resulting in a rejected PR, because we may be taking the core in a different direction than you might be aware of.

There is also a [contributing guide](contributing.md) for new developers.
