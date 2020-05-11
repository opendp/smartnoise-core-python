#!/bin/bash

# run from within the manylinux docker containers

# exit immediately upon failure, print commands while running
set -e -x


yum -y install openssl-devel

# install protoc if not exists
if ! [ -x "$(command -v protoc)" ]; then
  PROTOC_ZIP=protoc-$1-$2
	curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v$1/$PROTOC_ZIP
	unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
	unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
	rm -f $PROTOC_ZIP
fi

if ! [ -x "$(command -v cargo)" ]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
	export PATH="${HOME}/.cargo/bin:${PATH}"
fi

# compilation is only necessary for one generic version of python
# for PYBIN in /opt/python/cp3*/bin; do
#     "${PYBIN}/pip" install -r requirements.txt
#     "${PYBIN}/python" setup.py bdist_wheel -d /wheelhouse # ./ -w wheelhouse # bdist_wheel -d /wheelhouse
# done

# export makes the updated path available in subprocesses
export PATH="/opt/python/cp38-cp38/bin:$PATH"

pip install -r /io/requirements/base.txt

#python ../setup.py sdist -d /io/wheelhouse
python /io/scripts/code_generation.py
python /io/setup.py bdist_wheel -d /io/wheelhouse

for whl in /io/wheelhouse/*.whl; do
	auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/;
done;
