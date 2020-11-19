#set -e

# ensure the docker app is running
open -g -a Docker.app || exit
# Wait for the server to start up, if applicable.
i=0
while ! docker system info &>/dev/null; do
  (( i++ == 0 )) && printf %s '-- Waiting for Docker to finish starting up...' || printf '.'
  sleep 1
done
(( i )) && printf '\n'

# be sure to update paths within the adjacent .toml if you choose to run this
# python ./smartnoise-core/scripts/update_version.py

echo "(A) clean, delete all temporary directories";
bash scripts/clean.sh

echo "(B) set up manylinux docker";
DOCKER_IMAGE=quay.io/pypa/manylinux2010_x86_64
docker pull $DOCKER_IMAGE
chmod +x scripts/build_manylinux_binaries.sh


echo "(C) build manylinux binary";
docker run --rm -v `pwd`:/io $DOCKER_IMAGE /io/scripts/build_manylinux_binaries.sh

echo "(D) store binary in temp directory";
mkdir tmp_binaries
rm -f tmp_binaries/libsmartnoise_ffi.so
cp smartnoise-core/target/release/libsmartnoise_ffi.so tmp_binaries/libsmartnoise_ffi.so

# (1) check check for GLIBC_ <= ~2.3. Typically memcpy is an example that links to GLIBC ~2.15
#  - Outputs several versions. Look at versions <= 2.3
#  - Dump out link table to see if it's linking against older versions of GLIBC
#  - In the past, have uploaded libs not mentioned in the install
#
#  docker run --rm -v `pwd`:/io $DOCKER_IMAGE objdump -T /io/opendp/smartnoise/core/lib/libsmartnoise_ffi.so | grep GLIBC_
#  
# (2) Check that all necessary libraries are statically linked (look for non-existence of gmp/mpfr/mpc/openssl)
#
# docker run --rm -v `pwd`:/io $DOCKER_IMAGE ldd /io/opendp/smartnoise/core/lib/libsmartnoise_ffi.so
#

echo "(E) mac binaries/packaging";
export WN_USE_SYSTEM_LIBS=false;
export WN_DEBUG=false;
export WN_USE_VULNERABLE_NOISE=false;
python3 scripts/code_generation.py

# (3) check that all necessary libraries are statically linked (look for non-existence of gmp/mpfr/mpc/openssl)
# otool -L opendp/smartnoise/core/lib/libsmartnoise_ffi.dylib
#

echo "(F) move prior manylinux binary into the library";
cp tmp_binaries/libsmartnoise_ffi.so opendp/smartnoise/core/lib
echo "(G) move externally-built windows .dll into the library";
cp smartnoise_ffi.dll opendp/smartnoise/core/lib

echo "(H) package into wheel";
#workon psi
python3 setup.py bdist_wheel -d ./wheelhouse
