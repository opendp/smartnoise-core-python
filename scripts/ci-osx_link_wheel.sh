#
# Use delocate to include library dependencies
# ref: https://pypi.org/project/delocate/
# 4/30/2020
#
#
for awheel in ./wheelhouse/*.whl; do  # loop through the .whl files
  [ -e "$awheel" ] || continue  # (no output if nothing is found)
  echo $awheel                  # found a .whl file
  delocate-wheel -v $awheel     # run delocate
done

# Remove .zip files 
#
rm ./wheelhouse/*.zip
