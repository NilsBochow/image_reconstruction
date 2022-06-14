#!/bin/bash
# ALPHA Version
# Little Helper to come from pytorch output to NETCDF [Changed in the future]
# USAGE EXAMPLE
# in h5 dir, bash script/h5toNC.sh image
# change link and data if necessary
input=$1
rdir=$(dirname $input)
rname=$(basename $input)
output=$rdir/$rname.nc

cat script/netcdf_structure.txt >> $rdir/$rname.txt
ncdump -v sic $input | sed -e '1,/data:/d' >> $rdir/$rname.txt
ncgen -o $output-tmp $rdir/$rname.txt

cdo -remapcon,../reconstructions/test_sic.nc $output-tmp $output

rm $rdir/$rname.txt $output-tmp
