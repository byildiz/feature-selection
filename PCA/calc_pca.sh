#!/bin/sh

if [ $# -ne 1 ]; then
	echo "Usage: ./calc_pca.sh <img_path>"
	exit 0
fi

filename="${1%%.*}"
pgm="$filename.pgm"
lkeys="$filename.lkeys"
pkeys="$filename.pkeys"
ckeys="$filename.ckeys"

convert $1 $pgm
#./keypoints < $pgm > $lkeys
./cv_keypoints $pgm $ckeys
./recalckeys gpcavects.txt $pgm $ckeys $pkeys