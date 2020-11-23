#!/bin/bash

ulimit -s unlimited

dataset=$1
part=$2
method=$3

partsfile=$dataset/parts.txt
for i in $(seq 0 $((part-1))); do echo $i >> $partsfile; done

echo "Dateset: $dataset, Method: $method, 0"
path=$dataset/$method
echo "Counting groups..."
./count_groups.py $path > $path/groups.txt
echo "Calcing recall-precision..."
cat $partsfile | parallel "./calc_recall_alt.py $path/results_{}.txt $path/groups.txt > $path/recall_{}.txt"
echo "Merging recall-precision..."
./merge_recall_alt.py $path $part > $path.txt