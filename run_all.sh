#!/bin/bash

ulimit -s unlimited

dataset=$1
part=$2
gd="350"

partsfile=$dataset/parts.txt
for i in $(seq 0 $((part-1))); do echo $i >> $partsfile; done
echo "Dividing..."
./divide.py $dataset $part

echo "Dateset: $dataset, Method: all, 0"
path=$dataset/all
mkdir $path
echo "Indexing..."
cat $partsfile | parallel "build/build_index 0 0 $dataset/images_{}.txt $path/index_{}.yml.gz $gd > $path/build_output_{}.txt"
echo "Merging indicies..."
build/merge_index $path $path/index_all.yml.gz $part > $path/merge_output.txt
if [[ $# -ne 3 ]]; then
	echo "Querying all images..."
	cat $partsfile | parallel "build/query_index $path/index_all.yml.gz $dataset/images_{}.txt $path/results_{}.txt > $path/query_output_{}.txt"
else
	echo "Querying original images..."
	cat $partsfile | parallel "build/query_index $path/index_all.yml.gz $dataset/originals_{}.txt $path/results_{}.txt > $path/query_output_{}.txt"
fi
# echo "Merging results..."
# find $path -name "results_*.txt" -exec cat {} \; >> $path/query_all.txt
echo "Counting groups..."
./count_groups.py $path > $path/groups.txt
echo "Calcing recall-precision..."
cat $partsfile | parallel "./calc_recall.py $path/results_{}.txt $path/groups.txt > $path/recall_{}.txt"
echo "Merging recall-precision..."
./merge_recall.py $path $part > $path.txt