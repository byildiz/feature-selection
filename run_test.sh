#!/bin/bash

ulimit -s unlimited

dataset=$1
fcount=$2
part=$3
#methods=( "scale_kdtree" "scale" "response" "scale_density" "response_density" "response_kdtree" "scale_substract" )
#numbers=( -3 -1 1 -2 2 3 -4 )
methods=( "scale_kdtree" "response" )
numbers=( -3 1 )
gd="350"

partsfile=$dataset/parts.txt
rm $partsfile
for i in $(seq 0 $((part-1))); do echo $i >> $partsfile; done
echo "Dividing..."
./divide.py $dataset $part

for i in $(seq 0 1); do
	echo "Dateset: $dataset, Method: ${methods[i]}, ${numbers[i]}"
	path=$dataset/${methods[i]}_$fcount
	mkdir $path
	echo "Indexing..."
	cat $partsfile | parallel "build/build_index $fcount ${numbers[i]} $dataset/images_{}.txt $path/index_{}.yml.gz $gd > $path/build_output_{}.txt"
	echo "Merging indicies..."
	build/merge_index $path $path/index_${methods[i]}_$fcount.yml.gz $part > $path/merge_output.txt
	if [[ $# -ne 4 ]]; then
		echo "Querying all images..."
		cat $partsfile | parallel "build/query_index $path/index_${methods[i]}_$fcount.yml.gz $dataset/images_{}.txt $path/results_{}.txt > $path/query_output_{}.txt"
	else
		echo "Querying original images..."
		cat $partsfile | parallel "build/query_index $path/index_${methods[i]}_$fcount.yml.gz $dataset/originals_{}.txt $path/results_{}.txt > $path/query_output_{}.txt"
	fi
	# echo "Merging results..."
	# find $path -name "results_*.txt" -exec cat {} \; >> $path/query_${methods[i]}_$fcount.txt
	echo "Counting groups..."
	./count_groups.py $path > $path/groups.txt
	echo "Calcing recall-precision..."
	cat $partsfile | parallel "./calc_recall.py $path/results_{}.txt $path/groups.txt > $path/recall_{}.txt"
	echo "Merging recall-precision..."
	./merge_recall.py $path $part > $path.txt
done