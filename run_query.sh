#!/bin/bash

ulimit -s unlimited

dataset="art_50"
methods=( "scale_density" "response" "scale" "response_density" )
numbers=( -2 1 -1 2 )
gd="200"
fcount="40"
threshold="80"

for i in {0..3}; do
	echo "Dateset: $dataset, Method: ${methods[i]}, ${numbers[i]}"
	python query.py $threshold ~/near/$dataset/query_${methods[i]}_$fcount.txt > ~/near/$dataset/results_${methods[i]}_$fcount_$threshold.txt
done