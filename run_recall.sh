#!/bin/bash

ulimit -s unlimited

dataset="art_50"
methods=( "scale_density" "response" "scale" "response_density" )
numbers=( -2 1 -1 2 )
gd="200"
fcount="40"
recall="95"

for i in {0..3}; do
	echo "Dateset: $dataset, Method: ${methods[i]}, ${numbers[i]}"
	python recall.py $recall ~/near/$dataset/query_${methods[i]}_$fcount.txt > ~/near/$dataset/recall_${methods[i]}_$fcount_$recall.txt
done