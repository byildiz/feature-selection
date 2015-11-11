#!/bin/bash

ulimit -s unlimited

dataset=$1
methods=( "scale_density" "response" "scale" "response_density" )
numbers=( -2 1 -1 2 )
gd="200"
fcount="40"

for i in {0..3}; do
	echo "Dateset: $dataset, Method: ${methods[i]}, ${numbers[i]}"
	build/fs $fcount ${numbers[i]} ~/near/$dataset ~/near/$dataset/query_${methods[i]}_$fcount.txt $gd | tee ~/near/$dataset/output_${methods[i]}.txt
done