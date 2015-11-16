#!/bin/bash

ulimit -s unlimited

dataset=$1
methods=( "scale_density" "response" "scale" "response_density" )
numbers=( -2 1 -1 2 )
gd="200"
fcount="40"

for i in {0..3}; do
	echo "Dateset: $dataset, Method: ${methods[i]}, ${numbers[i]}"
	if [ "$2" == "th" ]; then
		python recall_th.py ~/near/$dataset/query_${methods[i]}_$fcount.txt > ~/near/$dataset/recall_th_${methods[i]}_$fcount_$recall.txt
	elif [ "$2" == "org" ]; then
		python recall_th.py ~/near/$dataset/query_${methods[i]}_$fcount.txt 1 > ~/near/$dataset/recall_org_${methods[i]}_$fcount_$recall.txt
	else
		python recall.py ~/near/$dataset/query_${methods[i]}_$fcount.txt > ~/near/$dataset/recall_${methods[i]}_$fcount_$recall.txt
	fi
done