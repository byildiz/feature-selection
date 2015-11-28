#!/usr/bin/python

import re
import sys
import operator
import math
from os.path import join

def main():
	args = sys.argv[1:]
	if len(args) != 1:
		print 'Usage: %s <path>' % sys.argv[0]
		sys.exit(1)

	path = args[0]
	
	group_counts = {}
	filepath = join(path, 'results_0.txt')
	file = open(filepath, "r")

	imgs = {}
	for row in file:
		col = row.split()
		img2 = find_filename(col[1])
		if not imgs.get(img2):
			group2 = int(find_group(img2))
			if not group_counts.get(group2):
				group_counts[group2] = 0	
			group_counts[group2] += 1
			imgs[img2] = True

	print len(group_counts)
	for i in group_counts:
		print i, group_counts[i]

def find_filename(path):
	regex = "^.*/(.*)\..*$"
	return re.findall(regex, path)[0]

def find_group(path):
	regex = "^([\d]+)_.*$"
	return re.findall(regex, path)[0]

def find_variation(path):
	regex = "^.*_([\d]+)$"
	return re.findall(regex, path)[0]

if __name__ == '__main__':
	main()
