#!/usr/bin/python

import re
import sys
import operator
import math
from os.path import join

def main():
	args = sys.argv[1:]
	if len(args) != 2:
		print 'Usage: %s <path> <part>' % sys.argv[0]
		sys.exit(1)

	path = args[0]
	part = int(args[1])
	
	group_counts = {}
	for i in range(part):
		filepath = join(path, 'results_' + str(i) + '.txt')
		file = open(filepath, "r")

		imgs = {}
		for row in file:
			col = row.split()
			img1 = find_filename(col[0])
			imgs[img1] = True
		
		for img1 in imgs:
			group1 = int(find_group(img1))
			if not group_counts.get(group1):
				group_counts[group1] = 0	
			group_counts[group1] += 1

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
