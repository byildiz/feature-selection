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
	
	img_count = 0
	part_count = 0;
	recall_sum = [0.0]*101
	precision_sum = [0.0]*101
	for i in range(part):
		f = open(join(path, 'recall_' + str(i) + '.txt'), 'r')
		part_count = int(f.readline())
		img_count += part_count
		for j in range(101):
			col = f.readline().split()
			recall_sum[j] += float(col[0]) * part_count
			precision_sum[j] += float(col[1]) * part_count
	print img_count
	print '-'*40
	for i in range(len(recall_sum)):
		print recall_sum[i] / img_count, precision_sum[i] / img_count

if __name__ == '__main__':
	main()
