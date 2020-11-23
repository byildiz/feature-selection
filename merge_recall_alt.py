#!/usr/bin/python

import re
import sys
import operator
import math
from os.path import join
import numpy as np

def main():
	args = sys.argv[1:]
	if len(args) != 2:
		print 'Usage: %s <path> <part>' % sys.argv[0]
		sys.exit(1)

	path = args[0]
	part = int(args[1])

	img_counts = np.zeros(51)
	# recall_sum = [0.0]*1001
	recall_sum = np.zeros((51, 1001))
	# precision_sum = [0.0]*1001
	precision_sum = np.zeros((51, 1001))
	for i in range(part):
		f = open(join(path, 'recall_' + str(i) + '.txt'), 'r')
		for k in range(51):
			img_counts[k] += float(f.readline())
		for k in range(51):
			for j in range(1001):
				col = f.readline().split()
				recall_sum[k][j] += float(col[0])
				precision_sum[k][j] += float(col[1])
	
	print img_counts
	for i in range(len(recall_sum)):
		print '-'*40
		print "ALTERATION " + str(i)
		for j in range(len(recall_sum[i])):
			print recall_sum[i][j] / img_counts[i], precision_sum[i][j] / img_counts[i]

if __name__ == '__main__':
	main()
