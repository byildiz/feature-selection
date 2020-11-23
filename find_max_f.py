#!/usr/bin/python

import re
import sys
import operator
import math
from os.path import join
import numpy as np

def main():
	args = sys.argv[1:]
	if len(args) != 1:
		print 'Usage: %s <path>' % sys.argv[0]
		sys.exit(1)

	path = args[0]

	results = np.zeros((1001,3))
	f = open(path, 'r')
	print "alteration", "recall", "precision", "f-measure"
	for i in range(51):
		f.readline()
		f.readline()
		for j in range(1001):
			col = f.readline().split()
			results[j][0] = float(col[0])
			results[j][1] = float(col[1])
			results[j][2] = 2*results[j][0]*results[j][1]/(results[j][0]+results[j][1])
		m = max(results, key=lambda(x):x[2])
		print i, m[0], m[1], m[2]

if __name__ == '__main__':
	main()
