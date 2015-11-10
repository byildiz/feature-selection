#!usr/bin/python

#!/usr/bin/env python2

import sys
import subprocess
import errno
from os import listdir, rename, makedirs, walk, remove
from os.path import isfile, join, isdir
from random import randrange
from shutil import copyfile

def main():
	if len(sys.argv) != 3:
		print "Usage: %s <path> <config_file>" % sys.argv[0]
		exit()

	path = sys.argv[1]
	config_file = sys.argv[2]

	img_paths = []
	for dirpath, dirnames, filenames in walk(path):
		for filename in filenames:
			if (filename.endswith(".png")):
				img_path = join(dirpath, filename)
				img_paths.append(img_path)
	
	for i in range(len(img_paths)):
		for j in range(i, len(img_paths)):
			print "./fs_parallel", config_file, img_paths[i], img_paths[j]

if __name__ == "__main__":
	main()