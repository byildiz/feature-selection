#!/usr/bin/python

import sys
import subprocess
import errno
from os import listdir, rename, makedirs, walk, remove
from os.path import isfile, join, isdir
from random import randrange
from shutil import copyfile

def main():
	if len(sys.argv) != 3:
		print "Usage: %s <path> <part>" % sys.argv[0]
		exit()

	path = sys.argv[1]
	part = int(sys.argv[2])

	img_paths = []
	org_paths = []
	for dirpath, dirnames, filenames in walk(path):
		for filename in filenames:
			if (filename.endswith(".png")):
				img_path = join(dirpath, filename)
				img_paths.append(img_path)
				if (filename.endswith('_0.png')):
					org_paths.append(img_path)

	img_paths = sorted(img_paths)
	org_paths = sorted(org_paths)

	img_count = len(img_paths)
	part_count = img_count / part
	remaining_cout = img_count % part
	index = 0;
	for i in range(part):
		images_filename = join(path, 'images_' + str(i) + '.txt')
		images_file = open(images_filename, 'w')
		if remaining_cout > 0:
			images_file.write(str(part_count+1) + '\n')
			images_file.write(img_paths[index] + '\n')
			index += 1
			remaining_cout -= 1
		else:
			images_file.write(str(part_count) + '\n')
		for j in range(part_count):
			images_file.write(img_paths[index] + '\n')
			index += 1
		images_file.close()

	img_count = len(org_paths)
	part_count = img_count / part
	remaining_cout = img_count % part
	index = 0;
	for i in range(part):
		images_filename = join(path, 'originals_' + str(i) + '.txt')
		images_file = open(images_filename, 'w')
		if remaining_cout > 0:
			images_file.write(str(part_count+1) + '\n')
			images_file.write(org_paths[index] + '\n')
			index += 1
			remaining_cout -= 1
		else:
			images_file.write(str(part_count) + '\n')
		for j in range(part_count):
			images_file.write(org_paths[index] + '\n')
			index += 1
		images_file.close()

if __name__ == "__main__":
	main()
